import math
import sys
import omegaconf
import torch_scatter
import trimesh
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import hydra
from jaxtyping import Float
from lightning_utilities.core.rank_zero import rank_zero_only
from pathlib import Path
import torch
from torch import Tensor
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from dataset import quantize_coordinates
from dataset.triangles import create_feature_stack_from_triangles, TriangleNodesWithFaces, TriangleNodesWithFacesDataloader
from model.encoder import GraphEncoder
from model.decoder import resnet34_decoder
from model.softargmax import softargmax
from trainer import create_trainer, step, create_conv_batch
from util.visualization import plot_vertices_and_faces, triangle_sequence_to_mesh
from util.misc import get_parameters_from_state_dict
from util.mesh_chamfer import batched_mesh_chamfer
from util.aux_losses import code_classification_loss, frozen_codebook, accumulate_grads
from util.vertex_vote import vertex_vote_logits
from external.shape2vecset.models_ae import (
    create_explicit_autoencoder,
    create_order_preserving_explicit_autoencoder,
    create_flow_explicit_autoencoder,
)


class TriangleTokenizationGraphConv(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.has_logger = True if self.config.logger else False
        self.save_hyperparameters(omegaconf.OmegaConf.to_container(config, resolve=True))
        if config.dataset == 'thingi10k':
            if config.only_chairs:
                forced_category = {'category': '3d-printing', 'tags': 'gear'}
                self.train_dataset = TriangleNodesWithFaces(config, 'train', config.scale_augment, config.shift_augment, forced_category)
                self.val_datasets = [TriangleNodesWithFaces(config, 'val', config.scale_augment, config.shift_augment, forced_category)]
                self.interesting_categories = [(forced_category['category'], "")]
            else:
                self.train_dataset = TriangleNodesWithFaces(config, 'train', config.scale_augment, config.shift_augment, None)
                self.val_datasets = [TriangleNodesWithFaces(config, 'val', config.scale_augment, config.shift_augment, None)]
                self.interesting_categories = [("", "")]

        else:  # shapenet
            if config.only_chairs:
                self.train_dataset = TriangleNodesWithFaces(config, 'train', config.scale_augment, config.shift_augment, '03001627')
                self.interesting_categories = [('03001627', "")]
                self.val_datasets = [TriangleNodesWithFaces(config, 'val', config.scale_augment_val, config.shift_augment_val, '03001627')]
            else:
                self.train_dataset = TriangleNodesWithFaces(config, 'train', config.scale_augment, config.shift_augment, None)
                self.interesting_categories = [('02828884', '_bench'), ('02871439', '_bookshelf'), ('03001627', ""), ('03211117', '_display'), ('04379243', '_table')]
                self.val_datasets = []
                for cat, name in self.interesting_categories:
                    self.val_datasets.append(TriangleNodesWithFaces(config, 'val', config.scale_augment_val, config.shift_augment_val, cat))
        self.encoder = GraphEncoder(no_max_pool=config.g_no_max_pool, aggr=config.g_aggr, graph_conv=config.graph_conv, use_point_features=config.use_point_feats, output_dim=config.per_vertex_embed_dim * 3)
        self.decoder = resnet34_decoder(512, config.num_tokens - 2, config.ce_output)
        self.pre_quant = torch.nn.Linear(config.per_vertex_embed_dim, config.embed_dim)
        self.post_quant = torch.nn.Linear(config.embed_dim * 3, 512)
        self.vq = ResidualVQ(
            dim=self.config.embed_dim,
            codebook_size=self.config.n_embed,  # codebook size
            num_quantizers=config.embed_levels,
            commitment_weight=self.config.embed_loss_weight,  # the weight on the commitment loss
            stochastic_sample_codes=True,
            sample_codebook_temp=config.stochasticity,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook=self.config.embed_share,
            decay=self.config.code_decay,
        )
        self.register_buffer('smoothing_weight', torch.tensor([2, 10, 200, 10, 2], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        # print('compiling model...')
        # self.model = torch.compile(model)  # requires PyTorch 2.0
        output_dir_prefix = self.config.output_dir_prefix if 'output_dir_prefix' in self.config else ''
        self.output_dir_image_val = Path(f'{output_dir_prefix}runs/{self.config.experiment}/image_val')
        self.output_dir_image_val.mkdir(exist_ok=True, parents=True)
        self.output_dir_mesh_val = Path(f'{output_dir_prefix}runs/{self.config.experiment}/mesh_val')
        self.output_dir_mesh_val.mkdir(exist_ok=True, parents=True)
        self.output_dir_image_train = Path(f'{output_dir_prefix}runs/{self.config.experiment}/image_train')
        self.output_dir_image_train.mkdir(exist_ok=True, parents=True)
        self.output_dir_mesh_train = Path(f'{output_dir_prefix}runs/{self.config.experiment}/mesh_train')
        self.output_dir_mesh_train.mkdir(exist_ok=True, parents=True)
        self.output_dir_metrics_val = Path(f'{output_dir_prefix}runs/{self.config.experiment}/metrics_val')
        self.output_dir_metrics_val.mkdir(exist_ok=True, parents=True)
        self.automatic_optimization = False
        self.distribute_features_fn = distribute_features if self.config.distribute_features else dummy_distribute

        # Load pretrained VQ components if specified (before creating shape2vecset_ae)
        if config.vq_resume is not None:
            self._load_pretrained_vq_components(config.vq_resume)

        # Init 3dshape2vecset autoencoder
        self.shape2vecset_ae = None
        if self.config.use_sequence_distillation:
            # Use raw cached face counts (before quantization/augmentation) to avoid
            # variability from random augmentation. The sort_vertices_and_faces function
            # can split faces into sub-faces, so we add a safety margin.
            max_num_faces_raw = max([len(self.train_dataset.cached_faces[i]) for i in range(len(self.train_dataset))])
            # Add 50% safety margin to account for face splitting during quantization
            # max_num_faces = int(max_num_faces_raw * 1.5)
            max_num_faces = max_num_faces_raw
            print(f"Max raw faces: {max_num_faces_raw}, with safety margin: {max_num_faces}")

            # Choose the autoencoder family:
            #   ae_type == "flow"     -> flow-matching (generative) decoder
            #   use_order_preserving  -> regression w/ positional decoder queries
            #   otherwise             -> regression w/ learnable decoder queries
            if getattr(config, 'ae_type', 'regression') == 'flow':
                ae_factory = create_flow_explicit_autoencoder
            elif config.use_order_preserving_ae:
                ae_factory = create_order_preserving_explicit_autoencoder
            else:
                ae_factory = create_explicit_autoencoder
            ae_extra_kwargs = {}
            if getattr(config, 'query_binding', 'none') != 'none':
                # the order-preserving regression AE and the flow AE implement the binding
                assert ae_factory in (create_order_preserving_explicit_autoencoder, create_flow_explicit_autoencoder), \
                    "query_binding needs use_order_preserving_ae=true (regression) or ae_type=flow"
                ae_extra_kwargs['query_binding'] = config.query_binding
            if int(getattr(config, 'latent_quant_bits', 0) or 0) > 0:
                assert ae_factory is create_order_preserving_explicit_autoencoder, "latent_quant_bits needs the order-preserving regression AE"
                ae_extra_kwargs['latent_quant_bits'] = int(config.latent_quant_bits)
                ae_extra_kwargs['latent_quant_train'] = bool(getattr(config, 'latent_quant_train', False))
            if getattr(config, 'ae_type', 'regression') == 'flow':
                ae_extra_kwargs.update({
                    'flow_decoder_depth': getattr(config, 'flow_decoder_depth', 8),
                    'flow_steps': getattr(config, 'flow_steps', 20),
                    'flow_t_sampling': getattr(config, 'flow_t_sampling', 'logit_normal'),
                    'flow_conditioning': getattr(config, 'flow_conditioning', 'cross_attn'),
                    'flow_mode_scale': getattr(config, 'flow_mode_scale', 0.25),
                    'flow_null_prob': getattr(config, 'flow_null_prob', 0.0),
                    'flow_guidance': getattr(config, 'flow_guidance', 1.0),
                    'repa_weight': getattr(config, 'repa_weight', 0.0),
                    'repa_layer': getattr(config, 'repa_layer', 2),
                })
            self.shape2vecset_ae = ae_factory(
                dim=self.config.per_vertex_embed_dim * 3,  # dimension of distilled latents
                M=self.config.num_sequence_tokens,  # num_latents (number of distilled latents)
                num_output_points=max_num_faces,
                latent_dim=64,  # KL-regularized latent dimension, not used if deterministic
                N=max_num_faces,  # num_inputs (number of input points/faces in point cloud)
                determinisitc=True,
                point_dim=self.config.per_vertex_embed_dim * 3,  # dimension face latents
                use_point_embed=False,
                use_nested_dropout=config.use_nested_dropout,
                nested_dropout_size_sampling=getattr(config, 'nested_dropout_size_sampling', 'uniform'),
                use_causal_attention=config.use_causal_attention,
                n_decoder_self_attn_blocks=config.n_decoder_self_attn_blocks,
                n_encoder_self_attn_blocks=getattr(config, 'n_encoder_self_attn_blocks', 0),
                decoder_use_rope=getattr(config, 'decoder_use_rope', getattr(config, 'use_rope', False)),
                encoder_use_rope=getattr(config, 'encoder_use_rope', False),
                encoder_use_pos_embed=getattr(config, 'encoder_use_pos_embed', False),
                varlen_mode=getattr(config, 'varlen_mode', 'nested_dropout'),
                use_rank_pe_latents=getattr(config, 'use_rank_pe_latents', False),
                num_sink_tokens=getattr(config, 'num_sink_tokens', 0),
                ae_cross_heads=getattr(config, 'ae_cross_heads', 1),
                **ae_extra_kwargs,
            )

            # Zero-init the residual branches of the encoder self-attn blocks so
            # they start as identity (ReZero-style). Without this, freshly
            # initialized encoder blocks inserted into an otherwise-pretrained
            # pipeline (vq_resume) feed garbage features to the RVQ early in
            # training, collapsing the EMA codebook irrecoverably (observed:
            # embed_loss 0.55 -> 0.13, val acc 0.77 -> 0.24 with 4 blocks).
            if getattr(config, 'encoder_zero_init', False):
                for attn_prenorm, ff_prenorm in self.shape2vecset_ae.encoder_layers:
                    torch.nn.init.zeros_(attn_prenorm.fn.to_out.weight)
                    torch.nn.init.zeros_(attn_prenorm.fn.to_out.bias)
                    torch.nn.init.zeros_(ff_prenorm.fn.net[2].weight)
                    torch.nn.init.zeros_(ff_prenorm.fn.net[2].bias)
                print(f"Zero-initialized residual branches of "
                      f"{len(self.shape2vecset_ae.encoder_layers)} encoder self-attn blocks")

            # Same trick for the decoder's latent self-attn blocks (grown stacks: the
            # checkpoint overwrites the old blocks, the new ones stay identity).
            if getattr(config, 'decoder_zero_init', False):
                for attn_prenorm, ff_prenorm in self.shape2vecset_ae.layers:
                    torch.nn.init.zeros_(attn_prenorm.fn.to_out.weight)
                    torch.nn.init.zeros_(attn_prenorm.fn.to_out.bias)
                    torch.nn.init.zeros_(ff_prenorm.fn.net[2].weight)
                    torch.nn.init.zeros_(ff_prenorm.fn.net[2].bias)
                print(f"Zero-initialized residual branches of "
                      f"{len(self.shape2vecset_ae.layers)} decoder self-attn blocks")
            # Fine-tuning from one of OUR checkpoints: also restore the AE.
            # (749-0 and other base-only checkpoints have no shape2vecset_ae
            # keys, so this is a no-op for them.) Runs after zero-init so
            # loaded weights win.
            if config.vq_resume is not None:
                self._load_pretrained_ae_components(config.vq_resume)

        # Freeze the base MeshGPT components (train only the shape2vecset AE).
        # Required for ae_type == "flow": the downstream CE loss cannot backprop
        # through ODE sampling, so the flow AE trains on flow_loss alone against
        # a frozen base. Also usable with regression + ae_only_loss for the
        # matched control arm.
        self.freeze_base = getattr(config, 'freeze_base', False)
        self.ae_only_loss = getattr(config, 'ae_only_loss', False)
        # split_grad: per step, a full-rate pass trains everything and a second
        # pass at the AE's sampled k trains only the AE (base and codebook
        # untouched). code_cls_weight: cross-entropy over codebook entries
        # between the AE's corner features and the clean features' codes.
        self.split_grad = getattr(config, 'split_grad', False)
        self.code_cls_weight = getattr(config, 'code_cls_weight', 0.0)
        # Curriculum (the image-side recipe): for the first N optimizer steps only the
        # latent-side losses train the AE (feature MSE, codebook CE, commitment); the
        # coordinate CE and geometry terms are switched on afterwards.
        self.curriculum_latent_steps = getattr(config, 'curriculum_latent_steps', 0)
        if self.split_grad:
            assert getattr(config, 'distill_before_vq', False) and config.use_sequence_distillation, "split_grad needs the AE before the VQ"
        self.ae_is_flow = getattr(config, 'ae_type', 'regression') == 'flow'
        if self.ae_is_flow:
            assert self.freeze_base, "ae_type=flow requires freeze_base=true (CE cannot backprop through ODE sampling)"
            assert self.shape2vecset_ae is not None, "ae_type=flow requires use_sequence_distillation=true"
        if self.freeze_base:
            assert config.vq_resume is not None, "freeze_base requires vq_resume (a pretrained base checkpoint)"
            for module in (self.encoder, self.decoder, self.pre_quant, self.post_quant, self.vq):
                for p in module.parameters():
                    p.requires_grad_(False)
            print("Base MeshGPT components frozen (encoder/decoder/pre_quant/post_quant/vq)")
        # freeze_codebook: with freeze_base=false, keep the RVQ codebook fixed while the
        # rest of the base trains. The EMA update has no learning-rate knob and, with
        # distill_before_vq, learns from the AE's (mostly truncated) outputs; the
        # buffers are restored after every training pass instead (see training_step).
        self.freeze_codebook = getattr(config, 'freeze_codebook', False)
        if self.freeze_codebook and not self.freeze_base:
            for p in self.vq.parameters():
                p.requires_grad_(False)
            print("Codebook frozen (RVQ buffers restored after every training pass); rest of the base trains")

        # Compute keep_k values for validation/visualization once.
        # Sweep when the AE is variable-length: mask-token nested dropout OR
        # true prefix truncation.
        varlen_active = (self.config.use_nested_dropout
                         or getattr(config, 'varlen_mode', 'nested_dropout') == 'prefix_truncation')
        if self.shape2vecset_ae is None or not varlen_active:
            self.val_keep_k_values: tuple[int | None, ...] = (None,)
        else:
            num_exp = int(math.log2(self.config.num_sequence_tokens)) + 1
            self.val_keep_k_values = tuple(2 ** i for i in range(num_exp))
            # Always include the full latent count, so a non-power-of-two M (e.g. 767)
            # logs its full-rate point (val/*/k767) next to the powers of two.
            if self.config.num_sequence_tokens not in self.val_keep_k_values:
                self.val_keep_k_values += (int(self.config.num_sequence_tokens),)

        self.visualize_groundtruth()

    def _log_figure_to_logger(
        self,
        key: str,
        content: plt.Figure | np.ndarray,
        step: int | None = None,
    ) -> None:
        """Low-level logging implementation for wandb or tensorboard.

        Args:
            key: The logging key/tag
            content: Content to log (numpy array or plt.Figure)
            step: Optional step number (defaults to global_step)
        """
        if not self.logger:
            return

        if step is None:
            step = self.global_step

        if isinstance(self.logger, WandbLogger):
            import wandb
            # Handle both raw content and pre-wrapped wandb.Image
            if not isinstance(content, wandb.Image):
                content = wandb.Image(content)
            self.logger.experiment.log({key: content}, step=step)
        elif isinstance(self.logger, TensorBoardLogger):
            # TensorBoard handles figures and images differently
            if isinstance(content, plt.Figure):
                self.logger.experiment.add_figure(key, content, global_step=step)
            elif isinstance(content, np.ndarray):
                # TensorBoard expects (C, H, W) format
                image_tensor = torch.from_numpy(content).permute(2, 0, 1).float() / 255.0
                self.logger.experiment.add_image(key, image_tensor, global_step=step)
            else:
                raise ValueError(f"Unsupported content type for TensorBoard: {type(content)}")

    def _create_and_log_image_grid(
        self,
        image_paths: list[Path | None],
        key: str,
        step: int | None = None,
        chamfer_plot: np.ndarray | None = None,
    ) -> None:
        """Create a 2-row image grid from saved image files and log to wandb/tensorboard.

        Args:
            image_paths: List of paths to images (or None for blank images). First should be GT, rest ordered by keep_k.
            key: The logging key/tag
            step: Optional step number (defaults to global_step)
            chamfer_plot: Optional numpy array (H, W, 3) of a chamfer distance vs keep_k plot to append below the grid.
        """
        if not self.logger or len(image_paths) == 0:
            return

        # Load images, using blank white images for None entries
        images = []
        default_size = None

        for path in image_paths:
            if path is not None and path.exists():
                img = np.array(Image.open(path))
                images.append(img)
                if default_size is None:
                    default_size = img.shape[:2]  # (height, width)
            else:
                # Create blank white image
                if default_size is None:
                    default_size = (800, 800)  # Default size if no valid images yet
                blank_img = np.ones((*default_size, 3), dtype=np.uint8) * 255
                images.append(blank_img)

        # Determine grid dimensions (max 5 columns per row)
        max_cols = 5
        num_images = len(images)

        # Split images into rows of max 5 columns each
        rows = []
        for i in range(0, num_images, max_cols):
            row_images = images[i:i + max_cols]
            rows.append(row_images)

        # Concatenate each row horizontally
        row_imgs = []
        for row in rows:
            if len(row) > 0:
                row_img = np.concatenate(row, axis=1)
                row_imgs.append(row_img)

        # Pad rows to have same width (use width of first row as reference)
        if len(row_imgs) > 1:
            max_width = row_imgs[0].shape[1]
            for i in range(1, len(row_imgs)):
                if row_imgs[i].shape[1] < max_width:
                    h = row_imgs[i].shape[0]
                    padding_width = max_width - row_imgs[i].shape[1]
                    padding = np.ones((h, padding_width, 3), dtype=np.uint8) * 255
                    row_imgs[i] = np.concatenate([row_imgs[i], padding], axis=1)

        # Concatenate all rows vertically
        grid = np.concatenate(row_imgs, axis=0) if len(row_imgs) > 0 else images[0]

        # Append chamfer distance plot below the image grid if provided
        if chamfer_plot is not None:
            grid_width = grid.shape[1]
            plot_h, plot_w = chamfer_plot.shape[:2]
            # Resize plot to match grid width
            if plot_w != grid_width:
                from PIL import Image as PILImage
                plot_img = PILImage.fromarray(chamfer_plot)
                new_h = int(plot_h * grid_width / plot_w)
                plot_img = plot_img.resize((grid_width, new_h), PILImage.LANCZOS)
                chamfer_plot = np.array(plot_img)
            grid = np.concatenate([grid, chamfer_plot], axis=0)

        # Log to appropriate logger
        self._log_figure_to_logger(key, grid)

    def configure_optimizers(self):
        if self.freeze_base:
            parameters = list(self.shape2vecset_ae.parameters())
        else:
            parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.pre_quant.parameters()) + list(self.post_quant.parameters()) + list(self.vq.parameters())
            parameters += list(self.shape2vecset_ae.parameters()) if self.shape2vecset_ae else []
        parameters = [p for p in parameters if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.config.lr, amsgrad=True, weight_decay=self.config.weight_decay)
        max_steps = int(self.config.max_epoch * len(self.train_dataset) / (self.config.batch_size * self.config.gradient_accumulation_steps))
        print('Max Steps | First cycle:', max_steps)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=max_steps, cycle_mult=1.0,
            max_lr=self.config.lr, min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps, gamma=1.0
        )
        return [optimizer], [scheduler]

    def create_conv_batch(self, encoded_features, batch, batch_size):
        return create_conv_batch(encoded_features, batch, batch_size, self.device)

    def _training_step_ae_only(self, data, batch_idx):
        """Frozen-base training of the shape2vecset AE alone (E1 arms).

        flow AE:       rectified-flow velocity MSE on the pre-VQ face features.
        regression AE: masked MSE reconstruction of the same features -- the
                       exactly matched control (same target, same frozen base).
        The full pipeline (VQ -> conv decoder -> CE) is skipped during training
        and still runs in validation_step for comparable metrics.
        """
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # Frozen encoder must stay in eval mode: its BatchNorm layers would
        # otherwise update running stats during the forward pass.
        self.encoder.eval()
        with torch.no_grad():
            encoded_x = self.encoder(data.x, data.edge_index, data.batch)  # num_faces x (192 * 3)
        # Pad to the AE's fixed num_output_points (dataset-wide max face count),
        # not the batch max: the decoder always emits num_output_points slots
        # and the loss compares position-for-position.
        packed, num_faces = pack_faces_into_batch(
            encoded_x, data.batch, self.config.batch_size,
            max_num_faces=self.shape2vecset_ae.num_output_points)

        if self.ae_is_flow:
            loss = self.shape2vecset_ae.flow_loss(packed, num_faces=num_faces)
            repa = getattr(self.shape2vecset_ae, 'last_repa_loss', None)
            if repa is not None:
                self.log("train/repa_loss", repa.item(), on_step=True, on_epoch=False,
                         prog_bar=False, logger=self.has_logger, sync_dist=True,
                         batch_size=self.config.batch_size)
        else:
            recon = self.shape2vecset_ae(packed, num_faces=num_faces)['points']
            mask = (torch.arange(packed.shape[1], device=packed.device).unsqueeze(0)
                    < num_faces.unsqueeze(1)).unsqueeze(-1).float()
            loss = ((recon.float() - packed.float()) ** 2 * mask).sum() / (mask.sum() * packed.shape[-1]).clamp(min=1)

        self.log("train/ae_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True,
                 logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=False,
                 logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)

        loss = loss / self.config.gradient_accumulation_steps
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            step(optimizer, [self.shape2vecset_ae])
            optimizer.zero_grad(set_to_none=True)  # type: ignore
            scheduler.step()  # type: ignore
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False,
                 prog_bar=False, logger=self.has_logger, sync_dist=True)  # type: ignore

    def _joint_forward(self, data, keep_k=None, encoded_clean=None):
        """One pass of the full pipeline under the joint objective.

        keep_k: explicit AE token count (None = the AE's own k sampling).
        encoded_clean: reuse an earlier pass's encoder output (pass it detached
        to keep the encoder out of the graph).
        Returns the total loss, its parts, and stats for logging.
        """
        if encoded_clean is None:
            encoded_clean = self.encoder(data.x, data.edge_index, data.batch)  # num_faces x (192 * 3)
        encoded_x = encoded_clean
        if self.config.distill_before_vq:
            encoded_x, ae_recon_loss = self.may_apply_shape2vecset_ae(encoded_x, batch_indices=data.batch, keep_k=keep_k)

        def to_vq_space(faces):
            x = faces.reshape(faces.shape[0] * 3, self.config.per_vertex_embed_dim)  # 3N x 192
            x = self.distribute_features_fn(x, data.faces, data.num_vertices.sum(), self.device)
            return self.pre_quant(x)  # 3N x 192

        encoded_x = to_vq_space(encoded_x)
        code_cls_loss = torch.zeros((), device=self.device)
        if self.code_cls_weight > 0 and self.config.distill_before_vq and self.shape2vecset_ae is not None:
            with torch.no_grad():
                clean_vq = to_vq_space(encoded_clean)
            code_cls_loss = code_classification_loss(encoded_x, clean_vq, self.vq)
        encoded_x, _, commit_loss = self.vq(encoded_x.unsqueeze(0))  # 3N x 192
        encoded_x = encoded_x.squeeze(0)  # (num_faces * 3) x 192
        commit_loss = commit_loss.mean()
        encoded_x = encoded_x.reshape(-1, 3 * encoded_x.shape[-1])  # num_faces x (192 * 3)
        if not self.config.distill_before_vq:
            encoded_x, ae_recon_loss = self.may_apply_shape2vecset_ae(encoded_x, batch_indices=data.batch, keep_k=keep_k)
        encoded_x = self.post_quant(encoded_x)  # num_faces x 512
        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x, data.batch, self.config.batch_size)  # 1 x num_faces x 512, num_faces
        decoded_x_conv = self.decoder(encoded_x_conv)
        if self.config.ce_output:
            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
            decoded_tri = softargmax(decoded_x) / (self.config.num_tokens - 3) - 0.5
            _, decoded_normals, decoded_areas, decoded_angles = create_feature_stack_from_triangles(decoded_tri.reshape(-1, 3, 3))
            if self.config.use_smoothed_loss:
                otarget = torch.nn.functional.one_hot(data.y.reshape(-1), num_classes=self.config.num_tokens - 2).float()
                otarget = otarget.unsqueeze(1)
                starget = torch.nn.functional.conv1d(otarget, self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1)
                if self.config.use_multimodal_loss:
                    starget_a = starget.reshape(-1, decoded_x.shape[-2] * decoded_x_conv.shape[-1])
                    starget_a = torch.nn.functional.normalize(starget_a, p=1.0, dim=-1, eps=1e-12).squeeze(1)
                    starget_b = torch.nn.functional.normalize(starget, p=1.0, dim=-1, eps=1e-12).squeeze(1)
                    loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-2] * decoded_x_conv.shape[-1]), starget_a).mean()
                    loss = loss * 0.1 + torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), starget_b).mean()
                else:
                    starget = torch.nn.functional.normalize(starget, p=1.0, dim=-1, eps=1e-12).squeeze(1)
                    loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), starget).mean()
            else:
                loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), data.y.reshape(-1), reduction='mean')
            y_coords = data.y / (self.config.num_tokens - 3) - 0.5
            loss_tri = torch.nn.functional.mse_loss(decoded_tri, y_coords, reduction='mean')
            loss_normals = torch.nn.functional.mse_loss(decoded_normals, data.x[:, 9:12], reduction='mean')
            loss_areas = torch.nn.functional.mse_loss(decoded_areas, data.x[:, 12:13], reduction='mean')
            loss_angles = torch.nn.functional.mse_loss(decoded_angles, data.x[:, 13:16], reduction='mean')
        else:
            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
            _, decoded_normals, decoded_areas, decoded_angles = create_feature_stack_from_triangles(decoded_x.reshape(-1, 3, 3))
            loss = torch.nn.functional.mse_loss(decoded_x, data.y, reduction='mean')
            loss_tri = torch.nn.functional.mse_loss(decoded_x, data.y, reduction='mean')
            loss_normals = torch.nn.functional.mse_loss(decoded_normals, data.x[:, 9:12], reduction='mean')
            loss_areas = torch.nn.functional.mse_loss(decoded_areas, data.x[:, 12:13], reduction='mean')
            loss_angles = torch.nn.functional.mse_loss(decoded_angles, data.x[:, 13:16], reduction='mean')

        latent_phase = self.training and self.global_step < self.curriculum_latent_steps
        downstream = 0.0 if latent_phase else 1.0
        total = ((loss + loss_tri * self.config.tri_weight + loss_normals * self.config.norm_weight
                  + loss_areas * self.config.area_weight + loss_angles * self.config.angle_weight) * downstream
                 + commit_loss + ae_recon_loss * self.config.ae_recon_weight
                 + code_cls_loss * self.code_cls_weight)
        return dict(
            loss=total, ce_loss=loss, mse_loss=loss_tri, norm_loss=loss_normals, area_loss=loss_areas,
            angle_loss=loss_angles, embed_loss=commit_loss, ae_recon_loss=ae_recon_loss, code_cls_loss=code_cls_loss,
            acc=self.get_accuracy(decoded_x, data.y), acc_tri=self.get_triangle_accuracy(decoded_x, data.y),
            downstream_on=torch.tensor(downstream), encoded_clean=encoded_clean,
        )

    def _log_joint(self, out, prefix, keys=None):
        bar = {'ce_loss', 'mse_loss', 'acc', 'acc_tri'}
        for key in keys or ('ce_loss', 'mse_loss', 'norm_loss', 'area_loss', 'angle_loss', 'embed_loss', 'ae_recon_loss', 'code_cls_loss', 'acc', 'acc_tri', 'loss', 'downstream_on'):
            self.log(f"{prefix}/{key}", out[key].item(), on_step=True, on_epoch=False, prog_bar=key in bar,
                     logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)

    def _codebook_guard(self):
        """The RVQ codebook stays fixed under freeze_base or freeze_codebook: its EMA
        buffers are snapshotted and restored around the training pass."""
        return frozen_codebook(self.vq, active=self.freeze_base or self.freeze_codebook)

    def training_step(self, data, batch_idx):
        if self.shape2vecset_ae is not None and (self.ae_is_flow or self.ae_only_loss):
            return self._training_step_ae_only(data, batch_idx)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        if self.freeze_base:
            # A frozen base must not move through the back door: the encoder's
            # BatchNorm running stats and the VQ's EMA codebook both update in
            # train mode. The VQ has to stay in train mode for its straight-through
            # gradient, so its buffers are restored after the pass instead.
            self.encoder.eval()
        split = self.split_grad and self.shape2vecset_ae is not None and not self.freeze_base
        accum = self.config.gradient_accumulation_steps
        with self._codebook_guard():
            out = self._joint_forward(data, keep_k=self.config.num_sequence_tokens if split else None)
        self._log_joint(out, "train")
        self.manual_backward(out['loss'] / accum)
        if split:
            # Second pass at the AE's sampled k: only the AE learns from it. The
            # base keeps its full-rate gradient from the first pass, and the
            # codebook is not updated from low-k features.
            with frozen_codebook(self.vq):
                out_k = self._joint_forward(data, keep_k=None, encoded_clean=out['encoded_clean'].detach())
            self._log_joint(out_k, "train_lowk", keys=('ce_loss', 'mse_loss', 'acc', 'loss'))
            accumulate_grads(out_k['loss'] / accum, [p for p in self.shape2vecset_ae.parameters() if p.requires_grad])
        # accumulate gradients of `n` batches
        if (batch_idx + 1) % accum == 0:
            if self.freeze_base:
                modules_to_step = [self.shape2vecset_ae]
            else:
                modules_to_step = [self.encoder, self.decoder, self.pre_quant, self.post_quant]
                if self.shape2vecset_ae:
                    modules_to_step.append(self.shape2vecset_ae)
            step(optimizer, modules_to_step)
            optimizer.zero_grad(set_to_none=True)  # type: ignore
            scheduler.step()  # type: ignore
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=self.has_logger, sync_dist=True)  # type: ignore

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        # print("Beginning validation step...")
        # print(f"Validating with {data.num_graphs} meshes w/ avg {data.x.shape[0] / data.num_graphs} faces...")

        # Encode once
        encoded_x = self.encoder(data.x, data.edge_index, data.batch)

        # Loop over different keep_k values
        for keep_k in self.val_keep_k_values:
            # Branch based on distill_before_vq config
            if self.config.distill_before_vq:
                encoded_x_ = encoded_x
                encoded_x_, _ = self.may_apply_shape2vecset_ae(encoded_x_, batch_indices=data.batch, keep_k=keep_k)
                encoded_x_ = encoded_x_.reshape(encoded_x_.shape[0] * 3, self.config.per_vertex_embed_dim)
                ae_corners = encoded_x_
                encoded_x_ = self.distribute_features_fn(encoded_x_, data.faces, data.num_vertices.sum(), self.device)
                encoded_x_ = self.pre_quant(encoded_x_)
                pre_vq = encoded_x_
                encoded_x_, _, commit_loss = self.vq(encoded_x_.unsqueeze(0))
                encoded_x_ = encoded_x_.squeeze(0)
                commit_loss = commit_loss.mean()
                encoded_x_ = encoded_x_.reshape(-1, 3 * encoded_x_.shape[-1])
            else:
                pre_vq = ae_corners = None
                encoded_x_ = encoded_x.reshape(encoded_x.shape[0] * 3, self.config.per_vertex_embed_dim)
                encoded_x_ = self.distribute_features_fn(encoded_x_, data.faces, data.num_vertices.sum(), self.device)
                encoded_x_ = self.pre_quant(encoded_x_)
                encoded_x_, _, commit_loss = self.vq(encoded_x_.unsqueeze(0))
                encoded_x_ = encoded_x_.squeeze(0)
                commit_loss = commit_loss.mean()
                encoded_x_ = encoded_x_.reshape(-1, 3 * encoded_x_.shape[-1])
                encoded_x_, _ = self.may_apply_shape2vecset_ae(encoded_x_, batch_indices=data.batch, keep_k=keep_k)

            # Post-quantization and decoding
            encoded_x_ = self.post_quant(encoded_x_)
            encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x_, data.batch, data.num_graphs)
            decoded_x_conv = self.decoder(encoded_x_conv)

            # Compute losses
            if self.config.ce_output:
                decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
                decoded_c = softargmax(decoded_x) / (self.config.num_tokens - 3) - 0.5
                loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), data.y.reshape(-1), reduction='mean')
                y_coords = data.y / (self.config.num_tokens - 3) - 0.5
                loss_c = torch.nn.functional.mse_loss(decoded_c, y_coords, reduction='mean')
                chamfer_dist = self.compute_chamfer_distance(decoded_c, y_coords)
                # Benchmark-style score: argmax bins, vertices merged, 10k samples, normalized clouds
                chamfer_mesh = batched_mesh_chamfer(decoded_x.argmax(-1), data.y, data.batch, num_samples=10_000)
                # Vertex-consistent decoding: per-vertex vote over the corners sharing a vertex
                voted = vertex_vote_logits(decoded_x, data.faces, int(data.num_vertices.sum()))
                acc_vote = self.get_accuracy(voted, data.y)
                chamfer_mesh_vote = batched_mesh_chamfer(voted.argmax(-1), data.y, data.batch, num_samples=10_000)
                # What the quantizer costs: decode the AE's continuous features with the VQ bypassed
                chamfer_mesh_novq = torch.tensor(float('nan'), device=self.device)
                if pre_vq is not None:
                    novq_conv, novq_mask = self.create_conv_batch(self.post_quant(pre_vq.reshape(-1, 3 * pre_vq.shape[-1])), data.batch, data.num_graphs)
                    novq_x = self.decoder(novq_conv)
                    novq_x = novq_x.reshape(-1, novq_x.shape[-2], novq_x.shape[-1])[novq_mask, :, :]
                    chamfer_mesh_novq = batched_mesh_chamfer(novq_x.argmax(-1), data.y, data.batch, num_samples=10_000)
                # What the face-vertex map buys at decode: quantize every corner on its own (no averaging)
                chamfer_mesh_noconn = torch.tensor(float('nan'), device=self.device)
                if ae_corners is not None and self.config.distribute_features:
                    q, _, _ = self.vq(self.pre_quant(ae_corners).unsqueeze(0))
                    nc_conv, nc_mask = self.create_conv_batch(self.post_quant(q.squeeze(0).reshape(-1, 3 * q.shape[-1])), data.batch, data.num_graphs)
                    nc_x = self.decoder(nc_conv)
                    nc_x = nc_x.reshape(-1, nc_x.shape[-2], nc_x.shape[-1])[nc_mask, :, :]
                    chamfer_mesh_noconn = batched_mesh_chamfer(nc_x.argmax(-1), data.y, data.batch, num_samples=10_000)
            else:
                decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
                loss = torch.nn.functional.mse_loss(decoded_x, data.y, reduction='mean')
                loss_c = torch.nn.functional.mse_loss(decoded_x, data.y, reduction='mean')
                chamfer_dist = self.compute_chamfer_distance(decoded_x, data.y)
                chamfer_mesh = torch.tensor(float('nan'), device=self.device)
                acc_vote = chamfer_mesh_vote = chamfer_mesh_novq = chamfer_mesh_noconn = chamfer_mesh

            # Compute accuracy
            acc = self.get_accuracy(decoded_x, data.y)
            acc_triangle = self.get_triangle_accuracy(decoded_x, data.y)

            # Log metrics with keep_k suffix (hierarchical naming for wandb grouping)
            # e.g., val/acc/k1, val/acc/k2 will be grouped under val/acc/ in wandb
            suffix = f"/k{keep_k}" if len(self.val_keep_k_values) > 1 else ""
            category_suffix = self.interesting_categories[dataloader_idx][1]

            if not torch.isnan(loss).any():
                self.log(f"val/ce_loss{category_suffix}{suffix}", loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(loss_c).any():
                self.log(f"val/mse_loss{category_suffix}{suffix}", loss_c.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(commit_loss).any():
                self.log(f"val/embed_loss{category_suffix}{suffix}", commit_loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(acc).any():
                self.log(f"val/acc{category_suffix}{suffix}", acc.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(acc_triangle).any():
                self.log(f"val/acc_tri{category_suffix}{suffix}", acc_triangle.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(chamfer_dist).any():
                self.log(f"val/chamfer_dist{category_suffix}{suffix}", chamfer_dist.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(chamfer_mesh).any():
                self.log(f"val/chamfer_mesh{category_suffix}{suffix}", chamfer_mesh.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(acc_vote).any():
                self.log(f"val/acc_vote{category_suffix}{suffix}", acc_vote.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(chamfer_mesh_vote).any():
                self.log(f"val/chamfer_mesh_vote{category_suffix}{suffix}", chamfer_mesh_vote.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(chamfer_mesh_novq).any():
                self.log(f"val/chamfer_mesh_novq{category_suffix}{suffix}", chamfer_mesh_novq.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)
            if not torch.isnan(chamfer_mesh_noconn).any():
                self.log(f"val/chamfer_mesh_noconn{category_suffix}{suffix}", chamfer_mesh_noconn.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=self.has_logger, sync_dist=True, batch_size=self.config.batch_size)

    @rank_zero_only
    def on_validation_epoch_end(self):
        assert not self.training

        # Plot metrics vs keep_k if sweeping over multiple keep_k values
        if len(self.val_keep_k_values) > 1:
            self._plot_metrics_vs_keep_k()

        # Per-sample chamfer distances and face counts for compression threshold
        sample_chamfer_dists: dict[tuple[int, int], dict[int, float]] = {}
        sample_num_faces: dict[tuple[int, int], int] = {}

        category_names = [""] + [x[0].strip("_") for x in self.interesting_categories]
        for didx, dataset in enumerate([self.train_dataset] + self.val_datasets):
            output_dir_image = self.output_dir_image_train if didx == 0 else self.output_dir_image_val
            output_dir_mesh = self.output_dir_mesh_train if didx == 0 else self.output_dir_mesh_val
            num_val_samples = min(self.config.num_val_samples, len(dataset))
            for k in range(num_val_samples):
                data = dataset.get(k * (len(dataset) // num_val_samples) % len(dataset))
                encoded_x = self.encoder(data.x.to(self.device), data.edge_index.to(self.device), torch.zeros([data.x.shape[0]], device=self.device).long())
                batch_indices = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

                # GT coords (as tensor on device) for per-sample chamfer distance
                if self.config.ce_output:
                    gt_coords = data.y.to(self.device).float() / (self.config.num_tokens - 3) - 0.5
                else:
                    gt_coords = data.y.to(self.device).float()

                per_keep_k_chamfer: dict[int, float] = {}

                # HACK: a lot of repeated code here
                if self.config.distill_before_vq:
                    for keep_k in self.val_keep_k_values:
                        encoded_x_ = encoded_x
                        encoded_x_, _ = self.may_apply_shape2vecset_ae(encoded_x_, batch_indices=batch_indices, keep_k=keep_k)
                        encoded_x_ = encoded_x_.reshape(encoded_x_.shape[0] * 3, self.config.per_vertex_embed_dim)
                        encoded_x_ = self.distribute_features_fn(encoded_x_, data.faces.to(self.device), data.num_vertices, self.device)
                        encoded_x_ = self.pre_quant(encoded_x_)
                        encoded_x_, _, _ = self.vq(encoded_x_.unsqueeze(0))
                        encoded_x_ = encoded_x_.squeeze(0)
                        encoded_x_ = encoded_x_.reshape(-1, 3 * encoded_x_.shape[-1])
                        encoded_x_ = self.post_quant(encoded_x_)
                        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x_, torch.zeros([data.x.shape[0]], device=self.device).long(), 1)
                        decoded_x_conv = self.decoder(encoded_x_conv)
                        if self.config.ce_output:
                            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
                            # softargmax for differentiable coords (chamfer), argmax for visualization
                            decoded_c = softargmax(decoded_x) / (self.config.num_tokens - 3) - 0.5
                            coords = decoded_x.argmax(-1).detach().cpu().numpy() / (self.config.num_tokens - 3) - 0.5
                        else:
                            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
                            decoded_c = decoded_x
                            coords = decoded_x.detach().cpu().numpy()
                        chamfer_val = self.compute_chamfer_distance(decoded_c, gt_coords)
                        if not torch.isnan(chamfer_val):
                            per_keep_k_chamfer[keep_k or 0] = chamfer_val.item()
                        gen_vertices, gen_faces = triangle_sequence_to_mesh(coords)
                        plot_vertices_and_faces(gen_vertices, gen_faces, output_dir_image / f"{self.current_epoch:06d}_{category_names[didx]}_{k}_{keep_k or 0}.jpg")
                        # try:
                        #     trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False).export(output_dir_mesh / f"{self.current_epoch:06d}_{category_names[didx]}_{k}_{keep_k or 0}.obj")
                        # except Exception as e:
                        #     pass  # sometimes the mesh is invalid (ngon) and we don't want to crash
                else:
                    encoded_x = encoded_x.reshape(encoded_x.shape[0] * 3, self.config.per_vertex_embed_dim)
                    encoded_x = self.distribute_features_fn(encoded_x, data.faces.to(self.device), data.num_vertices, self.device)
                    encoded_x = self.pre_quant(encoded_x)
                    encoded_x, _, _ = self.vq(encoded_x.unsqueeze(0))
                    encoded_x = encoded_x.squeeze(0)
                    encoded_x = encoded_x.reshape(-1, 3 * encoded_x.shape[-1])
                    for keep_k in self.val_keep_k_values:
                        encoded_x_ = encoded_x
                        encoded_x_, _ = self.may_apply_shape2vecset_ae(encoded_x_, batch_indices=batch_indices, keep_k=keep_k)
                        encoded_x_ = self.post_quant(encoded_x_)
                        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x_, torch.zeros([data.x.shape[0]], device=self.device).long(), 1)
                        decoded_x_conv = self.decoder(encoded_x_conv)
                        if self.config.ce_output:
                            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
                            decoded_c = softargmax(decoded_x) / (self.config.num_tokens - 3) - 0.5
                            coords = decoded_x.argmax(-1).detach().cpu().numpy() / (self.config.num_tokens - 3) - 0.5
                        else:
                            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
                            decoded_c = decoded_x
                            coords = decoded_x.detach().cpu().numpy()
                        chamfer_val = self.compute_chamfer_distance(decoded_c, gt_coords)
                        if not torch.isnan(chamfer_val):
                            per_keep_k_chamfer[keep_k or 0] = chamfer_val.item()
                        gen_vertices, gen_faces = triangle_sequence_to_mesh(coords)
                        plot_vertices_and_faces(gen_vertices, gen_faces, output_dir_image / f"{self.current_epoch:06d}_{category_names[didx]}_{k}_{keep_k or 0}.jpg")
                        # try:
                        #     trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False).export(output_dir_mesh / f"{self.current_epoch:06d}_{category_names[didx]}_{k}_{keep_k or 0}.obj")
                        # except Exception as e:
                        #     pass  # sometimes the mesh is invalid (ngon) and we don't want to crash

                sample_chamfer_dists[(didx, k)] = per_keep_k_chamfer
                sample_num_faces[(didx, k)] = data.x.shape[0]

        # Log image grids to wandb/tensorboard
        category_names = [""] + [x[0].strip("_") for x in self.interesting_categories]
        for didx, dataset in enumerate([self.train_dataset] + self.val_datasets):
            output_dir_image = self.output_dir_image_train if didx == 0 else self.output_dir_image_val
            num_val_samples = min(self.config.num_val_samples, len(dataset), 4)  # Max 4 samples

            for k in range(num_val_samples):
                # Collect image paths for this sample (None for missing images)
                image_paths = []

                # First add ground truth image
                gt_path = output_dir_image / f"GT_{category_names[didx]}_{k}.jpg"
                image_paths.append(gt_path if gt_path.exists() else None)

                # Then add reconstruction images for each keep_k value (in order)
                for keep_k in self.val_keep_k_values:
                    recon_path = output_dir_image / f"{self.current_epoch:06d}_{category_names[didx]}_{k}_{keep_k or 0}.jpg"
                    image_paths.append(recon_path if recon_path.exists() else None)

                # Create per-sample chamfer distance vs keep_k plot if we have data
                chamfer_plot = None
                chamfer_data = sample_chamfer_dists.get((didx, k), {})
                if len(chamfer_data) > 1:
                    sorted_keys = sorted(chamfer_data.keys())
                    # Compression threshold: keep_k where token storage = raw mesh storage
                    # Token storage: keep_k * per_vertex_embed_dim * 3
                    # Raw mesh storage: num_faces * 9 (3 vertices * 3 coords per face)
                    num_faces = sample_num_faces.get((didx, k))
                    compression_k = None
                    if num_faces is not None:
                        compression_k = num_faces * 3 / self.config.per_vertex_embed_dim
                    fig = create_metric_vs_keep_k_figure(
                        sorted_keys,
                        [chamfer_data[kk] for kk in sorted_keys],
                        'chamfer_dist',
                        figsize=(8, 4),
                        compression_threshold_k=compression_k,
                    )
                    chamfer_plot = figure_to_numpy(fig)
                    plt.close(fig)

                # Log the grid (always log, even if some images are missing)
                split_name = "train" if didx == 0 else "val"
                category_str = f"_{category_names[didx]}" if category_names[didx] else ""
                log_key = f"reconstruction_grids/{split_name}{category_str}_sample_{k}"
                self._create_and_log_image_grid(image_paths, log_key, chamfer_plot=chamfer_plot)

    def visualize_groundtruth(self):
        print("Visualizing groundtruth...")
        category_names = [""] + [x[0].strip("_") for x in self.interesting_categories]
        for didx, dataset in enumerate([self.train_dataset] + self.val_datasets):
            output_dir_image = self.output_dir_image_train if didx == 0 else self.output_dir_image_val
            num_val_samples = min(self.config.num_val_samples, len(dataset))
            for k in tqdm(range(num_val_samples), desc=f"Dataset {didx + 1}/{len(self.val_datasets) + 1}"):
                data = dataset.get(k * (len(dataset) // num_val_samples) % len(dataset))
                if self.config.ce_output:
                    coords = data.y / (self.config.num_tokens - 3) - 0.5
                else:
                    coords = data.y
                gen_vertices, gen_faces = triangle_sequence_to_mesh(coords)
                plot_vertices_and_faces(gen_vertices, gen_faces, output_dir_image / f"GT_{category_names[didx]}_{k}.jpg")
        print("Finished visualizing groundtruth.")

    def _load_pretrained_vq_components(self, checkpoint_path: str) -> None:
        """Load pretrained encoder, decoder, pre_quant, post_quant, and vq from a checkpoint.

        This allows initializing the original MeshGPT components from a pretrained
        checkpoint while leaving new components (e.g., shape2vecset_ae) randomly initialized
        for end-to-end training.

        Args:
            checkpoint_path: Path to the pretrained checkpoint file.
        """
        print(f"Loading pretrained VQ components from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

        self.encoder.load_state_dict(get_parameters_from_state_dict(state_dict, "encoder"))
        self.decoder.load_state_dict(get_parameters_from_state_dict(state_dict, "decoder"))
        self.pre_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "pre_quant"))
        self.post_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "post_quant"))
        self.vq.load_state_dict(get_parameters_from_state_dict(state_dict, "vq"))

        print("Pretrained VQ components loaded successfully")

    def _load_pretrained_ae_components(self, checkpoint_path: str) -> None:
        """Restore shape2vecset_ae weights when the checkpoint contains them.

        vq_resume historically loaded only the base MeshGPT components; when
        fine-tuning from one of our own checkpoints that silently reset the AE
        to random init (observed: step-0 chamfer ~0.06 instead of the source
        run's ~0.006). strict=False so architecture deltas (e.g. adding a REPA
        projector) load what matches and report the rest."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        ae_sd = get_parameters_from_state_dict(state_dict, "shape2vecset_ae")
        if not ae_sd:
            print("Checkpoint has no shape2vecset_ae weights (base-only, e.g. 749-0); AE starts fresh")
            return
        missing, unexpected = self.shape2vecset_ae.load_state_dict(ae_sd, strict=False)
        print(f"Loaded shape2vecset_ae from checkpoint: {len(ae_sd)} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    def _plot_metrics_vs_keep_k(self):
        """Plot validation metrics vs keep_k values and save to disk."""
        import re
        from collections import defaultdict

        # Parse callback_metrics to extract metrics by (metric_name, category_suffix, keep_k)
        metrics_data = defaultdict(lambda: defaultdict(list))  # {(metric_name, category_suffix): {keep_k: [values]}}

        for key, value in self.trainer.callback_metrics.items():
            # Match pattern: val/{metric_name}{category_suffix}/k{keep_k}
            # Examples: "val/ce_loss/k1", "val/acc_chair/k2", "val/mse_loss_bench/k4"
            match = re.match(r'val/([^/]+)/k(\d+)$', key)
            if match:
                metric_with_category = match.group(1)  # e.g., "ce_loss", "acc_chair", "mse_loss_bench"
                keep_k = int(match.group(2))

                # Split metric name and category suffix
                # Metric names: ce_loss, mse_loss, embed_loss, acc, acc_tri, chamfer_mesh, chamfer_dist
                # Category suffixes: "", "_chair", "_bench", etc.
                metric_name = None
                category_suffix = ""

                for base_metric in ['ce_loss', 'mse_loss', 'embed_loss', 'acc_tri', 'acc_vote', 'acc', 'chamfer_mesh_vote', 'chamfer_mesh_novq', 'chamfer_mesh_noconn', 'chamfer_mesh', 'chamfer_dist']:
                    if metric_with_category.startswith(base_metric):
                        metric_name = base_metric
                        category_suffix = metric_with_category[len(base_metric):]
                        break

                if metric_name:
                    key_tuple = (metric_name, category_suffix)
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    metrics_data[key_tuple][keep_k].append(value)

        # Create plots for each (metric_name, category_suffix) combination
        for (metric_name, category_suffix), keep_k_dict in metrics_data.items():
            # Sort by keep_k value
            keep_k_values = sorted(keep_k_dict.keys())
            metric_values = [keep_k_dict[k][0] if keep_k_dict[k] else float('nan') for k in keep_k_values]

            # Skip if we don't have multiple keep_k values
            if len(keep_k_values) <= 1:
                continue

            category_display = category_suffix.strip('_') if category_suffix else 'all'
            fig = create_metric_vs_keep_k_figure(
                keep_k_values, metric_values, metric_name, title_suffix=category_display,
            )

            # Save figure
            category_str = category_suffix if category_suffix else ""
            filename = f"epoch_{self.current_epoch:06d}_{metric_name}{category_str}_vs_tokens.png"
            save_path = self.output_dir_metrics_val / filename
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

            # Log to wandb/tensorboard
            log_key = f"metrics_plots/val_{metric_name}{category_str}_vs_tokens"
            self._log_figure_to_logger(log_key, fig)

            plt.close(fig)

    def train_dataloader(self):
        return TriangleNodesWithFacesDataloader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=not self.config.overfit, num_workers=self.config.num_workers, pin_memory=True)

    def val_dataloader(self):
        dataloaders = []
        for i, val_dataset in enumerate(self.val_datasets):
            dataloaders.append(TriangleNodesWithFacesDataloader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False, num_workers=self.config.num_workers))
        return dataloaders

    def get_accuracy(self, x, y):
        if self.config.ce_output:
            return (x.argmax(-1).reshape(-1) == y.reshape(-1)).sum() / (x.shape[0] * x.shape[1])
        return (quantize_coordinates(x, self.config.num_tokens - 2).reshape(-1) == quantize_coordinates(y, self.config.num_tokens - 2).reshape(-1)).sum() / (x.shape[0] * x.shape[1])

    def get_triangle_accuracy(self, x, y):
        if self.config.ce_output:
            return torch.all(x.argmax(-1).reshape(-1, 9) == y.reshape(-1, 9), dim=-1).sum() / x.shape[0]
        return torch.all((quantize_coordinates(x, self.config.num_tokens - 2).reshape(-1, 9) == quantize_coordinates(y, self.config.num_tokens - 2).reshape(-1, 9)), dim=-1).sum() / (x.shape[0])

    def compute_chamfer_distance(
        self,
        pred_coords: Float[Tensor, "num_faces coords_per_face"],  # coords_per_face = 9 (3 vertices * 3 coords)
        gt_coords: Float[Tensor, "num_faces coords_per_face"],
        num_samples: int = 2048,
    ) -> Float[Tensor, ""]:
        """
        Compute Chamfer Distance between predicted and ground truth meshes.

        Samples points uniformly from mesh surfaces and computes bidirectional Chamfer Distance.
        This is the standard approach used in 3D deep learning papers.

        Args:
            pred_coords: Predicted triangle coordinates (num_faces, 9)
            gt_coords: Ground truth triangle coordinates (num_faces, 9)
            num_samples: Number of points to sample from each mesh surface (default: 2048)

        Returns:
            chamfer_dist: Scalar tensor with bidirectional Chamfer Distance
        """
        try:
            # Reshape triangles to vertices: (num_faces, 9) → (num_vertices, 3)
            pred_vertices = pred_coords.reshape(-1, 3)
            gt_vertices = gt_coords.reshape(-1, 3)

            # Create face indices: each consecutive triplet of vertices forms a triangle
            # e.g., [[0,1,2], [3,4,5], [6,7,8], ...]
            num_faces = pred_coords.shape[0]
            faces = torch.arange(num_faces * 3, dtype=torch.long, device=pred_coords.device).reshape(num_faces, 3)

            # Create PyTorch3D Meshes (add batch dimension)
            pred_mesh = Meshes(verts=[pred_vertices], faces=[faces])
            gt_mesh = Meshes(verts=[gt_vertices], faces=[faces])

            # Sample points uniformly from mesh surfaces
            pred_points = sample_points_from_meshes(pred_mesh, num_samples=num_samples)
            gt_points = sample_points_from_meshes(gt_mesh, num_samples=num_samples)

            # Compute bidirectional chamfer distance
            # Returns (loss, loss_norm) where loss is the sum of forward and backward distances
            chamfer_dist, _ = chamfer_distance(pred_points, gt_points)

            return chamfer_dist
        except Exception as e:
            # Return NaN if computation fails (e.g., degenerate/empty meshes)
            return torch.tensor(float('nan'), device=pred_coords.device)

    def may_apply_shape2vecset_ae(
            self,
            encoded_x: Float[Tensor, "num_total_faces embedding_size"],
            batch_indices: Float[Tensor, "num_total_faces"] | None = None,
            keep_k: int | None = None,
    ):
        """Helper function to apply shape2vecset autoencoder with proper dimension handling.

        If shape2vecset_ae is not initialized, returns the input unchanged.
        Applies warmup blending during training if ae_warmup_steps > 0.
        Computes auxiliary reconstruction loss during training if ae_recon_weight > 0.

        Returns:
            tuple of (output, aux_recon_loss) where aux_recon_loss is 0.0 when not applicable
        """
        zero_loss = torch.tensor(0.0, device=encoded_x.device)
        if not self.shape2vecset_ae:
            return encoded_x, zero_loss

        original_input = encoded_x
        kwargs = {'keep_k': keep_k} if keep_k is not None else {}

        if batch_indices is not None:
            encoded_x, num_faces = pack_faces_into_batch(encoded_x, batch_indices, self.config.batch_size)
            kwargs['num_faces'] = num_faces
        else:
            encoded_x = encoded_x.unsqueeze(0)
            kwargs['num_faces'] = None

        encoded_x = self.shape2vecset_ae(encoded_x, **kwargs)['points']

        encoded_x = unpack_faces_from_batch(encoded_x, batch_indices)\
            if batch_indices is not None else encoded_x.squeeze(0)

        ae_output = encoded_x

        # Warmup: blend AE output with original input (only during training)
        if self.training and self.config.ae_warmup_steps > 0 and self.global_step < self.config.ae_warmup_steps:
            alpha = self.global_step / self.config.ae_warmup_steps
            encoded_x = alpha * encoded_x + (1 - alpha) * original_input

        # Auxiliary reconstruction loss (only during training if enabled)
        if self.training and self.config.ae_recon_weight > 0:
            # ae_recon_unblended: score the PURE AE output instead of the
            # warmup blend. The blended loss is artificially ~0 while alpha is
            # small, so the AE gets almost no recon signal during warmup and
            # the handoff at warmup end lands on an untrained AE (the Cycle-3
            # collapse-transient mechanism).
            aux_target = ae_output if getattr(self.config, 'ae_recon_unblended', False) else encoded_x
            # TODO: are there problems w/ permutation invariance? can this be chamfer distance?
            aux_loss = torch.nn.functional.mse_loss(aux_target, original_input)
        else:
            aux_loss = zero_loss

        return encoded_x, aux_loss


def distribute_features(features, face_indices, num_vertices, device):
    # N = num triangles
    # features is N3 x 192
    # face_indices is N x 3
    assert features.shape[0] == face_indices.shape[0] * face_indices.shape[1], "Features and face indices must match in size"
    vertex_features = torch.zeros([num_vertices, features.shape[1]], device=device)
    torch_scatter.scatter_mean(features, face_indices.reshape(-1), out=vertex_features, dim=0)
    distributed_features = vertex_features[face_indices.reshape(-1), :]
    return distributed_features


def dummy_distribute(features, _face_indices, _n, _device):
    return features


def pack_faces_into_batch(
        face_embeddings: Float[Tensor, "num_total_faces embedding_size"],
        batch_indices: Float[Tensor, "num_total_faces"],
        batch_size: int,
        max_num_faces: int | None = None,
) -> tuple[Float[Tensor, "batch_size max_num_faces embedding_size"], Float[Tensor, "batch_size"]]:
    """
    Efficient packing when batch_size << num_total_faces.

    Args:
        face_embeddings: (num_total_faces, embedding_size) tensor
        batch_indices: (num_total_faces,) tensor with values in [0, batch_size)
        batch_size: int - should be small (e.g., < 100)
        max_num_faces: int or None

    Returns:
        batched_embeddings: (batch_size, max_num_faces, embedding_size)
        num_faces: (batch_size,) tensor with the actual number of faces per batch element
    """
    device = face_embeddings.device
    embedding_size = face_embeddings.shape[1]

    # Determine max_num_faces if not provided
    faces_per_batch = torch.bincount(batch_indices, minlength=batch_size)
    if max_num_faces is None:
        max_num_faces = faces_per_batch.max().item()

    # Pre-allocate output
    output = torch.zeros(batch_size, max_num_faces, embedding_size,
                         device=device, dtype=face_embeddings.dtype)

    # Loop over batches (fast when batch_size is small)
    for i in range(batch_size):
        mask = (batch_indices == i)
        faces_in_batch = face_embeddings[mask]
        num_faces = faces_in_batch.shape[0]
        if num_faces > 0:  # Skip empty batches
            output[i, :num_faces] = faces_in_batch

    return output, faces_per_batch


def unpack_faces_from_batch(
        batched_embeddings: Float[Tensor, "batch_size max_num_faces embedding_size"],
        batch_indices: Float[Tensor, "num_total_faces"],
) -> Float[Tensor, "num_total_faces embedding_size"]:
    """
    Inverse of pack_faces_into_batch - unpacks batched embeddings back to original sequence.

    Args:
        batched_embeddings: (batch_size, max_num_faces, embedding_size) tensor
        batch_indices: (num_total_faces,) tensor with values in [0, batch_size)

    Returns:
        face_embeddings: (num_total_faces, embedding_size) tensor in original order
    """
    device = batched_embeddings.device
    num_total_faces = batch_indices.shape[0]
    batch_size = batched_embeddings.shape[0]

    # Compute position of each face within its batch
    # Only loop over batch_size (small), not num_total_faces
    positions = torch.zeros(num_total_faces, dtype=torch.long, device=device)
    for batch_idx in range(batch_size):
        mask = (batch_indices == batch_idx)
        num_in_batch = mask.sum()
        if num_in_batch > 0:
            positions[mask] = torch.arange(num_in_batch, device=device)

    # Vectorized extraction using advanced indexing
    output = batched_embeddings[batch_indices, positions]

    return output


def create_metric_vs_keep_k_figure(
    keep_k_values: list[int],
    metric_values: list[float],
    metric_name: str,
    title_suffix: str = "",
    figsize: tuple[float, float] = (8, 6),
    compression_threshold_k: float | None = None,
) -> plt.Figure:
    """Create a matplotlib figure plotting a metric vs keep_k values.

    Args:
        keep_k_values: Sorted list of keep_k values (x-axis).
        metric_values: Corresponding metric values (y-axis).
        metric_name: Name of the metric (used for axis label and title).
        title_suffix: Optional suffix for the title (e.g., category name).
        figsize: Figure size.
        compression_threshold_k: If provided, draw a vertical line at this keep_k value
            indicating where the token representation becomes compressive (i.e., uses
            fewer numbers than the raw mesh coordinates).

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(keep_k_values, metric_values, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('keep_k', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)

    title = f'{metric_name.replace("_", " ").title()} vs keep_k'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if compression_threshold_k is not None:
        ax.axvline(x=compression_threshold_k, color='red', linestyle='--', linewidth=1.5,
                   label=f'compression threshold (k={compression_threshold_k:.1f})')
        ax.legend(fontsize=10)

    if len(keep_k_values) > 1 and max(keep_k_values) / min(keep_k_values) >= 4:
        ax.set_xscale('log', base=2)
        ax.set_xticks(keep_k_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    return fig


def figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """Render a matplotlib figure to a numpy array (H, W, 3)."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.asarray(buf)[:, :, :3].copy()


@hydra.main(config_path='../config', config_name='shapenet-v2', version_base='1.2')
def main(config):
    trainer = create_trainer("TriangleTokens", config)
    model = TriangleTokenizationGraphConv(config)
    if getattr(config, 'eval_only', False):
        # One pass of the validation loop on the weights loaded via vq_resume
        # (base + codebook + AE), logged like any other validation; no optimizer.
        trainer.validate(model)
        return
    trainer.fit(model, ckpt_path=config.resume)


if __name__ == '__main__':
    main()
