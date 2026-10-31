# Copyright (c) 2025, Biao Zhang.

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard.writer import SummaryWriter

torch.set_num_threads(8)

import utils.misc as misc
from utils.objaverse import Objaverse
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from models import autoencoder
from models.hyper_autoencoder import HyperAutoEncoder
from engines.engine_hae import train_one_epoch_hae, validate_hae


def get_args_parser():
    parser = argparse.ArgumentParser('HyperAutoEncoder Training', add_help=False)

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument('--batch_size',  default=6,   type=int)
    parser.add_argument('--epochs',      default=200, type=int)
    parser.add_argument('--accum_iter',  default=1,   type=int)

    # ── Frozen base model ─────────────────────────────────────────────────────
    parser.add_argument('--base_model', default='learnable_vec1024x32_dim1024_depth24_nb', type=str)
    parser.add_argument('--base_pth',   required=True, type=str,
                        help='Path to pretrained base model checkpoint (.pth)')
    parser.add_argument('--point_cloud_size', default=8192, type=int)

    # ── HAE architecture ──────────────────────────────────────────────────────
    parser.add_argument('--working_dim',       default=512,   type=int,
                        help='Internal attention dimension of the HAE')
    parser.add_argument('--encoder_depth',     default=8,     type=int,
                        help='Number of self-attention layers in the hyper-encoder')
    parser.add_argument('--decoder_depth',     default=8,     type=int,
                        help='Number of self-attention layers in the hyper-decoder')
    parser.add_argument('--dim_head',          default=64,    type=int)
    parser.add_argument('--min_prefix',        default=64,    type=int,
                        help='Minimum prefix length L sampled during training')
    parser.add_argument('--bias_small_prefix', action='store_true',
                        help='Sample prefix length from an exponential distribution '
                             'biased toward smaller values (default: uniform)')

    # ── Validation ────────────────────────────────────────────────────────────
    parser.add_argument('--val_every',      default=5,  type=int,
                        help='Plot RD curve every N epochs')
    parser.add_argument('--n_val_shapes',   default=100, type=int,
                        help='Number of shapes to use for RD curve validation')
    parser.add_argument('--n_rd_intervals', default=10,  type=int,
                        help='Number of prefix lengths to evaluate on the RD curve')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    parser.add_argument('--latent_loss_weight', type=float, default=1.0,
                        help='Weight on direct latent MSE loss')
    parser.add_argument('--sdf_loss_weight',    type=float, default=1.0,
                        help='Weight on SDF composite loss; set to 0 to skip SDF branch entirely')
    parser.add_argument('--clip_grad',     type=float, default=1.0)
    parser.add_argument('--weight_decay',  type=float, default=0.05)
    parser.add_argument('--lr',            type=float, default=None,
                        help='Absolute LR (overrides --blr if set)')
    parser.add_argument('--blr',           type=float, default=5e-5,
                        help='Base LR; actual_lr = blr * total_batch_size / 256')
    parser.add_argument('--min_lr',        type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int,   default=1)

    # ── W&B ───────────────────────────────────────────────────────────────────
    parser.add_argument('--wandb',         action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='vecsetx-hae', type=str)
    parser.add_argument('--wandb_run_id',  default=None, type=str,
                        help='Resume an existing W&B run (used by sweep agent)')

    # ── Dataset / IO ──────────────────────────────────────────────────────────
    parser.add_argument('--data_path',   default='./data/', type=str)
    parser.add_argument('--output_dir',  default='./output/hae/')
    parser.add_argument('--log_dir',     default='./output/hae/')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=0,    type=int)
    parser.add_argument('--resume',          default=None, help='Resume HAE from checkpoint')
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='Discard saved optimizer/scaler state when resuming; '
                             'useful for fine-tuning with a different LR or loss.')
    parser.add_argument('--start_epoch', default=0,    type=int)
    parser.add_argument('--num_workers', default=32,   type=int)
    parser.add_argument('--pin_mem',     action='store_true')
    parser.add_argument('--no_pin_mem',  action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # ── Distributed ───────────────────────────────────────────────────────────
    parser.add_argument('--world_size',   default=1,       type=int)
    parser.add_argument('--local_rank',   default=-1,      type=int)
    parser.add_argument('--dist_on_itp',  action='store_true')
    parser.add_argument('--dist_url',     default='env://')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_train = Objaverse(
        split='train', sdf_sampling=True, sdf_size=1024,
        surface_sampling=True, surface_size=args.point_cloud_size,
    )

    num_tasks   = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )

    # Val loader — split='val' disables augmentation but returns vol+near points,
    # same layout as train, so loss_vol can be computed during validation.
    # Only used on rank 0; no distributed sampler needed.
    if global_rank == 0:
        dataset_val = Objaverse(
            split='val', sdf_sampling=True, sdf_size=1024,
            surface_sampling=True, surface_size=args.point_cloud_size,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )
    else:
        data_loader_val = None

    # ── Frozen base model ─────────────────────────────────────────────────────
    base_model = autoencoder.__dict__[args.base_model](pc_size=args.point_cloud_size)
    ckpt = torch.load(args.base_pth, map_location='cpu', weights_only=False)
    base_model.load_state_dict(ckpt['model'], strict=True)
    base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    print(f"Loaded and froze base model from {args.base_pth}")
    print(f"  latent shape: {base_model.num_latents} × {base_model.bottleneck.pre_bottleneck_proj.out_features}")

    # ── HAE ───────────────────────────────────────────────────────────────────
    latent_dim = base_model.bottleneck.pre_bottleneck_proj.out_features  # 32
    latent_n   = base_model.num_latents                                   # 1024

    hae = HyperAutoEncoder(
        latent_n=latent_n,
        latent_dim=latent_dim,
        working_dim=args.working_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        dim_head=args.dim_head,
        min_prefix=args.min_prefix,
        bias_small_prefix=args.bias_small_prefix,
    )
    hae.to(device)

    n_params = sum(p.numel() for p in hae.parameters() if p.requires_grad)
    print(f"HAE parameters: {n_params / 1e6:.2f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {args.blr:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"effective batch size: {eff_batch_size}")

    if args.distributed:
        hae = torch.nn.parallel.DistributedDataParallel(
            hae, device_ids=[args.gpu], find_unused_parameters=False,
        )
        hae_without_ddp = hae.module
    else:
        hae_without_ddp = hae

    optimizer   = torch.optim.AdamW(hae_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()
    criterion   = torch.nn.L1Loss()

    if args.resume:
        if args.reset_optimizer:
            # Load weights only; keep the freshly-created optimizer and scaler.
            # args.start_epoch must be set explicitly by the caller.
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            hae_without_ddp.load_state_dict(ckpt['model'])
            print(f"Loaded weights from {args.resume} (optimizer/scaler reset)")
        else:
            misc.load_model(args=args, model_without_ddp=hae_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # ── W&B init (rank 0 only) ────────────────────────────────────────────────
    if args.wandb and misc.is_main_process():
        import wandb
        wandb.init(
            project  = args.wandb_project,
            id       = args.wandb_run_id,     # resumes existing run when called from sweep agent
            resume   = 'allow',
            config   = vars(args),
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"Start training HAE for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_hae(
            base_model, hae, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer, args=args,
        )

        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=hae, model_without_ddp=hae_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
            )

        # ── RD curve validation (rank 0 only) ─────────────────────────────
        val_stats = None
        if misc.is_main_process() and (epoch % args.val_every == 0 or epoch + 1 == args.epochs):
            val_stats = validate_hae(
                base_model, hae, data_loader_val, criterion, device,
                epoch=epoch,
                output_dir=args.output_dir,
                log_writer=log_writer,
                n_shapes=args.n_val_shapes,
                n_intervals=args.n_rd_intervals,
            )

        # All ranks must wait here: without this barrier, ranks 1-3 would start
        # the next epoch's DDP backward (triggering all_reduce) while rank 0 is
        # still running validate_hae — causing a deadlock.
        if args.distributed:
            torch.distributed.barrier()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

        # ── W&B logging (rank 0 only) ──────────────────────────────────────
        if args.wandb and misc.is_main_process():
            wandb.log({f'train_{k}': v for k, v in train_stats.items()}, step=epoch)
            if val_stats is not None:
                wandb.log(
                    {f'val_near_iou_L{L}':    m['near_iou']    for L, m in val_stats.items()} |
                    {f'val_loss_latent_L{L}': m['loss_latent'] for L, m in val_stats.items()} |
                    {f'val_loss_vol_L{L}':    m['loss_vol']    for L, m in val_stats.items()} |
                    {f'val_loss_near_L{L}':   m['loss_near']   for L, m in val_stats.items()},
                    step=epoch,
                )

    if args.wandb and misc.is_main_process():
        wandb.finish()

    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
