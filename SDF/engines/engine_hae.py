# Copyright (c) 2025, Biao Zhang.

import math
import os
import sys
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn.functional as F

import utils.misc as misc
import utils.lr_sched as lr_sched


def calc_iou(output, labels, threshold=0):
    target = (labels >= threshold).float()
    pred   = (output >= threshold).float()
    intersection = (pred * target).sum(dim=1)
    union = (pred + target).gt(0).sum(dim=1) + 1e-5
    return (intersection / union).mean()


def points_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def train_one_epoch_hae(
    base_model: torch.nn.Module,
    hae: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None,
):
    hae.train(True)
    base_model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',         misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('prefix_len', misc.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    header = f'Epoch: [{epoch}]'
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (points, labels, surface, _, _) in enumerate(
        metric_logger.log_every(data_loader, 20, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points  = points.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):

            points    = points.requires_grad_(True)
            points_all = torch.cat([points, surface], dim=1)

            # ── Step 1: get shape latent from frozen encoder ───────────────
            # torch.no_grad here because gradients do not need to flow
            # through the base encoder — only through HAE and the base decoder.
            with torch.no_grad():
                shape_latent = base_model.encode(surface)['x']  # B × 1024 × 32

            # ── Step 2: HAE encode → prefix truncate → decode ─────────────
            reconstructed_latent, _, prefix_length = hae(shape_latent)  # B × 1024 × 32

            # ── Step 3: direct latent reconstruction loss ─────────────────
            # shape_latent was computed under torch.no_grad() so it is already
            # detached.  MSE here bypasses the 24 frozen transformer layers
            # entirely and gives the HAE a direct gradient on every latent
            # token equally — no 10× near-surface bias.
            # Valid because the learnable base model produces canonically
            # ordered latent tokens (fixed query indices 0..1023), so
            # token-wise MSE is meaningful across shapes.
            loss_latent = F.mse_loss(reconstructed_latent, shape_latent)
            loss        = args.latent_loss_weight * loss_latent

            # ── Step 4: SDF composite loss (skipped when sdf_loss_weight==0) ─
            # The SDF branch builds a large second-order graph via
            # points_gradient(retain_graph=True).  If it is not included in
            # the final loss the graph is never freed → OOM.  Skip the whole
            # branch when the weight is zero so the graph is never built.
            loss_vol = loss_near = loss_eikonal = loss_surface = torch.zeros(1, device=device)
            if args.sdf_loss_weight > 0:
                x      = base_model.learn(reconstructed_latent)       # B × 1024 × 1024
                output = base_model.decode(x, points_all).squeeze(-1) # B × N

                grad = points_gradient(points_all, output)

                loss_eikonal = (grad.norm(2, dim=-1) - 1).pow(2).mean()
                loss_vol     = criterion(output[:, :1024],     labels[:, :1024])
                loss_near    = criterion(output[:, 1024:2048], labels[:, 1024:2048])
                loss_surface = output[:, 2048:].abs().mean()

                loss_sdf = loss_vol + 10 * loss_near + 0.001 * loss_eikonal + loss_surface
                loss     = loss + args.sdf_loss_weight * loss_sdf

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss, optimizer, clip_grad=max_norm,
            parameters=hae.parameters(), create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_latent=loss_latent.item())
        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())
        metric_logger.update(loss_eikonal=loss_eikonal.item())
        metric_logger.update(loss_surface=loss_surface.item())
        metric_logger.update(prefix_len=float(prefix_length))

        max_lr = max(g["lr"] for g in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        # all_reduce_mean is a collective — must be called on all ranks
        if (data_iter_step + 1) % accum_iter == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('loss',       loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr',         max_lr,            epoch_1000x)
                log_writer.add_scalar('prefix_len', prefix_length,     epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_hae(
    base_model: torch.nn.Module,
    hae: torch.nn.Module,
    val_loader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    output_dir: str = None,
    log_writer=None,
    n_shapes: int = 100,
    n_intervals: int = 10,
):
    """
    For n_shapes validation shapes, reconstruct at log-spaced prefix lengths
    (powers of 2) matching the training Matryoshka checkpoints, then plot the
    RD curve.

    Uses split='val' data (no augmentation).
    Encodes each shape once, then decodes at each prefix length.
    """
    hae_module = hae.module if hasattr(hae, 'module') else hae
    hae_module.eval()
    base_model.eval()

    # Log-spaced prefix lengths matching the Matryoshka training checkpoints.
    # Using linspace here would miss all training checkpoints between 1 and
    # latent_n/n_intervals, making a progressive curve look flat.
    latent_n = hae_module.latent_n
    prefix_lengths = [2**i for i in range(int(math.log2(latent_n)) + 1)]
    if prefix_lengths[-1] != latent_n:
        prefix_lengths.append(latent_n)
    # n_intervals is ignored when using log spacing — all powers of 2 are used.

    # Accumulate per-shape metrics for each prefix length
    metrics = {L: {'loss_latent': [], 'loss_vol': [], 'loss_near': [],
                   'near_iou': [], 'loss_surface': []}
               for L in prefix_lengths}
    n_collected = 0

    for points, labels, surface, _, _ in val_loader:
        if n_collected >= n_shapes:
            break

        # Truncate batch to however many shapes we still need
        remaining = n_shapes - n_collected
        if points.shape[0] > remaining:
            points  = points[:remaining]
            labels  = labels[:remaining]
            surface = surface[:remaining]

        points  = points.to(device)
        labels  = labels.to(device)
        surface = surface.to(device)

        # Encode shape → hyper-latent once per shape
        shape_latent  = base_model.encode(surface)['x']   # B × latent_n × latent_dim
        hyper_latents = hae_module.encode(shape_latent)    # B × latent_n × latent_dim

        # points = cat([vol_points, near_points]), same layout as training:
        #   [:n_vol]          → vol points
        #   [n_vol:n_vol+n_near] → near points
        #   points_all[n_vol+n_near:] → surface points (appended below)
        n_vol  = labels.shape[1] // 2   # sdf_size // 2 * 2 = sdf_size; first half vol
        n_near = labels.shape[1] - n_vol
        points_all = torch.cat([points, surface], dim=1)  # B × (n_vol+n_near+pc_size) × 3

        for L in prefix_lengths:
            reconstructed = hae_module.decode(hyper_latents[:, :L])   # B × latent_n × latent_dim
            x      = base_model.learn(reconstructed)                   # B × latent_n × dim
            output = base_model.decode(x, points_all).squeeze(-1)     # B × (n_vol+n_near+pc_size)

            loss_latent  = F.mse_loss(reconstructed, shape_latent).item()
            loss_vol     = criterion(output[:, :n_vol],         labels[:, :n_vol]).item()
            loss_near    = criterion(output[:, n_vol:n_vol+n_near], labels[:, n_vol:]).item()
            loss_surface = output[:, n_vol + n_near:].abs().mean().item()
            near_iou     = calc_iou(output[:, n_vol:n_vol+n_near], labels[:, n_vol:]).item()

            metrics[L]['loss_latent'].append(loss_latent)
            metrics[L]['loss_vol'].append(loss_vol)
            metrics[L]['loss_near'].append(loss_near)
            metrics[L]['loss_surface'].append(loss_surface)
            metrics[L]['near_iou'].append(near_iou)

        n_collected += points.shape[0]

    # Average across shapes
    avg = {L: {k: float(np.mean(v)) for k, v in metrics[L].items()} for L in prefix_lengths}

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  RD Curve — Epoch {epoch}  ({n_collected} shapes, {n_intervals} intervals)")
    print(f"  {'tokens':>8}  {'loss_latent':>12}  {'loss_vol':>9}  {'loss_near':>10}  {'near_iou':>10}  {'loss_surf':>10}")
    print(f"  {'-'*68}")
    for L in prefix_lengths:
        print(f"  {L:>8}  {avg[L]['loss_latent']:>12.4f}  {avg[L]['loss_vol']:>9.4f}"
              f"  {avg[L]['loss_near']:>10.4f}  {avg[L]['near_iou']:>10.4f}"
              f"  {avg[L]['loss_surface']:>10.4f}")
    print(f"{'='*80}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'RD Curve — Epoch {epoch}', fontsize=12)

    Ls = prefix_lengths

    axes[0].plot(Ls, [avg[L]['loss_latent'] for L in Ls], 'o-', color='mediumseagreen', linewidth=2)
    axes[0].set_xlabel('Prefix length (tokens)')
    axes[0].set_ylabel('MSE (latent reconstruction)')
    axes[0].set_title('Latent distortion (lower is better)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(Ls, [avg[L]['loss_vol'] for L in Ls], 'o-', color='mediumpurple', linewidth=2)
    axes[1].set_xlabel('Prefix length (tokens)')
    axes[1].set_ylabel('L1 loss (volumetric SDF)')
    axes[1].set_title('Vol SDF distortion (lower is better)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(Ls, [avg[L]['loss_near'] for L in Ls], 'o-', color='steelblue', linewidth=2)
    axes[2].set_xlabel('Prefix length (tokens)')
    axes[2].set_ylabel('L1 loss (near-surface SDF)')
    axes[2].set_title('Near SDF distortion (lower is better)')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(Ls, [avg[L]['near_iou'] for L in Ls], 'o-', color='darkorange', linewidth=2)
    axes[3].set_xlabel('Prefix length (tokens)')
    axes[3].set_ylabel('IoU (near-surface)')
    axes[3].set_title('Quality (higher is better)')
    axes[3].set_ylim(0, 1)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save PNG
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'rd_curve_epoch{epoch:04d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  RD curve saved to {save_path}")

    # Log to TensorBoard — figure as image + individual scalars per prefix
    if log_writer is not None:
        log_writer.add_figure('val/rd_curve', fig, epoch)
        for L in prefix_lengths:
            log_writer.add_scalar(f'val/loss_latent_L{L:04d}', avg[L]['loss_latent'], epoch)
            log_writer.add_scalar(f'val/loss_vol_L{L:04d}',    avg[L]['loss_vol'],    epoch)
            log_writer.add_scalar(f'val/near_iou_L{L:04d}',    avg[L]['near_iou'],    epoch)
            log_writer.add_scalar(f'val/loss_near_L{L:04d}',   avg[L]['loss_near'],   epoch)

    plt.close(fig)
    hae_module.train()
    return avg
