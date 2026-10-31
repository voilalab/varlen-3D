# Copyright (c) 2025, Biao Zhang.
#
# RD-curve evaluation for a trained HyperAutoEncoder using Chamfer distance.
#
# For each of the first --n_shapes validation shapes:
#   1. Encode with base autoencoder  →  shape_latent
#   2. Encode with HAE               →  hyper_latents
#   3. For each prefix length L (powers of 2, same as training checkpoints):
#        a. Decode hyper_latents[:, :L]  →  reconstructed_latent
#        b. base_model.learn + decode on marching-cubes grid  →  SDF volume
#        c. Marching cubes  →  mesh
#        d. Sample --n_surface_samples points from mesh surface
#        e. Symmetric Chamfer distance vs ground-truth surface points
#        f. Save mesh as  output_dir/shape_XXXX/L{L:04d}.obj
#   4. Baseline: same pipeline using shape_latent directly (no HAE)
#        Save as  output_dir/shape_XXXX/baseline.obj
#
# Outputs:
#   output_dir/rd_curve.png   — RD curve (prefix length vs mean Chamfer)
#   output_dir/results.csv    — per-shape, per-prefix results

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import torch
import trimesh

from models import autoencoder
from models.hyper_autoencoder import HyperAutoEncoder
from utils.objaverse import Objaverse


# ── Chamfer distance ──────────────────────────────────────────────────────────

def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Symmetric Chamfer distance:
        mean_i  min_j  ||a_i - b_j||²
      + mean_j  min_i  ||b_j - a_i||²

    a : (N, 3),  b : (M, 3) — both on the same device.
    Uses torch.cdist (exact, O(N·M)).
    """
    sq = torch.cdist(a, b).pow(2)          # (N, M) pairwise squared Euclidean
    a2b = sq.min(dim=1).values.mean()
    b2a = sq.min(dim=0).values.mean()
    return (a2b + b2a).item()


# ── Marching cubes helpers ────────────────────────────────────────────────────

def build_grid(resolution: int, device: torch.device) -> torch.Tensor:
    """Dense 3-D grid in [-1, 1]^3; shape 1 × (R+1)^3 × 3."""
    lin = np.linspace(-1, 1, resolution + 1).astype(np.float32)
    xv, yv, zv = np.meshgrid(lin, lin, lin)
    pts = np.stack([xv, yv, zv]).reshape(3, -1).T          # (R+1)^3 × 3
    return torch.from_numpy(pts)[None].to(device)           # 1 × (R+1)^3 × 3


@torch.no_grad()
def reconstruct_mesh(latent: torch.Tensor,
                     base_model: torch.nn.Module,
                     grid: torch.Tensor,
                     resolution: int) -> 'trimesh.Trimesh | None':
    """
    Reconstruct a mesh from a shape latent using the base model.

    Follows the infer.py pipeline exactly:
      learn(latent)  →  decode on grid (chunked, 100k pts at a time)
      →  permute(1,0,2)  →  marching_cubes  →  verts * gap - 1
    """
    gap = 2.0 / resolution
    x   = base_model.learn(latent)         # 1 × latent_n × model_dim

    # Decode in chunks of 100k to avoid OOM on large grids
    sdf_parts = []
    for start in range(0, grid.shape[1], 100_000):
        out = base_model.decode(x, grid[:, start:start + 100_000]).squeeze(-1)
        sdf_parts.append(out)
    sdf = torch.cat(sdf_parts, dim=1)      # 1 × (R+1)^3

    volume = (sdf.view(resolution + 1, resolution + 1, resolution + 1)
                 .permute(1, 0, 2)
                 .cpu().numpy())

    try:
        verts, faces = mcubes.marching_cubes(volume, 0)
        verts = verts * gap - 1.0
        mesh  = trimesh.Trimesh(verts, faces)
        if len(mesh.vertices) == 0:
            return None
        return mesh
    except Exception as e:
        print(f"    marching_cubes failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    np.random.seed(0)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Base model ────────────────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}  from  {args.base_pth}")
    base_model = autoencoder.__dict__[args.base_model](pc_size=args.point_cloud_size)
    ckpt = torch.load(args.base_pth, map_location='cpu', weights_only=False)
    base_model.load_state_dict(ckpt['model'], strict=True)
    base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    latent_dim = base_model.bottleneck.pre_bottleneck_proj.out_features  # 32
    latent_n   = base_model.num_latents                                   # 1024

    # ── HAE ───────────────────────────────────────────────────────────────────
    print(f"Loading HAE from  {args.hae_pth}")
    hae = HyperAutoEncoder(
        latent_n      = latent_n,
        latent_dim    = latent_dim,
        working_dim   = args.working_dim,
        encoder_depth = args.encoder_depth,
        decoder_depth = args.decoder_depth,
        dim_head      = args.dim_head,
        min_prefix    = args.min_prefix,
    )
    hae_ckpt = torch.load(args.hae_pth, map_location='cpu', weights_only=False)
    hae.load_state_dict(hae_ckpt['model'], strict=True)
    hae.to(device).eval()
    for p in hae.parameters():
        p.requires_grad_(False)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = Objaverse(
        split='val',
        sdf_sampling=False,          # only need surface points
        surface_sampling=True,
        surface_size=args.point_cloud_size,
        dataset_folder=args.data_path,
    )
    n_shapes = min(args.n_shapes, len(dataset))
    print(f"Evaluating {n_shapes} shapes from validation set")

    # ── Prefix lengths: powers of 2 from 1 to latent_n ───────────────────────
    prefix_lengths = [2**i for i in range(int(math.log2(latent_n)) + 1)]
    if prefix_lengths[-1] != latent_n:
        prefix_lengths.append(latent_n)
    print(f"Prefix lengths: {prefix_lengths}")

    # ── Marching cubes grid (built once, reused for all shapes) ───────────────
    grid = build_grid(args.resolution, device)

    # ── Storage ───────────────────────────────────────────────────────────────
    chamfer_hae      = {L: [] for L in prefix_lengths}
    chamfer_baseline = []
    csv_rows         = []

    # ── Per-shape loop ────────────────────────────────────────────────────────
    for idx in range(n_shapes):
        _, _, surface, path, _ = dataset[idx]

        name      = os.path.splitext(os.path.basename(
                        path if isinstance(path, str) else str(path)))[0]
        shape_dir = os.path.join(args.output_dir, f'shape_{idx:04d}_{name}')
        os.makedirs(shape_dir, exist_ok=True)

        surface_gpu = surface.unsqueeze(0).to(device)  # 1 × pc_size × 3
        gt_pts      = surface.to(device)               # pc_size × 3  (ground truth)

        print(f"\n[{idx+1}/{n_shapes}]  {name}")

        with torch.no_grad():
            shape_latent  = base_model.encode(surface_gpu)['x']  # 1 × N × D
            hyper_latents = hae.encode(shape_latent)             # 1 × N × D

        # ── Baseline (base AE, no HAE) ─────────────────────────────────────
        mesh_base = reconstruct_mesh(shape_latent, base_model, grid, args.resolution)
        if mesh_base is not None:
            mesh_base.export(os.path.join(shape_dir, 'baseline.obj'))
            pts_base = torch.from_numpy(
                trimesh.sample.sample_surface(mesh_base, args.n_surface_samples)[0]
                .astype(np.float32)
            ).to(device)
            cd_base = chamfer_distance(pts_base, gt_pts)
        else:
            print("  WARNING: baseline mesh empty")
            cd_base = float('nan')

        chamfer_baseline.append(cd_base)
        csv_rows.append((idx, 'baseline', cd_base))
        print(f"  baseline   chamfer = {cd_base:.6f}")

        # ── HAE at each prefix length ──────────────────────────────────────
        for L in prefix_lengths:
            with torch.no_grad():
                recon = hae.decode(hyper_latents[:, :L])   # 1 × N × D

            mesh = reconstruct_mesh(recon, base_model, grid, args.resolution)
            if mesh is not None:
                mesh.export(os.path.join(shape_dir, f'L{L:04d}.obj'))
                pts = torch.from_numpy(
                    trimesh.sample.sample_surface(mesh, args.n_surface_samples)[0]
                    .astype(np.float32)
                ).to(device)
                cd = chamfer_distance(pts, gt_pts)
            else:
                print(f"  WARNING: L={L} mesh empty")
                cd = float('nan')

            chamfer_hae[L].append(cd)
            csv_rows.append((idx, L, cd))
            print(f"  L={L:4d}   chamfer = {cd:.6f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['shape_idx', 'prefix_or_baseline', 'chamfer'])
        writer.writerows(csv_rows)
    print(f"\nResults  →  {csv_path}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    mean_hae      = {L: float(np.nanmean(chamfer_hae[L])) for L in prefix_lengths}
    mean_baseline = float(np.nanmean(chamfer_baseline))

    print(f"\n{'='*50}")
    print(f"  RD Curve  ({n_shapes} shapes, resolution={args.resolution})")
    print(f"  {'tokens':>8}   {'mean chamfer':>14}")
    print(f"  {'-'*26}")
    for L in prefix_lengths:
        print(f"  {L:>8}   {mean_hae[L]:>14.6f}")
    print(f"  {'baseline':>8}   {mean_baseline:>14.6f}")
    print(f"{'='*50}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(prefix_lengths,
            [mean_hae[L] for L in prefix_lengths],
            'o-', color='steelblue', linewidth=2, markersize=5, label='HAE')

    ax.axhline(mean_baseline, color='darkorange', linewidth=2,
               linestyle='--', label=f'Base AE (no HAE)  [{mean_baseline:.4f}]')

    ax.set_xscale('log', base=2)
    ax.set_xticks(prefix_lengths)
    ax.set_xticklabels(prefix_lengths, rotation=45, fontsize=8)
    ax.set_xlabel('Prefix length (tokens)')
    ax.set_ylabel('Symmetric Chamfer distance')
    ax.set_title(f'HAE RD Curve — Chamfer  ({n_shapes} shapes, res={args.resolution})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'rd_curve.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"RD curve  →  {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HAE RD-curve evaluation (Chamfer distance)')

    # ── Models ────────────────────────────────────────────────────────────────
    parser.add_argument('--base_model',  default='learnable_vec1024x32_dim1024_depth24_nb', type=str)
    parser.add_argument('--base_pth',   required=True, type=str)
    parser.add_argument('--hae_pth',    required=True, type=str)

    # ── HAE architecture (must match the checkpoint) ──────────────────────────
    parser.add_argument('--working_dim',   default=512, type=int)
    parser.add_argument('--encoder_depth', default=8,   type=int)
    parser.add_argument('--decoder_depth', default=8,   type=int)
    parser.add_argument('--dim_head',      default=64,  type=int)
    parser.add_argument('--min_prefix',    default=1,   type=int)

    # ── Evaluation settings ───────────────────────────────────────────────────
    parser.add_argument('--data_path',          default='./data/', type=str)
    parser.add_argument('--point_cloud_size',   default=8192, type=int)
    parser.add_argument('--n_shapes',           default=10,   type=int)
    parser.add_argument('--resolution',         default=128,  type=int,
                        help='Marching cubes grid resolution')
    parser.add_argument('--n_surface_samples',  default=10000, type=int,
                        help='Points sampled from reconstructed mesh for Chamfer')
    parser.add_argument('--output_dir',         default='./eval_hae_rd/', type=str)

    main(parser.parse_args())
