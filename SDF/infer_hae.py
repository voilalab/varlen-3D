# Copyright (c) 2025, Biao Zhang.
#
# Single-shape HAE inference and RD-curve evaluation.
#
# Reads an OBJ file, normalizes it with sphere normalization (same as infer.py),
# samples a point cloud, then reconstructs the shape at every prefix length
# used during validation (powers of 2 from 1 to latent_n).
#
# For each prefix length L:
#   1. Encode point cloud → shape latent (frozen base model)
#   2. Encode shape latent → hyper latents (HAE)
#   3. Decode hyper_latents[:, :L] → reconstructed latent
#   4. Marching cubes → mesh   (saved to output_dir/L{L:04d}.obj)
#   5. Symmetric Chamfer distance vs original mesh surface samples
#
# Baseline (base AE, no HAE) is also reconstructed and saved.
#
# Outputs:
#   output_dir/baseline.obj
#   output_dir/L{L:04d}.obj  for each prefix length
#   output_dir/rd_curve.png
#   output_dir/results.csv

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


# ── Chamfer distance ──────────────────────────────────────────────────────────

def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Symmetric Chamfer distance:
        mean_i  min_j  ||a_i - b_j||²
      + mean_j  min_i  ||b_j - a_i||²

    a : (N, 3),  b : (M, 3) — both on the same device.
    """
    sq  = torch.cdist(a, b).pow(2)
    a2b = sq.min(dim=1).values.mean()
    b2a = sq.min(dim=0).values.mean()
    return (a2b + b2a).item()


# ── Marching cubes helpers ────────────────────────────────────────────────────

def build_grid(resolution: int, device: torch.device) -> torch.Tensor:
    """Dense 3-D grid in [-1, 1]^3; shape 1 × (R+1)^3 × 3."""
    lin = np.linspace(-1, 1, resolution + 1).astype(np.float32)
    xv, yv, zv = np.meshgrid(lin, lin, lin)
    pts = np.stack([xv, yv, zv]).reshape(3, -1).T
    return torch.from_numpy(pts)[None].to(device)


@torch.no_grad()
def reconstruct_mesh(latent: torch.Tensor,
                     base_model: torch.nn.Module,
                     grid: torch.Tensor,
                     resolution: int) -> 'trimesh.Trimesh | None':
    gap = 2.0 / resolution
    x   = base_model.learn(latent)

    sdf_parts = []
    for start in range(0, grid.shape[1], 100_000):
        out = base_model.decode(x, grid[:, start:start + 100_000]).squeeze(-1)
        sdf_parts.append(out)
    sdf = torch.cat(sdf_parts, dim=1)

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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Base model ────────────────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}  from  {args.base_pth}")
    base_model = autoencoder.__dict__[args.base_model](pc_size=args.pc_size)
    ckpt = torch.load(args.base_pth, map_location='cpu', weights_only=False)
    base_model.load_state_dict(ckpt['model'], strict=True)
    base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    latent_dim = base_model.bottleneck.pre_bottleneck_proj.out_features
    latent_n   = base_model.num_latents

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

    # ── Load and normalise input mesh (sphere normalisation, same as infer.py) ─
    print(f"Loading input mesh: {args.input}")
    input_mesh = trimesh.load(args.input)

    surface_gt, _ = trimesh.sample.sample_surface_even(input_mesh, count=100_000)
    surface_gt = surface_gt.astype(np.float32)

    shifts = (surface_gt.max(axis=0) + surface_gt.min(axis=0)) / 2
    surface_gt -= shifts
    distances = np.linalg.norm(surface_gt, axis=1)
    scale = 1.0 / np.max(distances)
    surface_gt *= scale

    # Ground-truth points for Chamfer (all 100k, kept on GPU)
    gt_pts = torch.from_numpy(surface_gt).to(device)   # (100k, 3)

    # Point cloud fed to the encoder (random subset of pc_size)
    ind     = np.random.default_rng(args.seed).choice(surface_gt.shape[0], args.pc_size, replace=False)
    surface = torch.from_numpy(surface_gt[ind])[None].to(device)  # 1 × pc_size × 3

    # ── Marching cubes grid ───────────────────────────────────────────────────
    grid = build_grid(args.resolution, device)

    # ── Encode ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        shape_latent  = base_model.encode(surface)['x']   # 1 × latent_n × latent_dim
        hyper_latents = hae.encode(shape_latent)           # 1 × latent_n × latent_dim

    # ── Prefix lengths: powers of 2 from 1 to latent_n ───────────────────────
    prefix_lengths = [2**i for i in range(int(math.log2(latent_n)) + 1)]
    if prefix_lengths[-1] != latent_n:
        prefix_lengths.append(latent_n)
    print(f"Prefix lengths: {prefix_lengths}")

    csv_rows = []

    # ── Baseline (base AE, no HAE) ────────────────────────────────────────────
    mesh_base = reconstruct_mesh(shape_latent, base_model, grid, args.resolution)
    if mesh_base is not None:
        mesh_base.export(os.path.join(args.output_dir, 'baseline.obj'))
        pts_base = torch.from_numpy(
            trimesh.sample.sample_surface(mesh_base, args.n_surface_samples)[0].astype(np.float32)
        ).to(device)
        cd_base = chamfer_distance(pts_base, gt_pts)
    else:
        print("  WARNING: baseline mesh empty")
        cd_base = float('nan')

    csv_rows.append(('baseline', cd_base))
    print(f"  baseline   chamfer = {cd_base:.6f}")

    # ── HAE at each prefix length ─────────────────────────────────────────────
    chamfer_hae = {}
    for L in prefix_lengths:
        with torch.no_grad():
            recon = hae.decode(hyper_latents[:, :L])   # 1 × latent_n × latent_dim

        mesh = reconstruct_mesh(recon, base_model, grid, args.resolution)
        if mesh is not None:
            mesh.export(os.path.join(args.output_dir, f'L{L:04d}.obj'))
            pts = torch.from_numpy(
                trimesh.sample.sample_surface(mesh, args.n_surface_samples)[0].astype(np.float32)
            ).to(device)
            cd = chamfer_distance(pts, gt_pts)
        else:
            print(f"  WARNING: L={L} mesh empty")
            cd = float('nan')

        chamfer_hae[L] = cd
        csv_rows.append((f'L{L:04d}', cd))
        print(f"  L={L:4d}   chamfer = {cd:.6f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prefix_or_baseline', 'chamfer'])
        writer.writerows(csv_rows)
    print(f"\nResults  →  {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(prefix_lengths,
            [chamfer_hae[L] for L in prefix_lengths],
            'o-', color='steelblue', linewidth=2, markersize=5, label='HAE')

    ax.axhline(cd_base, color='darkorange', linewidth=2,
               linestyle='--', label=f'Base AE (no HAE)  [{cd_base:.4f}]')

    ax.set_xscale('log', base=2)
    ax.set_xticks(prefix_lengths)
    ax.set_xticklabels(prefix_lengths, rotation=45, fontsize=8)
    ax.set_xlabel('Prefix length (tokens)')
    ax.set_ylabel('Symmetric Chamfer distance')
    ax.set_title(f'HAE RD Curve — {os.path.basename(args.input)}  (res={args.resolution})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'rd_curve.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"RD curve  →  {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HAE single-shape inference + RD-curve evaluation')

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument('--input',      required=True,  type=str, help='Input .obj file')
    parser.add_argument('--output_dir', default='./infer_hae_out/', type=str)

    # ── Models ────────────────────────────────────────────────────────────────
    parser.add_argument('--base_model', default='learnable_vec1024x32_dim1024_depth24_nb', type=str)
    parser.add_argument('--base_pth',   required=True,  type=str)
    parser.add_argument('--hae_pth',    required=True,  type=str)

    # ── HAE architecture (must match the checkpoint) ──────────────────────────
    parser.add_argument('--working_dim',   default=512, type=int)
    parser.add_argument('--encoder_depth', default=12,   type=int)
    parser.add_argument('--decoder_depth', default=12,   type=int)
    parser.add_argument('--dim_head',      default=64,  type=int)
    parser.add_argument('--min_prefix',    default=1,   type=int)

    # ── Inference settings ────────────────────────────────────────────────────
    parser.add_argument('--pc_size',          default=8192,  type=int)
    parser.add_argument('--resolution',       default=128,   type=int)
    parser.add_argument('--n_surface_samples', default=10000, type=int)
    parser.add_argument('--seed',             default=0,     type=int)

    main(parser.parse_args())
