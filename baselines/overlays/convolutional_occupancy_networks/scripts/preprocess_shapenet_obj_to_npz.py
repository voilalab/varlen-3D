#!/usr/bin/env python3
"""
Build points.npz + pointcloud.npz for ConvONet / CONs training from ShapeNet-style
folders that only have a mesh (e.g. models/model_normalized.obj).

Writes, per model directory:
  - points.npz: keys "points" (N,3), "occupancies" (np.packbits vector)
  - pointcloud.npz: keys "points", "normals" (surface samples)

Coordinates are normalized to roughly [-0.5, 0.5]^3 (centered, uniform scale),
then occupancy query points use the same cube expansion as common ONet data:
  (rand(3) - 0.5) * (1 + padding)

Requires watertight (or mostly watertight) meshes for meaningful inside/outside
labels (trimesh.winding_number or mesh.contains).

Example:
  python scripts/preprocess_shapenet_obj_to_npz.py \\
    --dataset-root /path/to/data/shapenet \\
    --split-list /path/to/data/shapenet/train.lst \\
    --category 03001627
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

# CONs repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parents[1]


def find_mesh(model_dir: Path) -> Path | None:
    for rel in (
        Path("models") / "model_normalized.obj",
        Path("models") / "model.obj",
        Path("model_normalized.obj"),
        Path("model.obj"),
    ):
        p = model_dir / rel
        if p.is_file():
            return p
    return None


def normalize_mesh_to_unit_cube(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Scale mesh into ~[-0.5, 0.5]^3 (matches typical occupancy network data)."""
    m = mesh.copy()
    m.vertices = m.vertices.astype(np.float64)
    center = (m.bounds[0] + m.bounds[1]) / 2.0
    m.vertices -= center
    mx = np.abs(m.vertices).max()
    if mx < 1e-12:
        return m
    m.vertices /= 2.0 * mx
    return m


def sample_occupancy_points(mesh: trimesh.Trimesh, n_points: int, padding: float):
    pts = (np.random.rand(n_points, 3).astype(np.float32) - 0.5) * (1.0 + padding)
    try:
        inside = mesh.contains(pts)
    except Exception:
        inside = np.zeros(len(pts), dtype=bool)
        # winding_number often works when contains fails
        try:
            inside = mesh.winding_number(pts) > 0.5
        except Exception:
            raise RuntimeError(
                "Could not classify inside/outside; mesh may be non-watertight."
            ) from None
    return pts, inside.astype(bool)


def process_model(model_dir: Path, n_vol: int, n_surf: int, padding: float, overwrite: bool) -> bool:
    out_points = model_dir / "points.npz"
    out_pcl = model_dir / "pointcloud.npz"
    if not overwrite and out_points.is_file() and out_pcl.is_file():
        return True

    mesh_path = find_mesh(model_dir)
    if mesh_path is None:
        return False

    loaded = trimesh.load(str(mesh_path), process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return False
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        return False

    mesh = normalize_mesh_to_unit_cube(mesh)
    if not mesh.is_watertight:
        try:
            mesh.fill_holes()
        except Exception:
            pass

    vol_pts, occ = sample_occupancy_points(mesh, n_vol, padding)
    occ_u8 = occ.astype(np.uint8)
    packed = np.packbits(occ_u8)

    surf_pts, face_idx = mesh.sample(n_surf, return_index=True)
    normals = mesh.face_normals[face_idx].astype(np.float32)

    np.savez(
        out_points,
        points=vol_pts,
        occupancies=packed,
    )
    np.savez(
        out_pcl,
        points=surf_pts.astype(np.float32),
        normals=normals,
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split-list",
        type=Path,
        required=True,
        help=".lst file with lines synset/model_id or model_id (use --category)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="If split lines are bare model hashes, set synset id (e.g. 03001627).",
    )
    parser.add_argument("--n-occupancy", type=int, default=100000)
    parser.add_argument("--n-surface", type=int, default=100000)
    parser.add_argument("--padding", type=float, default=0.1, help="Volume padding (config data.padding)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    lines = [
        ln.strip()
        for ln in args.split_list.read_text().splitlines()
        if ln.strip()
    ]

    jobs: list[tuple[str, str]] = []
    for line in lines:
        parts = line.split("/")
        if len(parts) == 2:
            cat, mid = parts
        elif len(parts) == 1 and args.category:
            cat, mid = args.category, parts[0]
        else:
            print(f"Skip line (need category/model_id or --category): {line}", file=sys.stderr)
            continue
        jobs.append((cat, mid))

    failed: list[str] = []
    for cat, mid in tqdm(jobs, desc="models"):
        model_dir = root / cat / mid
        if not model_dir.is_dir():
            failed.append(f"missing dir {model_dir}")
            continue
        try:
            ok = process_model(
                model_dir,
                n_vol=args.n_occupancy,
                n_surf=args.n_surface,
                padding=args.padding,
                overwrite=args.overwrite,
            )
            if not ok:
                failed.append(f"no mesh {model_dir}")
        except Exception as e:
            failed.append(f"{model_dir}: {e}")

    if failed:
        print(f"\n{len(failed)} failures (showing up to 20):", file=sys.stderr)
        for msg in failed[:20]:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    sys.exit(main() or 0)
