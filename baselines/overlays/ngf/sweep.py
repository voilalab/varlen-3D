#!/usr/bin/env python3
"""
NGF LOD Sweep - Trains on validation meshes at varying LOD levels.
"""

import json
import os
import shutil
import sys
import subprocess
import warnings
from pathlib import Path

import torch
import trimesh

# Paths
NGF_ROOT = Path(__file__).parent
SOURCE_ROOT = NGF_ROOT / "source"
DATA_DIR = NGF_ROOT.resolve().parents[1] / "data" / "shapenet"  # <repo>/data/shapenet
VAL_SPLIT = DATA_DIR / "val.lst"
TRAIN_SPLIT = DATA_DIR / "train.lst"
OUTPUT_DIR = NGF_ROOT / "parameter_sweep_results"

# Default sweep parameters (higher LOD for ShapeNet - better reconstruction)
DEFAULT_LODS = [5000, 4000, 2000, 1000, 500, 250, 125, 64, 32, 16, 8, 4]
FEATURES = 40
BATCH = 10


def count_mesh_params(mesh_path):
    """Count mesh parameters: vertices*3 + faces."""
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh", skip_materials=True)
        return 3 * len(mesh.vertices) + len(mesh.faces)
    except Exception:
        return None


def prepare_mesh_for_ngf(original_path, out_path):
    """
    Convert OBJ to STL via trimesh. NGF uses meshio which crashes on ShapeNet OBJs
    with inconsistent obj:vt (texture) or obj:vn (normal) data.
    """
    try:
        m = trimesh.load(str(original_path), force="mesh", skip_materials=True)
        if m is None or len(m.vertices) == 0:
            return None
        if m.faces is not None and len(m.faces) > 0 and m.faces.shape[1] != 3:
            m = m.triangulate()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        m.export(str(out_path))
        return out_path
    except Exception as e:
        print(f"    Preprocess error: {e}")
        return None


def get_mesh_paths(split_file, data_dir, max_meshes=None):
    """Get mesh paths from split file (format: category/model_id)."""
    if not split_file.exists():
        return []
    with open(split_file) as f:
        mesh_ids = [line.strip() for line in f if line.strip()]
    paths = []
    for mesh_id in mesh_ids:
        mesh_path = data_dir / mesh_id / "models" / "model_normalized.obj"
        if mesh_path.exists():
            paths.append((mesh_id, mesh_path))
            if max_meshes and len(paths) >= max_meshes:
                break
    return paths


def run_training(mesh_path, lod, features=FEATURES, batch=BATCH, timeout=3600, quad_timeout=180):
    """Run NGF train.py."""
    cmd = [
        sys.executable,
        str(SOURCE_ROOT / "train.py"),
        "--mesh", str(mesh_path),
        "--lod", str(lod),
        "--features", str(features),
        "--batch", str(batch),
        "--quad-timeout", str(quad_timeout),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SOURCE_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  Training warning (code {result.returncode})")
            if result.stderr:
                print(result.stderr[-500:])
        return True
    except subprocess.TimeoutExpired:
        print("  Training timed out")
        return False
    except Exception as e:
        print(f"  Training error: {e}")
        return False


def find_trained_model(mesh_path, lod, features=FEATURES):
    """Find the .pt file after training."""
    mesh_name = Path(mesh_path).stem
    basename = f"{mesh_name}-lod{lod}-f{features}"
    pt_path = SOURCE_ROOT / "results" / "torched" / f"{basename}.pt"
    return pt_path if pt_path.exists() else None


def find_trained_mesh(mesh_path, lod, features=FEATURES):
    """Find the .stl mesh file after training."""
    mesh_name = Path(mesh_path).stem
    basename = f"{mesh_name}-lod{lod}-f{features}"
    stl_path = SOURCE_ROOT / "results" / "stl" / f"{basename}.stl"
    return stl_path if stl_path.exists() else None


def count_ngf_model_params(pt_path):
    """Count parameters in a trained NGF .pt file."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(pt_path, map_location="cpu")
        mlp_params = sum(p.numel() for p in data["model"].parameters())
        points_params = data["points"].numel()
        features_params = data["features"].numel()
        return mlp_params + points_params + features_params
    except Exception:
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NGF LOD Sweep")
    parser.add_argument(
        "--lods",
        type=str,
        default=",".join(map(str, DEFAULT_LODS)),
        help="Comma-separated LOD values",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Which split to use",
    )
    parser.add_argument(
        "--max-meshes",
        type=int,
        default=None,
        help="Max meshes to process",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (e.g. for post-processing only)",
    )
    parser.add_argument(
        "--quad-timeout",
        type=int,
        default=600,
        help="Quadrangulation timeout in seconds (default 180 for ShapeNet)",
    )
    args = parser.parse_args()

    lods = [int(x) for x in args.lods.split(",")]
    split_file = VAL_SPLIT if args.split == "val" else TRAIN_SPLIT
    meshes = get_mesh_paths(split_file, DATA_DIR, args.max_meshes)

    if not meshes:
        print(f"ERROR: No meshes found in {split_file}")
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("NGF LOD Sweep")
    print("=" * 70)
    print(f"LOD values: {lods}")
    print(f"Meshes: {len(meshes)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    for lod in lods:
        lod_dir = OUTPUT_DIR / f"lod_{lod}"
        lod_dir.mkdir(exist_ok=True)
        print(f"\n--- LOD = {lod} ---")

        lod_results = []

        for mesh_id, mesh_path in meshes:
            mesh_name = mesh_id.split("/")[-1]
            mesh_out = lod_dir / mesh_name
            mesh_out.mkdir(exist_ok=True)

            # Preprocess OBJ -> STL to avoid meshio crashes on ShapeNet OBJs
            stl_prep_path = mesh_out / f"{mesh_name}.stl"
            train_mesh = prepare_mesh_for_ngf(mesh_path, stl_prep_path)
            if train_mesh is None:
                print(f"  Skipping {mesh_name}: preprocess failed")
                continue

            if not args.skip_training:
                print(f"  Training {mesh_name}...")
                run_training(train_mesh, lod, quad_timeout=args.quad_timeout)

            pt_path = find_trained_model(train_mesh, lod)
            mesh_params = count_mesh_params(mesh_path)
            model_params = count_ngf_model_params(pt_path) if pt_path else None
            param_ratio = (
                model_params / mesh_params if (model_params and mesh_params) else None
            )

            info = {
                "mesh_id": mesh_id,
                "lod": lod,
                "mesh_params": mesh_params,
                "model_params": model_params,
                "param_ratio": param_ratio,
            }

            if pt_path:
                shutil.copy(pt_path, mesh_out / "model.pt")
                shutil.copy(mesh_path, mesh_out / "original.obj")
                stl_path = find_trained_mesh(train_mesh, lod)
                if stl_path:
                    shutil.copy(stl_path, mesh_out / "reconstructed.stl")
                with open(mesh_out / "info.json", "w") as f:
                    json.dump(info, f, indent=2)
                ratio_str = f", ratio={param_ratio:.4f}" if param_ratio else ""
                print(f"  {mesh_name}: saved (model + mesh){ratio_str}")
                if param_ratio:
                    lod_results.append(info)
            else:
                print(f"  No model found for {mesh_name}")

        # Save LOD summary with parameter ratios
        if lod_results:
            summary_path = lod_dir / "results.json"
            with open(summary_path, "w") as f:
                json.dump(
                    {
                        "lod": lod,
                        "num_meshes": len(lod_results),
                        "avg_param_ratio": sum(r["param_ratio"] for r in lod_results) / len(lod_results),
                        "meshes": lod_results,
                    },
                    f,
                    indent=2,
                )
            print(f"  Summary: {summary_path}")

    # Save overall summary with param ratios per LOD
    all_summaries = []
    for lod in lods:
        summary_path = OUTPUT_DIR / f"lod_{lod}" / "results.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
                all_summaries.append(
                    {"lod": lod, "avg_param_ratio": s["avg_param_ratio"], "num_meshes": s["num_meshes"]}
                )
    if all_summaries:
        with open(OUTPUT_DIR / "param_ratios_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nParam ratios summary: {OUTPUT_DIR / 'param_ratios_summary.json'}")

    print(f"\nResults in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
