#!/usr/bin/env python3
"""
Measure inference time for each BACON per-shape run.

For every logs/bacon_{mesh_hash}_{L}/checkpoints/model_final.pth:
  - Read architecture from config.ini (matches init_model() in experiments/train_sdf.py)
  - Build MultiscaleBACON via modules.MultiscaleBACON  (modules.py)
  - Load weights, call generate_mesh() from experiments/render_sdf.py
  - Run marching cubes on the last output layer (highest quality)
  - Record wall-clock time, write timing.json into the run folder
  - Append to logs/inference_times.csv

Run from the bacon project root with the bacon conda environment.
"""

import sys, csv, json, time, configparser, math
from pathlib import Path

BACON_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACON_ROOT))
sys.path.insert(0, str(BACON_ROOT / "experiments"))

import numpy as np
import torch
import mcubes
import trimesh

import modules                                    # modules.py
from render_sdf import generate_mesh              # experiments/render_sdf.py

LOGS_DIR      = BACON_ROOT / "logs"
OUTPUT_LAYERS = [2, 4, 6, 8]
N             = 512   # matches render_sdf.py __main__ (N=512)


def _read_config(run_dir: Path) -> dict:
    # configargparse writes ini files without a [section] header; prepend one
    raw = (run_dir / "config.ini").read_text()
    cfg = configparser.ConfigParser()
    cfg.read_string("[DEFAULT]\n" + raw)
    s = cfg["DEFAULT"]
    return {
        "hidden_size":   int(s.get("hidden_size", 256)),
        "hidden_layers": int(s.get("hidden_layers", 8)),
        "max_freq":      int(s.get("max_freq", 384)),
    }


def _build_model(hidden_size: int, hidden_layers: int, max_freq: int) -> torch.nn.Module:
    """Architecture matching init_model() in experiments/train_sdf.py."""
    frequency    = (max_freq, max_freq, max_freq)
    input_scales = [1/24, 1/24, 1/24, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
    return modules.MultiscaleBACON(
        in_size=3, hidden_size=hidden_size, out_size=1,
        hidden_layers=hidden_layers, bias=True,
        frequency=frequency,
        quantization_interval=2 * math.pi,
        input_scales=input_scales,
        is_sdf=True,
        output_layers=OUTPUT_LAYERS,
        reuse_filters=True,
    )


def _infer(ckpt_path: Path, run_dir: Path):
    """Load model + run generate_mesh (render_sdf.py). Returns (mesh, elapsed_s)."""
    arch = _read_config(run_dir)

    t0 = time.perf_counter()

    model = _build_model(**arch)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.cuda().eval()

    # generate_mesh is from experiments/render_sdf.py; return_sdf=True skips file I/O
    with torch.no_grad():
        sdf_list = generate_mesh(model, N, return_sdf=True, num_outputs=len(OUTPUT_LAYERS))

    # marching cubes on highest-quality output (last layer)
    sdf = sdf_list[-1]
    vertices, triangles = mcubes.marching_cubes(-sdf, 0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.vertices = (mesh.vertices / N - 0.5) + 0.5 / N

    elapsed = time.perf_counter() - t0
    return mesh, elapsed


def main():
    rows = []
    run_dirs = sorted(d for d in LOGS_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("bacon_"))

    for run_dir in run_dirs:
        ckpt_path = run_dir / "checkpoints" / "model_final.pth"
        if not ckpt_path.exists():
            continue

        run_name = run_dir.name
        print(f"  {run_name}  ...", end="", flush=True)
        try:
            _, elapsed = _infer(ckpt_path, run_dir)
            print(f"  {elapsed:.3f}s")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            elapsed = float("nan")

        timing = {"run": run_name, "inference_time_s": elapsed}
        with open(run_dir / "timing.json", "w") as fh:
            json.dump(timing, fh, indent=2)
        rows.append(timing)

    csv_path = LOGS_DIR / "inference_times.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run", "inference_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
