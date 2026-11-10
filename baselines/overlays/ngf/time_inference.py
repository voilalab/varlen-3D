#!/usr/bin/env python3
"""
Measure inference time for each NGF parameter-sweep folder.

For every parameter_sweep_results/lod_{lod}/{mesh_hash}/model.pt:
  - Load via NGF.from_pt  (source/ngf.py)
  - Run the same pipeline as Trainer.export()  (source/train.py lines 185-195):
      sample_uniform → eval → base → make_cmap → ngfutil remapper + triangulate
  - Record wall-clock time, write timing.json into the folder
  - Append to parameter_sweep_results/inference_times.csv

Run with the ngf conda environment:
    conda run -n ngf python time_inference.py
"""

import sys, csv, json, time, warnings
from pathlib import Path

NGF_ROOT    = Path(__file__).parent
SOURCE_ROOT = NGF_ROOT / "source"
sys.path.insert(0, str(SOURCE_ROOT))

import torch
import trimesh
import ngfutil
from ngf import NGF                          # source/ngf.py
from util.miscellaneous import make_cmap     # source/util/miscellaneous.py

SWEEP_DIR = NGF_ROOT / "parameter_sweep_results"
RATE = 16   # matches Trainer.export() in source/train.py


@torch.no_grad()
def _infer(pt_path: Path):
    """Exact reconstruction from Trainer.export() (source/train.py lines 185-195)."""
    t0 = time.perf_counter()

    ngf   = NGF.from_pt(str(pt_path))
    uvs   = ngf.sample_uniform(RATE)
    V     = ngf.eval(*uvs).detach()
    base  = ngf.base(RATE).detach()
    cmap  = make_cmap(ngf.complexes, ngf.points.detach(), base, RATE)
    remap = ngfutil.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], RATE)
    faces = ngfutil.triangulate_shorted(V, ngf.complexes.shape[0], RATE)
    faces = remap.remap_device(faces)

    elapsed = time.perf_counter() - t0
    return trimesh.Trimesh(vertices=V.cpu().numpy(), faces=faces.cpu().numpy()), elapsed


def main():
    rows = []
    lod_dirs = sorted(
        (d for d in SWEEP_DIR.iterdir() if d.is_dir() and d.name.startswith("lod_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    for lod_dir in lod_dirs:
        lod = int(lod_dir.name.split("_")[1])
        for shape_dir in sorted(d for d in lod_dir.iterdir() if d.is_dir()):
            pt_path = shape_dir / "model.pt"
            if not pt_path.exists():
                continue
            mesh_hash = shape_dir.name
            print(f"  lod={lod:5d}  {mesh_hash}  ...", end="", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                _, elapsed = _infer(pt_path)
                print(f"  {elapsed:.3f}s")
            except Exception as exc:
                print(f"  ERROR: {exc}")
                elapsed = float("nan")

            timing = {"lod": lod, "mesh_hash": mesh_hash, "inference_time_s": elapsed}
            with open(shape_dir / "timing.json", "w") as fh:
                json.dump(timing, fh, indent=2)
            rows.append(timing)

    csv_path = SWEEP_DIR / "inference_times.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["lod", "mesh_hash", "inference_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
