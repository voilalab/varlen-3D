#!/usr/bin/env python3
"""
Measure inference time for each CON parameter-sweep folder.

For every parameter_sweep_results/cdim_{c}/:
  - Load config + model + generator via src/config.py  (same as generate.py)
  - Load model_best.pt via CheckpointIO  (same as generate.py)
  - Time generator.generate_mesh(data) on the first test batch
  - Write timing.json into the cdim folder
  - Append to parameter_sweep_results/inference_times.csv

Run from the convolutional_occupancy_networks root with the CON conda environment.
"""

import sys, csv, json, time
from pathlib import Path

CON_ROOT = Path(__file__).parent
sys.path.insert(0, str(CON_ROOT))

import torch
from src import config                     # src/config.py  (same as generate.py)
from src.checkpoints import CheckpointIO   # src/checkpoints.py  (same as generate.py)

SWEEP_DIR = CON_ROOT / "parameter_sweep_results"


def _infer_cdim(cdim_dir: Path, device: torch.device):
    """Load model for one cdim, time generate_mesh on first test batch."""
    config_path = cdim_dir / "config.yaml"
    training_dir = cdim_dir / "training"
    if not config_path.exists() or not training_dir.exists():
        return float("nan")

    # --- mirror generate.py setup ---
    cfg = config.load_config(str(config_path), str(CON_ROOT / "configs" / "default.yaml"))
    cfg["training"]["out_dir"] = str(training_dir)

    dataset  = config.get_dataset("test", cfg, return_idx=True)
    model    = config.get_model(cfg, device=device, dataset=dataset)

    checkpoint_io = CheckpointIO(str(training_dir), model=model)
    try:
        checkpoint_io.load(cfg["test"]["model_file"])
    except Exception as exc:
        print(f"    checkpoint load failed: {exc}")
        return float("nan")

    generator = config.get_generator(model, cfg, device=device)
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    data   = next(iter(loader))

    # --- timing: one generate_mesh call (mirrors generate.py lines 142-149) ---
    t0 = time.perf_counter()
    try:
        out = generator.generate_mesh(data)
    except Exception as exc:
        print(f"    generate_mesh failed: {exc}")
        return float("nan")
    elapsed = time.perf_counter() - t0

    return elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    cdim_dirs = sorted(
        (d for d in SWEEP_DIR.iterdir() if d.is_dir() and d.name.startswith("cdim_")),
        key=lambda d: int(d.name.split("_")[1]),
    )

    for cdim_dir in cdim_dirs:
        cdim = int(cdim_dir.name.split("_")[1])
        print(f"  cdim={cdim:3d}  ...", end="", flush=True)
        try:
            elapsed = _infer_cdim(cdim_dir, device)
            print(f"  {elapsed:.3f}s" if elapsed == elapsed else "  nan")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            elapsed = float("nan")

        timing = {"cdim": cdim, "inference_time_s": elapsed}
        with open(cdim_dir / "timing.json", "w") as fh:
            json.dump(timing, fh, indent=2)
        rows.append(timing)

    csv_path = SWEEP_DIR / "inference_times.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cdim", "inference_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
