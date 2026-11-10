#!/usr/bin/env python3
"""
Extract inference timing for each instant-ngp parameter-sweep folder.

instant-ngp does not save model snapshots, so training + marching-cubes cannot
be replayed from a checkpoint.  Instead this script parses the HH:MM:SS
timestamps in each folder's train.log to compute total elapsed time (first line
→ last line) and writes the result into timing.json / inference_times.csv.

For every parameter_sweep_results/log2_{T}/03001627/{mesh_hash}/:
  - Parse train.log for first and last timestamp
  - Compute elapsed wall-clock seconds
  - Write timing.json into the folder
  - Append to parameter_sweep_results/inference_times.csv

No conda environment needed (pure stdlib + re).
"""

import re, csv, json
from datetime import datetime
from pathlib import Path

NGP_ROOT  = Path(__file__).parent
SWEEP_DIR = NGP_ROOT / "parameter_sweep_results"

_TS_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})")


def _parse_elapsed(log_path: Path):
    """Return elapsed seconds between first and last timestamped line, or nan."""
    first = last = None
    try:
        with open(log_path, "r", errors="replace") as fh:
            for raw_line in fh:
                # strip ANSI escape codes before matching
                line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line)
                m = _TS_RE.match(line.strip())
                if m:
                    ts = datetime.strptime(m.group(1), "%H:%M:%S")
                    if first is None:
                        first = ts
                    last = ts
    except Exception:
        return float("nan")

    if first is None or last is None:
        return float("nan")

    diff = (last - first).total_seconds()
    # handle log spanning midnight
    if diff < 0:
        diff += 86400
    return diff


def main():
    rows = []

    log2_dirs = sorted(
        (d for d in SWEEP_DIR.iterdir() if d.is_dir() and d.name.startswith("log2_")),
        key=lambda d: int(d.name.split("_")[1]),
    )

    for log2_dir in log2_dirs:
        log2_val = int(log2_dir.name.split("_")[1])
        # category level (e.g. 03001627)
        for cat_dir in sorted(d for d in log2_dir.iterdir() if d.is_dir()):
            for shape_dir in sorted(d for d in cat_dir.iterdir() if d.is_dir()):
                log_path = shape_dir / "train.log"
                if not log_path.exists():
                    continue

                mesh_hash = shape_dir.name
                elapsed   = _parse_elapsed(log_path)
                status    = f"{elapsed:.1f}s" if elapsed == elapsed else "no timestamps"
                print(f"  log2={log2_val:2d}  {mesh_hash}  →  {status}")

                timing = {
                    "log2_hashmap_size": log2_val,
                    "mesh_hash":         mesh_hash,
                    "inference_time_s":  elapsed,
                    "note": "total training+marching-cubes time from train.log timestamps",
                }
                with open(shape_dir / "timing.json", "w") as fh:
                    json.dump(timing, fh, indent=2)

                rows.append({
                    "log2_hashmap_size": log2_val,
                    "mesh_hash":         mesh_hash,
                    "inference_time_s":  elapsed,
                })

    csv_path = SWEEP_DIR / "inference_times.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["log2_hashmap_size", "mesh_hash", "inference_time_s"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
