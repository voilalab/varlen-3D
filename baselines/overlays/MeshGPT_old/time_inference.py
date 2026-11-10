#!/usr/bin/env python3
"""
Measure inference time for MeshGPT_old vocabulary eval.

Replicates the eval loop from on_validation_epoch_end() in
trainer/train_vocabulary.py: for every val shape, runs the full
encoder → pre_quant → VQ → post_quant → decoder pipeline and
records wall-clock time per shape.

Uses:
  - TriangleTokenizationGraphConv   (trainer/train_vocabulary.py)
  - TriangleNodesWithFaces          (dataset/triangles.py)
  - pretrained/vocabulary/checkpoints/749-0.ckpt

Writes:
  pretrained/inference_times.csv   — per-shape timing
  pretrained/timing.json           — summary stats

Run from the MeshGPT_old root with the meshgpt conda environment.
"""

import sys, csv, json, time
from pathlib import Path

MESHGPT_ROOT = Path(__file__).parent
sys.path.insert(0, str(MESHGPT_ROOT))

import torch
import omegaconf

from trainer.train_vocabulary import TriangleTokenizationGraphConv   # trainer/train_vocabulary.py
from dataset.triangles import TriangleNodesWithFaces                 # dataset/triangles.py

CONFIG_PATH = MESHGPT_ROOT / "runs" / "vocabulary" / "config.yaml"
CKPT_PATH   = MESHGPT_ROOT / "pretrained" / "vocabulary" / "checkpoints" / "749-0.ckpt"
CATEGORY    = "03001627"


@torch.no_grad()
def _infer_one(model, data, device):
    """Encode-decode one shape. Mirrors on_validation_epoch_end() lines 218-235."""
    batch = torch.zeros([data.x.shape[0]], device=device, dtype=torch.long)

    encoded_x = model.encoder(data.x.to(device), data.edge_index.to(device), batch)
    encoded_x = encoded_x.reshape(encoded_x.shape[0] * 3, 192)
    encoded_x = model.distribute_features_fn(
        encoded_x, data.faces.to(device), data.num_vertices, device
    )
    encoded_x = model.pre_quant(encoded_x)
    encoded_x, _, _ = model.vq(encoded_x.unsqueeze(0))
    encoded_x = encoded_x.squeeze(0)
    encoded_x = encoded_x.reshape(-1, 3 * encoded_x.shape[-1])
    encoded_x = model.post_quant(encoded_x)
    encoded_x_conv, conv_mask = model.create_conv_batch(encoded_x, batch, 1)
    decoded_x_conv = model.decoder(encoded_x_conv)
    return decoded_x_conv, conv_mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CONFIG_PATH.exists():
        sys.exit(f"Config not found: {CONFIG_PATH}")
    if not CKPT_PATH.exists():
        sys.exit(f"Checkpoint not found: {CKPT_PATH}")

    cfg = omegaconf.OmegaConf.load(CONFIG_PATH)

    # Load model from existing class (trainer/train_vocabulary.py)
    model = TriangleTokenizationGraphConv(cfg)
    ckpt  = torch.load(str(CKPT_PATH), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # Val dataset (dataset/triangles.py)
    dataset = TriangleNodesWithFaces(
        cfg, "val",
        scale_augment=False, shift_augment=False,
        force_category=CATEGORY,
    )
    print(f"Val shapes: {len(dataset)}")

    rows = []
    for k in range(len(dataset)):
        data = dataset.get(k)
        shape_name = dataset.names[k]
        print(f"  [{k+1}/{len(dataset)}]  {shape_name}  ...", end="", flush=True)
        try:
            t0 = time.perf_counter()
            _infer_one(model, data, device)
            elapsed = time.perf_counter() - t0
            print(f"  {elapsed:.3f}s")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            elapsed = float("nan")

        timing = {"shape": shape_name, "inference_time_s": elapsed}
        rows.append(timing)

    # Write CSV
    out_dir  = MESHGPT_ROOT / "pretrained"
    csv_path = out_dir / "inference_times.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["shape", "inference_time_s"])
        writer.writeheader()
        writer.writerows(rows)

    valid_times = [r["inference_time_s"] for r in rows if r["inference_time_s"] == r["inference_time_s"]]
    summary = {
        "n_shapes":          len(rows),
        "mean_time_s":       sum(valid_times) / len(valid_times) if valid_times else float("nan"),
        "total_time_s":      sum(valid_times),
        "ckpt":              str(CKPT_PATH),
    }
    json_path = out_dir / "timing.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved → {csv_path}")
    print(f"Saved → {json_path}")
    print(f"Mean inference time: {summary['mean_time_s']:.3f}s")


if __name__ == "__main__":
    main()
