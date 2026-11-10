#!/usr/bin/env python3
"""
Graph MEDIAN Chamfer distance vs compression ratio for all methods.
No meshes are filtered — every mesh with data is included.

Marker area at each point is proportional to cumulative wall-clock seconds for
evaluating the ladder 1..k; see inference_timing.py.
Only methods with an inference_times.csv get scaled markers: the provided curves
ship theirs in distances/, the baselines' are written by their time_inference.py
under Projects/. Methods without one are drawn at a constant marker size.
"""

import math
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from inference_timing import marker_sizes_for_method

DIST_DIR = Path(__file__).resolve().parent.parent / "distances"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
LEGEND_LABEL = {"MeshGPT_HAE": "MeshGPT (Ours)",
                "VecSet_HAE": "VecSet (Ours)"}

METHODS = {
    "NGP":       {"dir": "ngp",        "pattern": r"log2_(\d+)_distance\.txt"},
    "NGF":       {"dir": "ngf",        "pattern": r"lod_(\d+)_distance\.txt"},
    "CONs":      {"dir": "CONs",       "pattern": r"cdim_(\d+)_distance\.txt"},
    "BACON":     {"dir": "bacon",      "pattern": r"level_(\d+)_distance\.txt"},
    "MeshGPT_HAE": {"dir": "MeshGPT_HAE", "pattern": r"keepk_(\d+)_distance\.txt"},
    "VecSet_HAE": {"dir": "VecSet_HAE", "pattern": r"ratio_L(\d+)\.txt"},
}

LINE_RE = re.compile(
    r"Chamfer Distance for (.+?)(?:\.\w+)? with ratio ([\d.]+): ([\d.e+-]+|nan)"
)


def parse_file(path):
    """Return list of (mesh_id, ratio, cd). NaN Chamfer values are dropped."""
    entries = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                try:
                    cd = float(m.group(3))
                except ValueError:
                    continue
                if math.isnan(cd):
                    continue
                entries.append((m.group(1), float(m.group(2)), cd))
    return entries


def load_method(name, cfg):
    method_dir = DIST_DIR / cfg["dir"]
    pat = re.compile(cfg["pattern"])
    configs = []
    for txt in sorted(method_dir.glob("*.txt")):
        m = pat.match(txt.name)
        if not m:
            continue
        entries = parse_file(txt)
        if not entries:
            continue
        param = int(m.group(1))
        ratio = float(np.mean([e[1] for e in entries]))
        cds = [cd for _, _, cd in entries]
        configs.append((param, ratio, cds))
    configs.sort(key=lambda x: x[1])
    return configs


def _plot_curve(ax, method_name, configs, y_fn, colors, markers):
    if not configs:
        return
    params = [p for p, _, _ in configs]
    ratios = [r for _, r, _ in configs]
    ys = [y_fn(cds) for _, _, cds in configs]
    color = colors[method_name]
    marker = markers[method_name]
    label = LEGEND_LABEL.get(method_name, method_name)

    ax.plot(ratios, ys, linestyle="-", linewidth=1.3, color=color, label=label, zorder=2)
    sizes = marker_sizes_for_method(method_name, params)
    if sizes is None:
        sizes = [25.0] * len(ratios)
    ax.scatter(
        ratios, ys, s=sizes, marker=marker, color=color,
        edgecolors="white", linewidths=0.5, zorder=3,
    )


def main():
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {
        "NGP": "#1f77b4", "NGF": "#ff7f0e", "CONs": "#2ca02c", "BACON": "#d62728",
        "MeshGPT_HAE": "#9467bd", "VecSet_HAE": "#8c564b",
    }
    markers = {
        "NGP": "o", "NGF": "s", "CONs": "^", "BACON": "D",
        "MeshGPT_HAE": "v", "VecSet_HAE": "P",
    }

    for method_name, cfg in METHODS.items():
        configs = load_method(method_name, cfg)
        _plot_curve(ax, method_name, configs, np.median, colors, markers)
        if configs:
            print(f"{method_name}:")
            for p, r, cds in configs:
                print(f"  k={p} ratio={r:.4f}  median={np.median(cds):.6f}  mean={np.mean(cds):.6f}  n={len(cds)}")

    orig_file = DIST_DIR / "MeshGPT_orig1" / "distance.txt"
    if orig_file.is_file():
        entries = parse_file(orig_file)
        if entries:
            ratio = float(np.mean([e[1] for e in entries]))
            cds = [cd for _, _, cd in entries]
            y = np.median(cds)
            sizes = marker_sizes_for_method("MeshGPT_orig", [0]) or [120.0]
            ax.scatter(
                [ratio], [y], s=sizes[0], marker="*",
                color="#e377c2", label="MeshGPT (original)", zorder=4,
                edgecolors="black", linewidths=0.5,
            )
            print("MeshGPT_orig:")
            print(f"  ratio={ratio:.4f}  median={y:.6f}  n={len(cds)}")

    vs_gt_file = DIST_DIR / "VecSet_GT" / "ratio.txt"
    if vs_gt_file.is_file():
        entries = parse_file(vs_gt_file)
        if entries:
            ratio = float(np.mean([e[1] for e in entries]))
            cds = [cd for _, _, cd in entries]
            y = np.median(cds)
            ax.scatter(
                [ratio], [y], marker="*", s=120,
                color="#7f7f7f", label="VecSet (original)", zorder=4,
                edgecolors="black", linewidths=0.5,
            )
            print("VecSet_GT:")
            print(f"  ratio={ratio:.4f}  median={y:.6f}  n={len(cds)}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(10**0, color="gray", linestyle="--", linewidth=1.2, alpha=0.85, zorder=0)
    ax.set_xlabel("Compression Ratio (payload bits / mesh bits)", fontsize=13)
    ax.set_ylabel("Chamfer Distance (median)", fontsize=13)
    ax.set_title(
        "Median Chamfer Distance vs Compression Ratio\n"
        "(marker area ∝ cumulative wall-clock for ladder 1..k)",
        fontsize=14,
    )
    ax.legend(fontsize=11, title="dot area = cum. time 1..k", title_fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "graph_median_cd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
