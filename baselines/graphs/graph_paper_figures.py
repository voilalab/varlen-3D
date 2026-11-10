#!/usr/bin/env python3
"""
Paper figures: normalized Chamfer distance vs compression ratio, split by
output representation.

  Figure 1 (implicit)  SDF / implicit-field methods: NGP, BACON, CONs, VecSet.
  Figure 2 (mesh)      explicit-mesh methods:        NGF, MeshGPT.

Both panels share the same axes limits so they can be placed side by side.
Chamfer values in distances/*/ are already normalized (both point sets are
independently rescaled to the unit cube, 10,000 sampled points each, ICP-aligned;
see metrics/distance.py).

X-axis per curve point: the ratio stamped in that distance file (every line of a
file carries the same aggregate ratio; the mean over lines is taken defensively).
NaN Chamfer values are dropped, so a setting with no finite value (VecSet-HAE
L1, L2) is not drawn, and a partly-NaN one is aggregated over its finite meshes.

Markers are a uniform size. Pass --time-markers to instead scale marker area by
the cumulative wall-clock seconds for evaluating the ladder 1..k; see
inference_timing.py.

Styling follows the existing paper figures (the "tokens kept k" rate curves):
DejaVu Sans, full box spines, major-only grid, tab10 colors, filled o/s/^/D/P
markers on 2.5pt lines, framed legend inside the axes, and the paper's down-arrow
axis labels marking which direction is better. Both axes stay logarithmic here --
unlike the paper's linear-y panels -- because these curves span four decades of
Chamfer distance and five of compression ratio.

Outputs (PNG + PDF) go to figures/paper_figures/.
"""

import argparse
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from inference_timing import marker_sizes_for_method

DIST_DIR = Path(__file__).resolve().parent.parent / "distances"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "paper_figures"

# Matches the "tokens kept k" figures already in the paper.
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#b0b0b0",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.55,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

STYLE = {
    "NGP":        {"color": "#1f77b4", "marker": "o", "label": "NGP"},
    "BACON":      {"color": "#d62728", "marker": "D", "label": "BACON"},
    "CONs":       {"color": "#2ca02c", "marker": "^", "label": "CONs"},
    "VecSet_HAE": {"color": "#8c564b", "marker": "P", "label": "VecSet (Ours)"},
    "NGF":        {"color": "#ff7f0e", "marker": "s", "label": "NGF"},
    "MeshGPT_HAE": {"color": "#9467bd", "marker": "v", "label": "MeshGPT (Ours)"},
}

# Curves: method -> (subdir, filename pattern capturing the ladder parameter).
CURVES = {
    "NGP":        ("ngp",        r"log2_(\d+)_distance\.txt"),
    "BACON":      ("bacon",      r"level_(\d+)_distance\.txt"),
    "CONs":       ("CONs",       r"cdim_(\d+)_distance\.txt"),
    "VecSet_HAE": ("VecSet_HAE", r"ratio_L(\d+)\.txt"),
    "NGF":        ("ngf",        r"lod_(\d+)_distance\.txt"),
    "MeshGPT_HAE": ("MeshGPT_HAE", r"keepk_(\d+)_distance\.txt"),
}

# Single-point references: the unmodified upstream models.
REFERENCES = {
    "VecSet_orig":  {"path": DIST_DIR / "VecSet_GT/ratio.txt",
                     "label": "VecSet (original)", "color": "#7f7f7f"},
    "MeshGPT_orig": {"path": DIST_DIR / "MeshGPT_orig1/distance.txt",
                     "label": "MeshGPT (original)", "color": "#e377c2"},
}

FIGURES = {
    "implicit": {
        "curves": ["NGP", "BACON", "CONs", "VecSet_HAE"],
        "refs": ["VecSet_orig"],
        "title": "SDF",
        "stem": "fig_rd_implicit",
        "legend_loc": "upper right",
    },
    "mesh": {
        "curves": ["NGF", "MeshGPT_HAE"],
        "refs": ["MeshGPT_orig"],
        "title": "Mesh",
        "stem": "fig_rd_mesh",
        "legend_loc": "upper right",
    },
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


def load_curve(subdir, pattern, keep_ids=None):
    """Return ratio-sorted [(param, mean ratio, [cd, ...]), ...] for one method."""
    method_dir = DIST_DIR / subdir
    pat = re.compile(pattern)
    configs = []
    for txt in sorted(method_dir.glob("*.txt")):
        m = pat.match(txt.name)
        if not m:
            continue
        entries = parse_file(txt)
        if keep_ids is not None:
            entries = [e for e in entries if e[0] in keep_ids]
        if not entries:
            continue
        configs.append((
            int(m.group(1)),
            float(np.mean([r for _, r, _ in entries])),
            [cd for _, _, cd in entries],
        ))
    configs.sort(key=lambda x: x[1])
    return configs


def load_reference(path, keep_ids=None):
    """Return (ratio, [cd, ...]) for a single-point reference, or None."""
    if not Path(path).is_file():
        return None
    entries = parse_file(path)
    if keep_ids is not None:
        entries = [e for e in entries if e[0] in keep_ids]
    if not entries:
        return None
    return (float(np.mean([r for _, r, _ in entries])),
            [cd for _, _, cd in entries])


def common_mesh_ids():
    """Mesh ids evaluated by every method, so all curves cover the same shapes."""
    ids = None
    for subdir, pattern in CURVES.values():
        pat = re.compile(pattern)
        for txt in sorted((DIST_DIR / subdir).glob("*.txt")):
            if not pat.match(txt.name):
                continue
            here = {e[0] for e in parse_file(txt)}
            ids = here if ids is None else (ids & here)
    for ref in REFERENCES.values():
        if Path(ref["path"]).is_file():
            ids &= {e[0] for e in parse_file(ref["path"])}
    return ids


def draw_figure(spec, data, refs, ylim, xlim, stat_name, time_markers, dpi):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for name in spec["curves"]:
        configs = data.get(name)
        if not configs:
            continue
        st = STYLE[name]
        params = [p for p, _, _ in configs]
        ratios = [r for _, r, _ in configs]
        ys = [v for _, _, v in configs]
        # Uniform markers ride on the line itself, as in the paper's figures;
        # --time-markers needs per-point areas, so those go through scatter.
        sizes = marker_sizes_for_method(name, params) if time_markers else None
        ax.plot(ratios, ys, "-", color=st["color"], label=st["label"],
                marker=None if sizes else st["marker"], zorder=2)
        if sizes:
            ax.scatter(ratios, ys, s=sizes, marker=st["marker"],
                       color=st["color"], zorder=3)

    for name in spec["refs"]:
        pt = refs.get(name)
        if pt is None:
            continue
        cfg = REFERENCES[name]
        ax.scatter([pt[0]], [pt[1]], marker="*", s=220, color=cfg["color"],
                   label=cfg["label"], edgecolors="black", linewidths=0.5,
                   zorder=4)

    # Payload the size of the raw mesh: everything left of this actually compresses.
    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("compression ratio $\\downarrow$")
    ax.set_ylabel(f"Chamfer distance ({stat_name}) $\\downarrow$")
    ax.set_title(spec["title"])
    ax.grid(True, which="major")
    ax.grid(False, which="minor")
    legend_kw = ({"title": "dot area $\\propto$ cum. time", "title_fontsize": 9}
                 if time_markers else {})
    ax.legend(loc=spec.get("legend_loc", "best"), **legend_kw)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{spec['stem']}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stat", choices=["mean", "median"], default="mean",
                    help="statistic aggregated over meshes (default: mean)")
    ap.add_argument("--common-meshes", action="store_true",
                    help="restrict every curve to meshes all methods evaluated")
    ap.add_argument("--time-markers", action="store_true",
                    help="scale marker area by cumulative wall-clock time for the "
                         "ladder 1..k (default: uniform markers)")
    ap.add_argument("--free-axes", action="store_true",
                    help="let each figure autoscale instead of sharing limits")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    plt.rcParams.update(PAPER_RC)
    agg = np.mean if args.stat == "mean" else np.median
    keep_ids = common_mesh_ids() if args.common_meshes else None
    if keep_ids is not None:
        # Every config file of every method has to contain the mesh, so this
        # intersection is currently empty -- bail with that rather than an
        # empty-sequence error downstream.
        if not keep_ids:
            raise SystemExit(
                "--common-meshes: no mesh is present in every config of every method. "
                "The binding constraints are NGP (log2_2 / log2_3 / log2_5 keep only "
                "11-16 meshes, largely disjoint) and VecSet_HAE L1 / L2 (no finite "
                "Chamfer value). NGF alone still shares 44 meshes across its LODs."
            )
        print(f"Restricting to {len(keep_ids)} meshes common to all methods\n")

    data, refs = {}, {}
    for name, (subdir, pattern) in CURVES.items():
        configs = load_curve(subdir, pattern, keep_ids)
        data[name] = [(p, r, float(agg(cds))) for p, r, cds in configs]
        print(f"{name}:")
        for (p, r, cds) in configs:
            print(f"  k={p:<5} ratio={r:<10.4f} {args.stat}={agg(cds):.6f}  n={len(cds)}")
    for name, cfg in REFERENCES.items():
        pt = load_reference(cfg["path"], keep_ids)
        if pt is None:
            print(f"{name}: MISSING ({cfg['path']})")
            continue
        refs[name] = (pt[0], float(agg(pt[1])))
        print(f"{name}:\n  ratio={pt[0]:.4f} {args.stat}={agg(pt[1]):.6f}  n={len(pt[1])}")

    # Shared limits so the two panels are directly comparable. The top headroom
    # scales with the busiest legend so the box never lands on a curve.
    if args.free_axes:
        xlim = ylim = None
    else:
        xs = [r for c in data.values() for _, r, _ in c] + [p[0] for p in refs.values()]
        ys = [y for c in data.values() for _, _, y in c] + [p[1] for p in refs.values()]
        rows = max(len(s["curves"]) + len(s["refs"]) for s in FIGURES.values())
        xlim = (min(xs) / 2.0, max(xs) * 2.0)
        ylim = (min(ys) / 1.6, max(ys) * 10 ** (0.34 * rows))  # a third of a decade per row

    print()
    for spec in FIGURES.values():
        draw_figure(spec, data, refs, ylim, xlim, args.stat,
                    args.time_markers, args.dpi)


if __name__ == "__main__":
    main()
