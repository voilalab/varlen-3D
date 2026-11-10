#!/usr/bin/env python3
"""
Marker sizing for the RD graphs (graph_mean.py, graph_median.py and
graph_paper_figures.py --time-markers).

A point at config k is drawn with marker AREA proportional to the cumulative
wall-clock seconds to evaluate the ladder 1..k, i.e. the sum of per-config mean
inference times for all k' <= k (k' taken in the order the caller passes
`params`, which is ratio-sorted). Areas are normalized per method, from
MIN_AREA at the first config to MAX_AREA at that method's last one, so marker
sizes show growth along one curve and are not comparable across curves.

Public API used by the graph scripts:
    marker_sizes_for_method(method_name, params) -> list[float] | None
"""

import csv
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

DIST = Path(__file__).resolve().parent.parent / "distances"
PROJ = Path(os.environ.get("BENCH_WORKSPACE", Path(__file__).resolve().parents[1])) / "Projects"

# Marker-area range (matplotlib scatter `s`, in points^2).
MIN_AREA = 15.0
MAX_AREA = 100.0

# method name -> (csv path, param column). BACON is handled specially.
TIMING_CSV = {
    "NGP":        (PROJ / "instant-ngp/parameter_sweep_results/inference_times.csv", "log2_hashmap_size"),
    "NGF":        (PROJ / "ngf/parameter_sweep_results/inference_times.csv", "lod"),
    "CONs":       (PROJ / "convolutional_occupancy_networks/parameter_sweep_results/inference_times.csv", "cdim"),
    # Provided with the VecSet_HAE distance files.
    "VecSet_HAE": (DIST / "VecSet_HAE/inference_times.csv", "L"),
}
BACON_CSV = PROJ / "bacon/logs/inference_times.csv"

# MeshGPT-family curves: measured per-keep_k decode times from
# distances/<method>/inference_times.csv (MeshGPT_HAE ships one). The fallback
# below (scale one full-generation time by keep_k) applies only to a curve named
# exactly "MeshGPT", which this package does not have.
MESHGPT_CSV = PROJ / "MeshGPT_old/pretrained/inference_times.csv"


@lru_cache(maxsize=None)
def _meshgpt_measured_times(method_name="MeshGPT"):
    """{keep_k:int -> mean measured decode time} or None if not yet recorded.

    Each MeshGPT-family curve keeps its CSV beside its own distance files, so the
    lookup is keyed by the method/directory name (MeshGPT, MeshGPT_v2, ...).
    """
    csv_path = DIST / method_name / "inference_times.csv"
    if not csv_path.is_file():
        return None
    acc = defaultdict(list)
    for row in _csv_rows(csv_path):
        try:
            acc[int(row["keep_k"])].append(float(row["inference_time_s"]))
        except (ValueError, KeyError):
            continue
    return {k: float(np.mean(v)) for k, v in acc.items()} or None


@lru_cache(maxsize=None)
def _meshgpt_full_time():
    if not MESHGPT_CSV.is_file():
        return None
    t = [float(r["inference_time_s"]) for r in _csv_rows(MESHGPT_CSV)]
    return float(np.mean(t)) if t else None


def _csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=None)
def load_mean_step_times(method_name):
    """{param:int -> mean per-config inference time (s)} or None if unavailable."""
    if method_name == "BACON":
        if not BACON_CSV.is_file():
            return None
        totals = [float(r["inference_time_s"]) for r in _csv_rows(BACON_CSV)]
        if not totals:
            return None
        # One run emits all 4 bands; split the total evenly across levels 1..4.
        per = float(np.mean(totals)) / 4.0
        return {1: per, 2: per, 3: per, 4: per}

    spec = TIMING_CSV.get(method_name)
    if spec is None:
        return None
    path, col = spec
    if not path.is_file():
        return None
    acc = defaultdict(list)
    for row in _csv_rows(path):
        try:
            acc[int(row[col])].append(float(row["inference_time_s"]))
        except (ValueError, KeyError):
            continue
    if not acc:
        return None
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cumulative_wall_clock_seconds(method_name, params):
    """Cumulative seconds at each config, accumulating in the given param order."""
    if method_name.startswith("MeshGPT"):
        measured = _meshgpt_measured_times(method_name)
        if measured:
            cum, out = 0.0, []
            for p in params:
                cum += measured.get(int(p), 0.0)
                out.append(cum)
            return out
        # The scale-the-full-generation-time model only ever applied to the original
        # MeshGPT sweep. Other MeshGPT-family curves either have their own measured
        # CSV (above) or get no timing at all -- notably MeshGPT_orig, a single point
        # at k=0, where kmax would be 0.
        if method_name != "MeshGPT":
            return None
        full = _meshgpt_full_time()
        if full is None or not params:
            return None
        kmax = max(int(p) for p in params)
        if kmax <= 0:
            return None
        cum, out = 0.0, []
        for p in params:
            cum += full * int(p) / kmax
            out.append(cum)
        return out

    st = load_mean_step_times(method_name)
    if st is None:
        return None
    cum, out = 0.0, []
    for p in params:
        cum += st.get(int(p), 0.0)
        out.append(cum)
    return out


def marker_sizes_for_method(method_name, params):
    """Marker areas ∝ cumulative wall-clock for the ladder 1..k, or None.

    Normalized per line (against that method's own total cumulative time) so the
    dots on every curve visibly grow from MIN_AREA at k=1 to MAX_AREA at the
    final config, proportional to the cumulative time 1..k.
    """
    cum = cumulative_wall_clock_seconds(method_name, params)
    if cum is None:
        return None
    total = cum[-1] if cum else 0.0
    if total <= 0:
        return [MIN_AREA] * len(cum)
    return [MIN_AREA + (MAX_AREA - MIN_AREA) * (c / total) for c in cum]


if __name__ == "__main__":
    for name in list(TIMING_CSV) + ["BACON", "MeshGPT_HAE"]:
        st = (_meshgpt_measured_times(name) if name.startswith("MeshGPT")
              else load_mean_step_times(name))
        if st is None:
            print(f"{name}: no timing data (markers fall back to constant size)")
            continue
        params = sorted(st)
        cum = cumulative_wall_clock_seconds(name, params)
        print(f"{name}: total={sum(st.values()):.2f}s  "
              + "  ".join(f"k={p}:{c:.2f}s" for p, c in zip(params, cum)))
