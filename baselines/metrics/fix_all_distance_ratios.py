#!/usr/bin/env python3
"""
Rewrite the ratio on every distances/<method>/ file this package computes, in bits.
Provided curves (VecSet, MeshGPT ours; see PROVIDED) are left as-is.

Ratio per file = Σ(payload bits) / Σ(GT mesh bits) over the meshes with a finite
Chamfer value in that file (all meshes if none are finite). Every line gets the
same aggregate ratio; Chamfer values are left untouched. What each method is
charged is defined in metrics/bit_accounting.py.

MeshGPT's decimated vertex/face counts come from metrics/meshgpt_decimated_counts.json;
generate it once in the meshgpt env first.

Usage:
    python metrics/meshgpt_decimated_counts.py              # meshgpt env, once
    python metrics/fix_all_distance_ratios.py --dry-run     # preview
    python metrics/fix_all_distance_ratios.py               # rewrite
    python metrics/fix_all_distance_ratios.py --charge-connectivity  # reference only, not this benchmark's convention
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bit_accounting import (  # noqa: E402
    bacon_bits, cons_bits, gt_mesh_bits, meshgpt_decimated_counts,
    meshgpt_orig_bits, ngf_bits, ngp_bits,
)

WORKSPACE = Path(os.environ.get("BENCH_WORKSPACE", Path(__file__).resolve().parents[1]))
SHAPENET = WORKSPACE / "data" / "shapenet"
DIST_DIR = Path(__file__).resolve().parent.parent / "distances"
CATEGORY = "03001627"

# Our method's curves (from ../SDF, ../meshgpt) with ratios already stamped; never rewritten here.
PROVIDED = {"VecSet_HAE", "VecSet_GT", "MeshGPT_HAE"}

LINE_RE = re.compile(r"^Chamfer Distance for (.+?) with ratio ([\d.eE+-]+): (\S+)$")

_gt_cache: dict[str, int | None] = {}


def gt_bits(mesh_hash: str) -> int | None:
    if mesh_hash not in _gt_cache:
        _gt_cache[mesh_hash] = gt_mesh_bits(SHAPENET / CATEGORY / mesh_hash / "models" / "model_normalized.obj")
    return _gt_cache[mesh_hash]


def build_methods(connectivity: bool):
    """[(distance subdir, filename regex, payload_fn(param, mesh_hash) -> bits | None)].

    `param` is the regex's first group as an int, or None when it has no group.
    """
    ngp_root = WORKSPACE / "Projects" / "instant-ngp" / "parameter_sweep_results"
    ngf_pt_dir = WORKSPACE / "Projects" / "ngf" / "source" / "results" / "torched"

    def ngp(log2, h):
        return ngp_bits(ngp_root / f"log2_{log2}" / CATEGORY / h)

    def ngf(lod, h):
        pt = ngf_pt_dir / f"{h}-lod{lod}-f40.pt"
        return ngf_bits(pt, connectivity) if pt.is_file() else None

    def meshgpt_orig(pickle_key):
        def fn(_, h):
            vf = meshgpt_decimated_counts(pickle_key).get(h)
            return None if vf is None else meshgpt_orig_bits(vf[0], vf[1], connectivity)
        return fn

    return [
        ("ngp", r"log2_(\d+)_distance\.txt$", ngp),
        ("ngf", r"lod_(\d+)_distance\.txt$", ngf),
        ("CONs", r"cdim_(\d+)_distance\.txt$", lambda c_dim, _: cons_bits(c_dim)),
        ("bacon", r"level_(\d+)_distance\.txt$", lambda level, _: bacon_bits(level)),
        ("MeshGPT_orig1", r"distance\.txt$", meshgpt_orig("meshgpt_767")),
    ]


def rewrite_file(path: Path, payload_fn, param, dry_run: bool):
    """Recompute and stamp the aggregate bit ratio. Returns a summary dict or None."""
    lines = path.read_text().splitlines(keepends=True)
    parsed = []  # (line index, mesh id as written, mesh hash, old ratio, cd text, finite)
    for i, line in enumerate(lines):
        m = LINE_RE.match(line.rstrip("\n"))
        if not m:
            continue
        mesh_id, old_ratio, cd_text = m.groups()
        try:
            finite = math.isfinite(float(cd_text))
        except ValueError:
            finite = False
        parsed.append((i, mesh_id, re.sub(r"\.\w+$", "", mesh_id), float(old_ratio), cd_text, finite))
    if not parsed:
        return None

    scored = [p for p in parsed if p[5]] or parsed
    sum_payload = sum_mesh = 0
    skipped = []
    for _, _, mesh_hash, _, _, _ in scored:
        payload, gt = payload_fn(param, mesh_hash), gt_bits(mesh_hash)
        if payload is None or gt is None:
            skipped.append(mesh_hash)
            continue
        sum_payload += payload
        sum_mesh += gt
    if sum_mesh == 0:
        return None

    ratio = sum_payload / sum_mesh
    for i, mesh_id, _, _, cd_text, _ in parsed:
        lines[i] = f"Chamfer Distance for {mesh_id} with ratio {ratio:.8f}: {cd_text}\n"
    if not dry_run:
        path.write_text("".join(lines))

    n = len(scored) - len(skipped)
    return {"old": parsed[0][3], "new": ratio, "n": n, "skipped": skipped,
            "payload_per_mesh": sum_payload / n, "gt_per_mesh": sum_mesh / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    ap.add_argument("--charge-connectivity", action="store_true",
                    help="Also charge structural connectivity (MeshGPT face lists, NGF quad "
                         "complexes) that the decoder needs as input but this benchmark does not "
                         "count as a model parameter. For the MeshGPTs this also switches their VQ "
                         "payload to the deduplicated per-vertex index layout, which the face list "
                         "makes more expensive, not less. Reference only, not the default convention.")
    args = ap.parse_args()

    methods = build_methods(connectivity=args.charge_connectivity)
    covered = {subdir for subdir, _, _ in methods}
    for d in sorted(p for p in DIST_DIR.iterdir() if p.is_dir()):
        if d.name in PROVIDED:
            print(f"{d.name}/: provided, left as-is")
        elif d.name not in covered and any(d.glob("*.txt")):
            print(f"WARN: {d.name}/ has .txt files but no bit accounting; left untouched")

    pfx = "[dry-run] " if args.dry_run else ""
    for subdir, pattern, payload_fn in methods:
        method_dir = DIST_DIR / subdir
        if not method_dir.is_dir():
            continue
        print(f"\n=== {subdir} ===")
        pat = re.compile(pattern)
        files = []
        for txt in method_dir.glob("*.txt"):
            m = pat.match(txt.name)
            if m:
                files.append((int(m.group(1)) if m.groups() else -1, txt))
        for param, txt in sorted(files):
            res = rewrite_file(txt, payload_fn, param if param >= 0 else None, args.dry_run)
            if res is None:
                print(f"  {txt.name}: no scorable lines, left untouched")
                continue
            print(f"  {pfx}{txt.name:<26} old={res['old']:<12.6f} new={res['new']:.8f}  n={res['n']:<3} "
                  f"payload/mesh={res['payload_per_mesh']:>13,.0f} b  gt/mesh={res['gt_per_mesh']:>11,.0f} b")
            if res["skipped"]:
                print(f"    skipped {len(res['skipped'])} (no payload or GT): {res['skipped'][:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
