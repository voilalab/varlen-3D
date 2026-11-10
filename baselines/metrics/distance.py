"""
Compute Chamfer distances for reconstruction sweeps and write distance files.

Chamfer: NUM_POINTS (10,000) surface points are sampled from the GT mesh
(model_normalized.obj) and from the reconstruction; each cloud is normalized to
the unit cube independently (centroid-subtract, divide by max |coord|); the
reconstruction is ICP-aligned to the GT (--no-icp skips this); then

    CD = mean_r min_g ||r - g||²  +  mean_g min_r ||g - r||²

computed by pytorch3d.loss.chamfer_distance (exact nearest neighbours, run on
GPU when available). GT samples are seeded per mesh, so every method is scored
against the same GT points.

VecSet / VecSet-HAE and MeshGPT (ours) are not computed here: their distance files
in distances/ are provided as-is.

Ratio per file = Σ(payload bits) / Σ(GT mesh bits) across the meshes with a
Chamfer value in the file.  Every line gets the same aggregate ratio.  What each
method is charged is defined in metrics/bit_accounting.py.
"""

import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.loss import chamfer_distance as _pytorch3d_chamfer_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bit_accounting import (  # noqa: E402
    bacon_bits, cons_bits, gt_mesh_bits, meshgpt_decimated_counts,
    meshgpt_orig_bits, ngf_bits, ngp_bits,
)

NUM_POINTS = 10_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MeshGPT decodes in its training frame, GT[:, [2, 0, 1]]; this maps it back.
MESHGPT_TO_GT_AXES = [1, 2, 0]


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------

def _load_mesh(path, axis_order=None):
    tm = trimesh.load(str(path), force="mesh", skip_materials=True)
    if len(tm.vertices) == 0 or len(tm.faces) == 0:
        raise ValueError(f"Empty mesh: {path}")
    if axis_order is not None:
        tm = trimesh.Trimesh(vertices=np.asarray(tm.vertices)[:, axis_order], faces=tm.faces, process=False)
    return tm


def _sample_normalized(mesh, num_points, seed):
    pts, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    pts = pts - pts.mean(axis=0)
    scale = np.abs(pts).max()
    return pts / scale if scale > 0 else pts


_gt_cache = {}


def _to_tensor(pts):
    return torch.from_numpy(pts).float().unsqueeze(0).to(DEVICE)


def _gt_cloud(gt_path, num_points):
    key = (str(gt_path), num_points)
    if key not in _gt_cache:
        seed = zlib.crc32(str(Path(gt_path).resolve()).encode())
        pts = _sample_normalized(_load_mesh(gt_path), num_points, seed)
        _gt_cache[key] = _to_tensor(pts)
    return _gt_cache[key]


USE_ICP = True  # set False by --no-icp


def _icp_align(source, target):
    """Rigidly align source (N,3) onto target (M,3); unaligned source if ICP fails."""
    try:
        _, aligned, _ = trimesh.registration.icp(source, target, max_iterations=50)
        return aligned
    except Exception as e:
        print(f"ICP failed: {e}")
        return source


def chamfer_distance(gt_path, recon_path, num_points=NUM_POINTS, axis_order=None):
    """Normalized Chamfer distance between two meshes, or None on failure.

    Both point clouds are independently centered and rescaled to the unit cube
    (see _sample_normalized), then the reconstruction is ICP-aligned to the GT
    (max 50 iterations) unless USE_ICP is False. This ICP step is what produced
    the stored distances (e.g. MeshGPT_orig1 reproduces exactly with it).
    Computed with pytorch3d.loss.chamfer_distance (exact nearest neighbours,
    mean squared distance, both directions summed -- matches the formula above).
    """
    try:
        gt_pts = _gt_cloud(gt_path, num_points)
        rc_mesh = _load_mesh(recon_path, axis_order)
        rc_np = _sample_normalized(rc_mesh, num_points, seed=0)
        if USE_ICP:
            rc_np = _icp_align(rc_np, gt_pts[0].cpu().numpy())
        rc_pts = _to_tensor(rc_np)
        loss, _ = _pytorch3d_chamfer_distance(rc_pts, gt_pts)
        return float(loss.item())
    except Exception as e:
        print(f"Error: {recon_path}: {e}")
        return None


def calculate_chamfer_distance_from_meshes_norm(mesh_path1, mesh_path2, num_points=NUM_POINTS, device=None):
    """Backwards-compatible entry point: mesh_path1 is the GT, mesh_path2 the reconstruction."""
    return chamfer_distance(mesh_path1, mesh_path2, num_points=num_points)


calculate_chamfer_distance_from_objs_norm = calculate_chamfer_distance_from_meshes_norm


# ---------------------------------------------------------------------------
# Per-method entry collection
# ---------------------------------------------------------------------------

def _gt_obj(data_dir, category, mesh_hash):
    return os.path.join(data_dir, category, mesh_hash, "models", "model_normalized.obj")


def _collect_ngp(input_dir, data_dir, category):
    """Collect NGP entries grouped by log2 value.

    Returns {log2: [(mesh_hash, gt_path, recon_path, payload_bits), ...]}.
    """
    by_config = {}
    for folder in sorted(os.listdir(input_dir)):
        m = re.match(r"log2_(\d+)$", folder)
        if not m:
            continue
        log2 = int(m.group(1))
        cat_dir = os.path.join(input_dir, folder, category)
        if not os.path.isdir(cat_dir):
            continue
        entries = []
        for mesh_hash in sorted(os.listdir(cat_dir)):
            mesh_dir = os.path.join(cat_dir, mesh_hash)
            if not os.path.isdir(mesh_dir):
                continue
            recon = os.path.join(mesh_dir, "reconstructed_mesh.obj")
            gt = _gt_obj(data_dir, category, mesh_hash)
            if not os.path.exists(recon) or not os.path.exists(gt):
                continue
            payload = ngp_bits(mesh_dir)
            if payload is None:
                continue
            entries.append((mesh_hash, gt, recon, payload))
        if entries:
            by_config[f"log2_{log2}"] = entries
    return by_config


def _collect_ngf(input_dir, data_dir, category, ngf_features, workspace):
    """Collect NGF entries grouped by LOD.

    Only includes meshes where the .pt checkpoint exists (training succeeded);
    the payload is counted from that checkpoint.

    Returns {lod_N: [(mesh_hash, gt_path, recon_path, payload_bits), ...]}.
    """
    pt_dir = Path(workspace) / "Projects" / "ngf" / "source" / "results" / "torched"

    stl_files = [f for f in os.listdir(input_dir) if f.endswith(".stl")]
    by_lod_raw = {}
    for f in stl_files:
        m = re.match(r"(.+)-lod(\d+)-f(\d+)\.stl$", f)
        if m:
            mesh_name, lod, feats = m.group(1), int(m.group(2)), int(m.group(3))
            if feats != ngf_features:
                continue
            by_lod_raw.setdefault(lod, []).append(
                (mesh_name, os.path.join(input_dir, f))
            )

    by_config = {}
    for lod in sorted(by_lod_raw.keys()):
        entries = []
        for mesh_name, recon_file in sorted(by_lod_raw[lod], key=lambda x: x[0]):
            gt = _gt_obj(data_dir, category, mesh_name)
            if not os.path.exists(gt):
                continue
            pt = pt_dir / f"{mesh_name}-lod{lod}-f{ngf_features}.pt"
            payload = ngf_bits(pt) if pt.is_file() else None
            if payload is None:
                continue
            entries.append((mesh_name, gt, recon_file, payload))
        if entries:
            by_config[f"lod_{lod}"] = entries
    return by_config


def _collect_cons(input_dir, data_dir, category):
    """Collect CONs entries grouped by c_dim.

    Returns {cdim_N: [(mesh_hash, gt_path, recon_path, payload_bits), ...]}.
    """
    by_config = {}
    for folder in sorted(os.listdir(input_dir)):
        m = re.match(r"cdim_(\d+)$", folder)
        if not m:
            continue
        cdim = int(m.group(1))
        payload = cons_bits(cdim)
        mesh_dir = os.path.join(input_dir, folder, "training", "generation", "meshes", category)
        if not os.path.isdir(mesh_dir):
            continue
        entries = []
        for item in sorted(os.listdir(mesh_dir)):
            mesh_hash = item.split(".")[0]
            gt = _gt_obj(data_dir, category, mesh_hash)
            if os.path.exists(gt):
                entries.append((mesh_hash, gt, os.path.join(mesh_dir, item), payload))
        if entries:
            by_config[f"cdim_{cdim}"] = entries
    return by_config


def _collect_meshgpt_orig(input_dir, data_dir, category, pickle_key):
    """Collect MeshGPT (original) reconstructions: <category>_<hash>.obj in the model frame.

    Returns {"distance": [(mesh_hash, gt_path, recon_path, payload_bits), ...]}.
    """
    decimated = meshgpt_decimated_counts(pickle_key)
    entries = []
    for fn in sorted(os.listdir(input_dir)):
        if not fn.endswith(".obj") or "_" not in fn:
            continue
        cat, mesh_hash = fn[:-len(".obj")].split("_", 1)
        gt = _gt_obj(data_dir, cat, mesh_hash)
        if cat != category or mesh_hash not in decimated or not os.path.exists(gt):
            continue
        num_vertices, num_faces = decimated[mesh_hash]
        entries.append((mesh_hash, gt, os.path.join(input_dir, fn), meshgpt_orig_bits(num_vertices, num_faces)))
    return {"distance": entries} if entries else {}


def _collect_bacon_flat(input_dir, data_dir, category):
    """Collect BACON-flat entries grouped by level.

    Returns {level_N: [(mesh_hash, gt_path, recon_path, payload_bits), ...]}.
    """
    params_files = [f for f in os.listdir(input_dir) if f.endswith("_params.txt")]
    by_level_raw = {}
    for pf in params_files:
        m = re.match(r"bacon_(.+?)_(\d+)_params\.txt$", pf)
        if not m:
            continue
        mesh_hash, layers = m.group(1), m.group(2)
        params_path = os.path.join(input_dir, pf)
        with open(params_path) as f:
            for line in f:
                lm = re.match(
                    r"level_(\d+)\s+layer_(\d+)\s+model_params=(\d+)\s+data_params=(\d+)\s+ratio=([\d.]+)",
                    line.strip(),
                )
                if not lm:
                    continue
                level = int(lm.group(1))
                obj_name = f"bacon_{mesh_hash}_{layers}_{level}.obj"
                obj_path = os.path.join(input_dir, obj_name)
                if os.path.exists(obj_path):
                    by_level_raw.setdefault(level, []).append((mesh_hash, obj_path))

    by_config = {}
    for level in sorted(by_level_raw.keys()):
        payload = bacon_bits(level)
        entries = []
        for mesh_hash, recon_file in sorted(by_level_raw[level], key=lambda x: x[0]):
            gt = _gt_obj(data_dir, category, mesh_hash)
            if os.path.exists(gt):
                entries.append((mesh_hash, gt, recon_file, payload))
        if entries:
            by_config[f"level_{level}"] = entries
    return by_config


# ---------------------------------------------------------------------------
# Aggregate ratio + Chamfer + write
# ---------------------------------------------------------------------------

def compute_and_write(entries, output_file, num_points, axis_order=None):
    """Run Chamfer, compute the aggregate ratio, and write one distance file.

    entries: list of (mesh_id, gt_path, recon_path, payload_bits)
    """
    results = []
    for mesh_id, gt_path, recon_path, _ in entries:
        ncd = None
        if os.path.exists(recon_path):
            ncd = chamfer_distance(gt_path, recon_path, num_points, axis_order)
        results.append((mesh_id, ncd))
        print(f"    {mesh_id}: " + (f"chamfer={ncd:.6e}" if ncd is not None else "FAILED"), flush=True)

    # Aggregate over the meshes that were scored (all of them if none were).
    scored = [e for e, (_, ncd) in zip(entries, results) if ncd is not None] or entries
    sum_payload = sum_mesh = 0
    gt_bits = {}
    for _, gt_path, _, payload in scored:
        if gt_path not in gt_bits:
            gt_bits[gt_path] = gt_mesh_bits(gt_path)
        if gt_bits[gt_path] and payload > 0:
            sum_payload += payload
            sum_mesh += gt_bits[gt_path]
    ratio = sum_payload / sum_mesh if sum_mesh > 0 else 0.0
    print(f"  {os.path.basename(output_file)}: ratio={ratio:.8f}  "
          f"({sum(r is not None for _, r in results)}/{len(entries)} scored)", flush=True)

    with open(output_file, "w") as f:
        for mesh_id, ncd in results:
            if ncd is None:
                f.write(f"[FAIL] Could not compute Chamfer Distance for {mesh_id}\n")
            else:
                f.write(f"Chamfer Distance for {mesh_id} with ratio {ratio:.8f}: {ncd:.6e}\n")
    return output_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # Reconstructions (Projects/) and GT meshes (data/shapenet) live under BENCH_WORKSPACE
    # (default: this repo);
    # GT paths must not move, since the GT sampling seed is a hash of the GT path.
    WORKSPACE = str(Path(os.environ.get("BENCH_WORKSPACE", Path(__file__).resolve().parents[1])))
    REPO = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Compute Chamfer distances with aggregate compression ratios"
    )
    parser.add_argument(
        "--project",
        choices=["bacon-flat", "cons", "ngp", "ngf", "meshgpt-orig"],
        required=True,
    )
    parser.add_argument("--input-dir", default=None, help="Override input directory")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(WORKSPACE, "data", "shapenet"),
        help="ShapeNet root",
    )
    parser.add_argument("--category", default="03001627")
    parser.add_argument(
        "--out-dir",
        default=str(REPO / "distances"),
    )
    parser.add_argument(
        "--ngf-features", type=int, default=40,
        help="NGF feature dimension to use (filters STL files by -f<N> suffix)",
    )
    parser.add_argument(
        "--out-name", default=None,
        help="Subdirectory under --out-dir (default: the project's usual name). Use this "
             "to score a second sweep of the same method into its own curve, "
             "e.g. --project meshgpt-orig --out-name MeshGPT_orig_v2",
    )
    parser.add_argument(
        "--meshgpt-pickle-key", default=None,
        help="Entry of metrics/meshgpt_decimated_counts.json matching the pickle the MeshGPT "
             "reconstructions were encoded from (default: meshgpt_767)",
    )
    parser.add_argument("--num-points", type=int, default=NUM_POINTS,
                        help="Surface points sampled per mesh for Chamfer")
    parser.add_argument("--no-icp", action="store_true",
                        help="Skip ICP alignment (diagnostic only: every committed distance file used ICP)")
    parser.add_argument("--configs", default=None,
                        help="Comma-separated config names to score (default: all)")
    parser.add_argument("--list-configs", action="store_true",
                        help="Print the config names found and exit")
    args = parser.parse_args()

    project = args.project
    category = args.category
    USE_ICP = not args.no_icp

    proj_roots = {
        "bacon-flat": os.path.join(WORKSPACE, "Projects", "bacon", "experiments", "outputs", "meshes"),
        "cons": os.path.join(WORKSPACE, "Projects", "convolutional_occupancy_networks", "parameter_sweep_results"),
        "ngp": os.path.join(WORKSPACE, "Projects", "instant-ngp", "parameter_sweep_results"),
        "ngf": os.path.join(WORKSPACE, "Projects", "ngf", "source", "results", "stl"),
        "meshgpt-orig": os.path.join(WORKSPACE, "Projects", "MeshGPT_old", "runs", "vocabulary", "mesh_val"),
    }
    out_map = {
        "bacon-flat": "bacon",
        "cons": "CONs",
        "ngp": "ngp",
        "ngf": "ngf",
        "meshgpt-orig": "MeshGPT_orig1",
    }
    # Distance-file names the graph scripts expect.
    file_name = {"meshgpt-orig": "{}.txt"}
    axis_order = {"meshgpt-orig": MESHGPT_TO_GT_AXES}.get(project)
    pickle_key = args.meshgpt_pickle_key or "meshgpt_767"

    input_dir = args.input_dir or proj_roots[project]
    out_path = os.path.join(args.out_dir, args.out_name or out_map[project])

    if project == "ngp":
        by_config = _collect_ngp(input_dir, args.data_dir, category)
    elif project == "ngf":
        by_config = _collect_ngf(input_dir, args.data_dir, category,
                                 args.ngf_features, WORKSPACE)
    elif project == "cons":
        by_config = _collect_cons(input_dir, args.data_dir, category)
    elif project == "meshgpt-orig":
        by_config = _collect_meshgpt_orig(input_dir, args.data_dir, category, pickle_key)
    elif project == "bacon-flat":
        by_config = _collect_bacon_flat(input_dir, args.data_dir, category)
    else:
        by_config = {}

    if not by_config:
        sys.exit(f"No entries found for {project} in {input_dir}")
    if args.list_configs:
        print("\n".join(sorted(by_config)))
        sys.exit(0)
    wanted = sorted(by_config) if args.configs is None else args.configs.split(",")

    os.makedirs(out_path, exist_ok=True)
    print(f"Project: {project}")
    print(f"Input:   {input_dir}")
    print(f"Output:  {out_path}")
    print(f"Points:  {args.num_points:,}")

    for config_name in wanted:
        print(f"\n  == {config_name} ({len(by_config[config_name])} meshes)", flush=True)
        compute_and_write(
            by_config[config_name],
            os.path.join(out_path, file_name.get(project, "{}_distance.txt").format(config_name)),
            args.num_points, axis_order,
        )

    print(f"\nDone. Results in {out_path}")
