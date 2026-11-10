#!/usr/bin/env python3
"""
Helper utilities for instant-ngp parameter sweeps.

Features:
- Get MLP params via pyngp.Testbed (mlp_params = n_params - n_encoding_params)
- Count mesh baseline params = 3*num_vertices + num_faces
- Solve for HashGrid log2_hashmap_size (T) using:
    mlp_params + L * 2^T * F = mesh_params * ratio
- Compute normalized chamfer distance between meshes without scipy (chunked O(N^2))
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def count_mesh_params(mesh_path: str) -> int:
    # Dependency-light: count vertices and faces from OBJ/STL text without trimesh.
    # Faces are counted as triangles (fan triangulation for polygons), to match common loaders.
    path = str(mesh_path)
    if path.lower().endswith(".obj"):
        v, f = _count_obj_v_f(path)
        return 3 * v + f
    raise ValueError(f"Unsupported mesh format for param counting (expected .obj): {mesh_path}")


def get_hashgrid_LF(config: dict) -> Tuple[int, int]:
    enc = config.get("encoding", {})
    if enc.get("otype") != "HashGrid":
        raise ValueError(f'Expected encoding.otype="HashGrid", got {enc.get("otype")}')
    L = int(enc["n_levels"])
    F = int(enc["n_features_per_level"])
    return L, F


def solve_log2_hashmap_size(
    *,
    mlp_params: int,
    mesh_params: int,
    ratio: float,
    n_levels: int,
    n_features_per_level: int,
    min_log2: int = 10,
    max_log2: int = 24,
    scale_encoding: bool = False,  # Changed default to False for true compression ratio
) -> Tuple[int, float, float]:
    """Return (log2_hashmap_size, target_params, actual_ratio).
    
    Target: model_params = mesh_params * ratio (true compression ratio)
    
    Solves for log2_hashmap_size such that:
        mlp_params + L * 2^T * F = mesh_params * ratio
    
    Returns actual_ratio since we may not hit target exactly due to discrete T.
    """
    target = float(mesh_params) * float(ratio)
    denom = float(n_levels) * float(n_features_per_level)
    if denom <= 0:
        raise ValueError("Invalid HashGrid L/F; denom must be > 0")

    # Solve: target - mlp_params = L * 2^T * F
    # => 2^T = (target - mlp_params) / (L * F)
    rhs = (target - float(mlp_params)) / denom
    
    # Compute minimum possible model params (at min_log2)
    min_encoding = n_levels * (2 ** min_log2) * n_features_per_level
    min_total = mlp_params + min_encoding
    min_ratio = min_total / mesh_params
    
    if rhs <= 1.0:
        # Target is smaller than minimum possible - clamp to min_log2
        actual_params = min_total
        actual_ratio = min_ratio
        return int(min_log2), target, actual_ratio

    T = math.log(rhs, 2.0)
    T_int = int(round(T))
    T_int = max(min_log2, T_int)
    
    # Compute actual params and ratio
    actual_encoding = n_levels * (2 ** T_int) * n_features_per_level
    actual_params = mlp_params + actual_encoding
    actual_ratio = actual_params / mesh_params
    
    return T_int, target, actual_ratio


def write_updated_config(
    *,
    base_config_path: str,
    out_config_path: str,
    log2_hashmap_size: int,
) -> None:
    cfg = _load_json(base_config_path)
    if "encoding" not in cfg:
        raise ValueError(f"Config missing encoding: {base_config_path}")
    cfg["encoding"]["log2_hashmap_size"] = int(log2_hashmap_size)
    

    out_dir = os.path.dirname(out_config_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def get_mlp_params_via_testbed(scene_mesh_path: str, network_config_path: str) -> Tuple[int, int, int]:
    # Import lazily so this helper can be used without pyngp for non-MLP tasks.
    build_dir = Path(__file__).resolve().parent / "build"
    if build_dir.exists():
        sys.path.insert(0, str(build_dir))
    import pyngp as ngp  # noqa

    testbed = ngp.Testbed()
    # Match scripts/run.py behavior: set root_dir so resources/paths resolve consistently.
    testbed.root_dir = str(Path(__file__).resolve().parent)
    # Ensure correct mode/dims by loading an SDF mesh.
    testbed.load_training_data(scene_mesh_path)
    testbed.reload_network_from_file(network_config_path)

    total = int(testbed.n_params())
    enc = int(testbed.n_encoding_params())
    mlp = int(total - enc)
    return mlp, total, enc


def _count_obj_v_f(obj_path: str) -> Tuple[int, int]:
    n_v = 0
    n_f = 0
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                n_v += 1
            elif line.startswith("f "):
                n_f += 1
    return n_v, n_f


def cmd_mlp_params(args) -> int:
    mlp, total, enc = get_mlp_params_via_testbed(args.scene, args.network)
    if args.mlp_only:
        print(mlp)
    else:
        print(json.dumps({"mlp_params": mlp, "total_params": total, "encoding_params": enc}))
    return 0


def cmd_mesh_params(args) -> int:
    print(count_mesh_params(args.mesh))
    return 0


def cmd_prepare_config(args) -> int:
    cfg = _load_json(args.base_config)
    L, F = get_hashgrid_LF(cfg)
    mesh_params = count_mesh_params(args.mesh)
    log2_size, target, actual_ratio = solve_log2_hashmap_size(
        mlp_params=args.mlp_params,
        mesh_params=mesh_params,
        ratio=args.ratio,
        n_levels=L,
        n_features_per_level=F,
        min_log2=args.min_log2,
        max_log2=args.max_log2,
    )
    write_updated_config(
        base_config_path=args.base_config,
        out_config_path=args.out_config,
        log2_hashmap_size=log2_size,
    )
    
    # Compute actual params
    encoding_params = L * (2 ** log2_size) * F
    actual_model_params = args.mlp_params + encoding_params
    
    # Compute minimum possible ratio for this mesh
    min_encoding = L * (2 ** args.min_log2) * F
    min_model_params = args.mlp_params + min_encoding
    min_ratio = min_model_params / mesh_params

    # Bash-friendly key=value output
    print(f"MESH_PARAMS={mesh_params}")
    print(f"TARGET_PARAMS={int(target)}")
    print(f"ACTUAL_MODEL_PARAMS={actual_model_params}")
    print(f"REQUESTED_RATIO={args.ratio}")
    print(f"ACTUAL_RATIO={actual_ratio:.4f}")
    print(f"MIN_POSSIBLE_RATIO={min_ratio:.4f}")
    print(f"ENCODING_PARAMS={encoding_params}")
    print(f"MLP_PARAMS={args.mlp_params}")
    print(f"N_LEVELS={L}")
    print(f"N_FEATURES_PER_LEVEL={F}")
    print(f"LOG2_HASHMAP_SIZE={log2_size}")
    return 0


def cmd_prepare_config_direct(args) -> int:
    """Create config with a specific log2_hashmap_size and compute the resulting ratio."""
    cfg = _load_json(args.base_config)
    L, F = get_hashgrid_LF(cfg)
    mesh_params = count_mesh_params(args.mesh)
    log2_size = args.log2_hashmap_size
    
    write_updated_config(
        base_config_path=args.base_config,
        out_config_path=args.out_config,
        log2_hashmap_size=log2_size,
    )
    
    # Compute actual params and ratio
    encoding_params = L * (2 ** log2_size) * F
    model_params = args.mlp_params + encoding_params
    ratio = model_params / mesh_params

    # Bash-friendly key=value output
    print(f"MESH_PARAMS={mesh_params}")
    print(f"MODEL_PARAMS={model_params}")
    print(f"ENCODING_PARAMS={encoding_params}")
    print(f"MLP_PARAMS={args.mlp_params}")
    print(f"N_LEVELS={L}")
    print(f"N_FEATURES_PER_LEVEL={F}")
    print(f"LOG2_HASHMAP_SIZE={log2_size}")
    print(f"RATIO={ratio:.6f}")
    return 0


def cmd_chamfer(args) -> int:
    chamfer, norm = normalized_chamfer(args.mesh_a, args.mesh_b, n_samples=args.n_samples)
    if args.json:
        print(json.dumps({"chamfer": chamfer, "normalized_chamfer": norm, "n_samples": args.n_samples}))
    else:
        print(norm)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_mlp = sub.add_parser("mlp_params", help="Get MLP param count via Testbed")
    p_mlp.add_argument("--scene", required=True, help="Path to an SDF training mesh (.obj/.stl)")
    p_mlp.add_argument("--network", required=True, help="Path to network json")
    p_mlp.add_argument("--mlp-only", action="store_true", help="Print only the integer MLP param count")
    p_mlp.set_defaults(func=cmd_mlp_params)

    p_mesh = sub.add_parser("mesh_params", help="Compute baseline mesh param count = 3*V + F")
    p_mesh.add_argument("--mesh", required=True)
    p_mesh.set_defaults(func=cmd_mesh_params)

    p_cfg = sub.add_parser("prepare_config", help="Solve log2_hashmap_size from ratio and write updated config")
    p_cfg.add_argument("--mesh", required=True)
    p_cfg.add_argument("--ratio", type=float, required=True, help="Target compression ratio: model_params/mesh_params")
    p_cfg.add_argument("--mlp-params", type=int, required=True, dest="mlp_params")
    p_cfg.add_argument("--base-config", required=True, dest="base_config")
    p_cfg.add_argument("--out-config", required=True, dest="out_config")
    p_cfg.add_argument("--min-log2", type=int, default=10, help="Minimum log2_hashmap_size (default: 10)")
    p_cfg.add_argument("--max-log2", type=int, default=19, help="Maximum log2_hashmap_size (default: 19)")
    p_cfg.set_defaults(func=cmd_prepare_config)

    p_cfg_direct = sub.add_parser("prepare_config_direct", help="Write config with specific log2_hashmap_size and compute ratio")
    p_cfg_direct.add_argument("--mesh", required=True)
    p_cfg_direct.add_argument("--log2-hashmap-size", type=int, required=True, dest="log2_hashmap_size",
                              help="log2_hashmap_size value to use")
    p_cfg_direct.add_argument("--mlp-params", type=int, required=True, dest="mlp_params")
    p_cfg_direct.add_argument("--base-config", required=True, dest="base_config")
    p_cfg_direct.add_argument("--out-config", required=True, dest="out_config")
    p_cfg_direct.set_defaults(func=cmd_prepare_config_direct)

    p_ch = sub.add_parser("chamfer", help="Compute chamfer and normalized chamfer between two meshes")
    p_ch.add_argument("--mesh-a", required=True, dest="mesh_a")
    p_ch.add_argument("--mesh-b", required=True, dest="mesh_b")
    p_ch.add_argument("--n-samples", type=int, default=5000)
    p_ch.add_argument("--json", action="store_true", help="Print JSON")
    p_ch.set_defaults(func=cmd_chamfer)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

