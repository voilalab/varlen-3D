#!/usr/bin/env python3
"""
Build processed_data.pkl for MeshGPT from raw ShapeNet meshes.

Validation split is fixed to the project validation list (not CONs/ngp test lists).

By default meshes are quadric-decimated to ``--max-faces`` (767) to match typical
vocabulary training. For **original** (non-decimated) ``model_normalized.obj`` meshes,
use ``--max-faces 0``. You must then **train a new checkpoint** on that pickle: the
shape2vecset encoder positional encoding size follows the max face count in the
training set, so a model trained on decimated data cannot be applied to much denser
meshes without retraining (see ``reconstruct_vocabulary_keep_k.py`` /
``max_num_faces_override``).
"""

import argparse
import pickle
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm

DEFAULT_SHAPENET_DIR = Path(__file__).resolve().parents[2] / "data" / "shapenet"  # <repo>/data/shapenet
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "shapenet" / "processed_data.pkl"
DEFAULT_MAX_FACES = 767


def load_mesh_ids(lst_path):
    with open(lst_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_mesh(mesh_id, shapenet_dir: Path, max_faces: int, skip_if_over: int | None):
    obj_path = shapenet_dir / mesh_id / "models" / "model_normalized.obj"
    if not obj_path.exists():
        return None, None, None
    try:
        loaded = trimesh.load(str(obj_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(loaded.dump())
        else:
            mesh = loaded
        n_face = len(mesh.faces)
        if skip_if_over is not None and n_face > skip_if_over:
            return None, None, None
        if max_faces > 0 and n_face > max_faces:
            # open3d quadric decimation, called directly. This is what trimesh 3.x did inside
            # simplify_quadric_decimation; trimesh 4.x switched to fast_simplification, which
            # keeps only 77/100 val meshes under the cap and does not reproduce our pickle.
            import open3d as o3d
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                          o3d.utility.Vector3iVector(mesh.faces))
            m = m.simplify_quadric_decimation(target_number_of_triangles=max_faces)
            mesh = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles))
            if len(mesh.faces) > max_faces:
                return None, None, None
        name = mesh_id.replace("/", "_")
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)
        return name, vertices, faces
    except Exception as e:
        print(f"  Error loading {mesh_id}: {e}")
        return None, None, None


def process_split(mesh_ids, split_name, shapenet_dir: Path, max_faces: int, skip_if_over: int | None):
    names, vertices, faces = [], [], []
    skipped = 0
    for mesh_id in tqdm(mesh_ids, desc=split_name):
        n, v, f = load_mesh(mesh_id, shapenet_dir, max_faces, skip_if_over)
        if n is not None:
            names.append(n)
            vertices.append(v)
            faces.append(f)
        else:
            skipped += 1
    cap_msg = f"decimate≤{max_faces}" if max_faces > 0 else "no decimation"
    skip_msg = f", skip_if_over={skip_if_over}" if skip_if_over else ""
    print(f"  {split_name}: {len(names)}/{len(mesh_ids)} loaded ({skipped} skipped; {cap_msg}{skip_msg})")
    return names, vertices, faces


def main():
    ap = argparse.ArgumentParser(description="Build MeshGPT processed_data.pkl from ShapeNet OBJs")
    ap.add_argument(
        "--shapenet-dir",
        type=Path,
        default=DEFAULT_SHAPENET_DIR,
        help="Root with synset/hash/models/model_normalized.obj layout",
    )
    ap.add_argument(
        "--train-lst",
        type=Path,
        default=None,
        help="Default: <shapenet-dir>/train.lst",
    )
    ap.add_argument(
        "--val-lst",
        type=Path,
        default=None,
        help="Default: <shapenet-dir>/val.lst",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .pkl path",
    )
    ap.add_argument(
        "--max-faces",
        type=int,
        default=DEFAULT_MAX_FACES,
        help="Quadric decimation target face count; use 0 for no decimation (full meshes)",
    )
    ap.add_argument(
        "--skip-if-over",
        type=int,
        default=None,
        metavar="N",
        help="Skip meshes with more than N faces (useful with --max-faces 0 to drop huge shapes)",
    )
    args = ap.parse_args()

    shapenet_dir = args.shapenet_dir.resolve()
    train_lst = args.train_lst or (shapenet_dir / "train.lst")
    val_lst = args.val_lst or (shapenet_dir / "val.lst")

    if not train_lst.is_file():
        raise FileNotFoundError(f"Missing train list: {train_lst}")
    if not val_lst.is_file():
        raise FileNotFoundError(f"Missing validation list: {val_lst}")

    train_ids = load_mesh_ids(train_lst)
    val_ids = load_mesh_ids(val_lst)
    print(f"ShapeNet dir: {shapenet_dir}")
    print(f"Train list: {train_lst} ({len(train_ids)} ids)")
    print(f"Validation list: {val_lst} ({len(val_ids)} ids)")
    print(f"max_faces: {args.max_faces} (0 = no decimation)\n")

    name_train, verts_train, faces_train = process_split(
        train_ids, "train", shapenet_dir, args.max_faces, args.skip_if_over
    )
    name_val, verts_val, faces_val = process_split(
        val_ids, "val", shapenet_dir, args.max_faces, args.skip_if_over
    )

    data = {
        "name_train": name_train,
        "vertices_train": verts_train,
        "faces_train": faces_train,
        "name_val": name_val,
        "vertices_val": verts_val,
        "faces_val": faces_val,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved: {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
