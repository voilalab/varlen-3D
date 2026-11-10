import os
from pathlib import Path
import numpy as np
import trimesh

def convert_obj_to_xyz(obj_path, out_path, num_samples=100000, seed=0):
    print(f"Loading mesh: {obj_path}")
    mesh = trimesh.load(obj_path, force='mesh', skip_materials=True, process=False)

    if isinstance(mesh, trimesh.Scene):
        # If OBJ loads as a Scene, merge geometry
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded file is not a Trimesh.")

    if mesh.is_empty or mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError("Mesh is empty or invalid.")

    # Basic cleanup
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.merge_vertices()

    # Merge all components (don't keep only largest - chairs/tables need legs, seat, back)
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = trimesh.util.concatenate(comps)
    elif len(comps) == 0:
        raise ValueError("Mesh split returned no components")

    # Ensure normals exist
    mesh.rezero()
    mesh.vertices -= mesh.centroid
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 0:
        mesh.vertices /= scale

    np.random.seed(seed)

    print(f"Sampling {num_samples} surface points...")
    points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)

    # Face normals (fast baseline)
    normals = mesh.face_normals[face_indices]

    # Normalize normals defensively
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(nrm, 1e-12, None)

    print(f"Saving xyz+normals to {out_path}")
    np.savetxt(out_path, np.hstack([points, normals]), fmt="%.8f")
    print("Done.")
# Example usage:
# Run from repo root: python utils/obj_to_xyz.py

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Convert OBJ to XYZ with normals')
    parser.add_argument('--data-dir', default=str(Path(__file__).resolve().parents[1] / 'data' / 'shapenet'),
                        help='Data directory containing category folders')
    parser.add_argument('--val-split', default=None,
                        help='Path to val.lst (if set, only process validation meshes)')
    parser.add_argument('--num-samples', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    val_split = Path(args.val_split) if args.val_split is not None else data_dir / 'val.lst'

    if val_split.exists():
        with open(val_split) as f:
            mesh_ids = [line.strip() for line in f if line.strip()]
        print(f"Processing {len(mesh_ids)} meshes from {val_split}")
    else:
        # Fallback: scan all category/mesh_id/models/*.obj
        mesh_ids = []
        for cat_dir in data_dir.iterdir():
            if cat_dir.is_dir():
                for mesh_dir in cat_dir.iterdir():
                    if mesh_dir.is_dir():
                        mesh_ids.append(f"{cat_dir.name}/{mesh_dir.name}")
        print(f"No val.lst found, processing {len(mesh_ids)} meshes from {data_dir}")

    for mesh_id in mesh_ids:
        obj_path = data_dir / mesh_id / 'models' / 'model_normalized.obj'
        if not obj_path.exists():
            print(f"Skipping {mesh_id}: OBJ not found")
            continue
        xyz_path = obj_path.with_suffix('.xyz')
        try:
            convert_obj_to_xyz(str(obj_path), str(xyz_path), args.num_samples, args.seed)
        except Exception as e:
            print(f"Failed {mesh_id}: {e}")

