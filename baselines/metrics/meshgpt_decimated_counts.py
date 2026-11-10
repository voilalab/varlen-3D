#!/usr/bin/env python3
"""
Per-mesh decimated vertex/face counts for the MeshGPT processed_data pickles.

Both MeshGPT variants are charged per-face VQ codes (84 bits/face), so the ratio
scales with F_dec; V_dec is only needed for the deduplicated index layout that
metrics/bit_accounting.py computes under --charge-connectivity. Counts
are taken *after* the dataset's quantize-and-merge step (`create_feature_stack`),
i.e. exactly the V and F the model sees at eval time.

Needs the `meshgpt` conda env (imports the MeshGPT dataset code):

    conda activate meshgpt
    python metrics/meshgpt_decimated_counts.py

Writes metrics/meshgpt_decimated_counts.json: {pickle_key: {mesh_hash: [V, F]}}.
"""

import json
import os
import pickle
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("BENCH_WORKSPACE", Path(__file__).resolve().parents[1]))
MESHGPT_ROOT = WORKSPACE / "Projects" / "MeshGPT_old"
sys.path.insert(0, str(MESHGPT_ROOT))

from dataset.triangles import create_feature_stack, normalize_vertices  # noqa: E402

# All MeshGPT checkpoints in the benchmark use num_tokens=131 (128 quantization bins).
NUM_TOKENS = 131

# Keys are referenced by MESHGPT_DIRS in fix_all_distance_ratios.py.
PICKLES = {
    # distances/MeshGPT_orig1, from Projects/MeshGPT_old/preprocess_shapenet_quadric767.py
    "meshgpt_767": MESHGPT_ROOT / "data" / "shapenet" / "processed_data.pkl",
}


def main():
    out = {}
    for key, path in PICKLES.items():
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        counts = {}
        for name, verts, faces in zip(data["name_val"], data["vertices_val"], data["faces_val"]):
            # <synset>_<hash>[_decNN] -> <hash>. ShapeNet hashes are hex (no underscores),
            # so dropping the variant suffix keys the mesh by the hash its exported .obj
            # and GT model_normalized.obj are named by.
            mesh_hash = name.split("_")[1]
            _, _, _, _, v, f = create_feature_stack(normalize_vertices(verts), faces, NUM_TOKENS - 3)
            counts[mesh_hash] = [int(len(v)), int(len(f))]
        out[key] = counts
        print(f"{key}: {len(counts)} val meshes  "
              f"ΣV={sum(c[0] for c in counts.values())}  ΣF={sum(c[1] for c in counts.values())}")
    dst = Path(__file__).with_suffix(".json")
    dst.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
