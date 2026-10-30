"""Re-split the EXISTING processed_data.pkl geometry along the benchmark split lists.

Rationale: the old pickle's geometry is fine (Blender planar decimation from the
original MeshGPT preprocessing, the same distribution 749-0 and all prior runs
were trained on) -- only its train/val SPLIT is leaked (val is 115/115 contained
in train at shape level, as decimation variants). QEM re-decimation
(build_clean_pickle.py) fails on ~26% of meshes (disconnected triangle soups),
so instead: keep the old geometry, reassign entries by base shape hash.

  train = ALL decimation variants of shapes in train_subset.lst (augmentation,
          matching the old training protocol)
  val   = ONE variant per shape in val_subset.lst (prefer _dec05, the variant
          the old val split used; else the variant with the most faces)

Shapes absent from the old pickle (never survived the original preprocessing)
are reported and skipped.

Run server-side (meshgpt env, numpy<2):
    python dataset/repickle_clean_split.py \
        --pickle data/shapenet/processed_data.pkl \
        --out data/shapenet/processed_data_clean.pkl
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if int(np.__version__.split('.')[0]) >= 2:
    sys.exit(f"Refusing to write pickle with numpy {np.__version__}; use the meshgpt env (numpy<2).")

parser = argparse.ArgumentParser()
parser.add_argument('--pickle', default='data/shapenet/processed_data.pkl')
parser.add_argument('--splits-dir', default='data/splits')
parser.add_argument('--out', default='data/shapenet/processed_data_clean.pkl')
parser.add_argument('--category', default='03001627')
args = parser.parse_args()

with open(args.pickle, 'rb') as f:
    old = pickle.load(f)


def load_lst(name):
    path = Path(args.splits_dir) / name
    return [line.split('/')[1] for line in path.read_text().split() if '/' in line]


train_hashes = load_lst('train_subset.lst')
val_hashes = load_lst('val_subset.lst')
assert not set(train_hashes) & set(val_hashes)

# Index every old entry (train and val alike -- the split is being redefined)
# by base shape hash. Dedupe exact names (the old splits overlap on 5 entries).
by_hash = defaultdict(dict)  # hash -> {name: (verts, faces)}
for s in ('train', 'val'):
    for name, v, f in zip(old[f'name_{s}'], old[f'vertices_{s}'], old[f'faces_{s}']):
        parts = name.split('_')
        if parts[0] != args.category:
            continue
        by_hash[parts[1]][name] = (v, f)

n_tr, v_tr, f_tr = [], [], []
missing_train = []
for h in train_hashes:
    if h not in by_hash:
        missing_train.append(h)
        continue
    for name, (v, f) in sorted(by_hash[h].items()):
        n_tr.append(name)
        v_tr.append(v)
        f_tr.append(f)

n_va, v_va, f_va = [], [], []
missing_val = []
for h in val_hashes:
    if h not in by_hash:
        missing_val.append(h)
        continue
    variants = by_hash[h]
    dec05 = [n for n in variants if n.endswith('_dec05')]
    name = dec05[0] if dec05 else max(variants, key=lambda n: len(variants[n][1]))
    v, f = variants[name]
    n_va.append(name)
    v_va.append(v)
    f_va.append(f)

tr_shapes = {n.split('_')[1] for n in n_tr}
va_shapes = {n.split('_')[1] for n in n_va}
assert not tr_shapes & va_shapes, 'shape-level train/val overlap!'

data = {
    'name_train': n_tr, 'vertices_train': v_tr, 'faces_train': f_tr,
    'name_val': n_va, 'vertices_val': v_va, 'faces_val': f_va,
}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'wb') as f:
    pickle.dump(data, f)

print(f'train: {len(tr_shapes)}/{len(train_hashes)} shapes covered '
      f'({len(n_tr)} entries incl. decimation variants); missing {len(missing_train)}')
print(f'val:   {len(va_shapes)}/{len(val_hashes)} shapes covered; missing {len(missing_val)}')
print(f'val variants used: {sum(1 for n in n_va if n.endswith("_dec05"))} x _dec05, '
      f'{sum(1 for n in n_va if not n.endswith("_dec05"))} x other')
print(f'max faces: train={max(len(f) for f in f_tr)} val={max(len(f) for f in f_va)}')
print(f'missing val shapes: {missing_val}')
print(f'wrote {out}')
