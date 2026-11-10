# MeshGPT (original)

## What it is

[MeshGPT](https://github.com/audi/MeshGPT) tokenises a triangle mesh with a learned **vocabulary**: a
graph-convolutional encoder, a residual vector quantiser (VQ) and a decoder. A transformer then
generates token sequences. This benchmark uses only the vocabulary stage, as a mesh compressor:
encode a mesh to VQ indices, then decode it back.

We do **not** train it. We use the authors' released vocabulary checkpoint and only run it on our
val meshes. It is a single fixed-rate point in the figures, not a curve.

**Payload charged per mesh:** the VQ indices for every face corner, `3 F_dec × 2 levels` indices of
`log2(16384) = 14` bits each (84 bits per face). Here `F_dec` is the face count of the decimated mesh
the model sees. The codebook is shared, so it is free. See [metrics.md](metrics.md).

## Upstream and our changes

- Upstream: [audi/MeshGPT](https://github.com/audi/MeshGPT) @ `028648f`. Upstream files use CRLF line
  endings; `setup_methods.sh` converts the files it patches to LF before applying the patch.
- `patches/MeshGPT_old.patch`:
  - `config/meshgpt.yaml`: `eval_only: true`, `resume: pretrained/vocabulary/checkpoints/749-0.ckpt`,
    `only_chairs: true`.
  - `trainer/train_vocabulary.py`: with `eval_only`, runs `trainer.validate` instead of `fit`.
    Validation now reconstructs **every** val shape, not a few samples, and saves each one as
    `runs/vocabulary/mesh_val/<synset>_<hash>.obj`. Dataloaders are not shuffled. It also writes an
    in-trainer Chamfer file to `distances/MeshGPT_orig/`, which uses a different method and is not used.
  - `trainer/train_transformer.py`, `trainer/__init__.py`, `inference/infer_meshgpt.py`: similar
    in-trainer evaluation for the transformer stage, plus checkpoint-retention changes. These are not
    used for the benchmark.
- `overlays/MeshGPT_old/` (our scripts):
  - `preprocess_shapenet_quadric767.py`: builds the input pickle.
  - `scripts/train_and_eval_meshgpt.sh`: the run driver.
  - `time_inference.py`: per-shape encode→decode timing.
  - `config.yaml`: reference transformer config, unused.

## Environment

```bash
conda env create -f envs/meshgpt.yml && conda activate meshgpt
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5'
```

## Checkpoints / downloads

Download the authors' **pretrained models** from the link in the upstream README
([Google Drive](https://drive.google.com/drive/folders/1Gzuxn6c1pguvRWrsedmCa9xKtest8aC2)) and put the
`pretrained/` folder inside `Projects/MeshGPT_old/`. The file used is:

```
Projects/MeshGPT_old/pretrained/vocabulary/checkpoints/749-0.ckpt
  size    330,531,564 bytes
  sha256  394222c8847458375f7c2378c55ea87a3d4ce49d8365ed8e81db1994e174d6b8
```

Verify the download before running:

```bash
sha256sum Projects/MeshGPT_old/pretrained/vocabulary/checkpoints/749-0.ckpt
```

**Do not use the authors' preprocessed data** (`processed_data.pkl` in their Drive folder). Its
val split is a different set of shapes: only 2 of our 100 val meshes are in it. Build the pickle
from our splits instead (step 1 below).

## Run

```bash
cd Projects/MeshGPT_old

# 1. Input pickle: train + val meshes, each quadric-decimated to <= 767 faces
python preprocess_shapenet_quadric767.py        # -> data/shapenet/processed_data.pkl

# 2. Reconstruct every val mesh with the pretrained vocabulary (no training; eval_only=true)
CUDA_VISIBLE_DEVICES=0 bash scripts/train_and_eval_meshgpt.sh
```

Step 1 reads `data/shapenet/{train,val}.lst`, decimates every mesh with **open3d's quadric decimation**
to at most 767 faces (`--max-faces`), and pickles vertices and faces in MeshGPT's format. It
keeps 981 train and 99 val meshes and reproduces our pickle exactly (identical vertices and faces
for every mesh). The decimation backend matters: trimesh 4.x's default (`fast_simplification`)
keeps only 77 val meshes under the cap and gives different meshes.

Step 2 loads `749-0.ckpt`, runs one validation pass over the val split, and writes:

```
runs/vocabulary/mesh_val/03001627_<hash>.obj      # reconstruction, in the model's axis order (z, x, y)
```

The script's name is historical. With the patched config it only evaluates. It defaults to
`CUDA_VISIBLE_DEVICES=1,2`, so set the variable yourself.

**Timing** (for `--time-markers` only):
```bash
python time_inference.py     # -> pretrained/inference_times.csv
```

## Score

```bash
cd ../.. && conda activate distance
python metrics/distance.py --project meshgpt-orig     # -> distances/MeshGPT_orig1/distance.txt
```

`metrics/distance.py` maps the reconstruction back from the model's axis order (`[1, 2, 0]`) and
reads `F_dec` for the payload from `metrics/meshgpt_decimated_counts.json`. To regenerate that cache
from your own pickle:

```bash
conda activate meshgpt
python metrics/meshgpt_decimated_counts.py
```

## Expected results (committed `distances/MeshGPT_orig1/`)

| ratio | meshes scored | median CD | mean CD |
|---|---|---|---|
| 0.0376 | 99 | 6.562e-04 | 2.074e-03 |

Re-scoring our reconstructions with `metrics/distance.py` reproduces these values: medians identical,
most per-mesh values to the last digit.

## Known behaviour

- **1 of 100 val meshes is missing** (`8ec247d0e133ad3ce922ceea1248b9b0`): it cannot be decimated
  below the face budget, so it is not in the pickle.
- **MeshGPT reconstructs the decimated mesh but is scored against the full-resolution GT**, like
  every other method. Decimation alone sets a floor on its Chamfer distance; the vocabulary is
  close to lossless on its own input.
- The ratio is small because the denominator is the undecimated fp32 GT mesh; the decimation step
  itself is not charged.
