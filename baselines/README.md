# Mesh compression benchmark: replication package

Rate-distortion benchmark of neural 3D shape representations on ShapeNet chairs (synset
`03001627`): Chamfer distance vs. compression ratio.

This folder trains and scores the **baselines** (NGP, NGF, BACON, CONs, MeshGPT original). The
curves for our method (**VecSet-HAE** and **MeshGPT (ours)**) and for the original **VecSet** base
model come from the code at the top level of this repository ([`../SDF/`](../SDF/) and
[`../meshgpt/`](../meshgpt/)). Their distance files are included here as-is and used directly by
the figures.

The pipeline is: **train each baseline → reconstruct the 100 val meshes → score them with
`metrics/distance.py` → plot with `graphs/`**. All measured numbers are committed in `distances/`, so
the figures can be redrawn without running any training.

```
setup_methods.sh   clone each baseline's public upstream at a pinned commit, apply patches/, add overlays/
patches/<method>.patch  our modifications to upstream files
overlays/<method>/ files we wrote from scratch: sweep/timing drivers (paths mirror the upstream repo)
envs/*.yml         conda environments (one per method + `distance` for scoring)
data/shapenet/     val.lst / train.lst; put ShapeNet chairs here (see step 1)
preprocessing/     OBJ -> .xyz point cloud with normals (BACON input)
metrics/           Chamfer distance + compression ratio -> distances/
distances/         the measurements: one dir per method, one .txt per setting
graphs/            figures from distances/ -> figures/
figures/           generated figures
third_party_licenses/  copies of the upstream licenses (see THIRD_PARTY_NOTICES.md)
```

**Detailed per-method guides** (how each method works, what we changed, checkpoints and downloads,
expected numbers) are in [`docs/`](docs/README.md).

## 0. Just redraw the figures

```bash
conda env create -f envs/distance.yml && conda activate distance
python graphs/graph_paper_figures.py      # figures/paper_figures/fig_rd_{implicit,mesh}.{png,pdf}
python graphs/graph_median.py             # figures/graph_median_cd.png
python graphs/graph_mean.py               # figures/graph_mean_cd.png
```

`--time-markers` on `graph_paper_figures.py` scales markers by decode time. It reads the
`inference_times.csv` files written in step 3; the provided curves ship theirs in `distances/`.

## Our method's curves (produced by `../SDF/` and `../meshgpt/`, not by this folder)

| Directory | Method | Files |
|---|---|---|
| `distances/VecSet_HAE/` | VecSet + ours | `ratio_L<k>.txt` (k tokens kept), `inference_times.csv` |
| `distances/VecSet_GT/` | VecSet, original | `ratio.txt` (single fixed-rate point) |
| `distances/MeshGPT_HAE/` | MeshGPT + ours | `keepk_<k>_distance.txt`, `inference_times.csv` |

They use the same line format and the same Chamfer/ratio definitions as the files computed here.
`metrics/fix_all_distance_ratios.py` leaves them untouched.

## 1. Data

Download ShapeNetCore v2 and place the chair meshes listed in `data/shapenet/{train,val}.lst` at:

```
data/shapenet/03001627/<hash>/models/model_normalized.obj
```

`val.lst` (100 meshes) is the evaluation set for every method. `train.lst` (1000 meshes) is used to
train CONs and MeshGPT.

## 2. Method code

```bash
bash setup_methods.sh            # or: bash setup_methods.sh ngf bacon ...
```

This clones each repo into `Projects/<method>/` at the commit below, applies `patches/<method>.patch`
(our edits to upstream files), and copies in `overlays/<method>/` (our own scripts). Re-running it is
safe; already-applied patches are skipped. Upstream code is not redistributed here and stays under
its own license.

| Method | Upstream | Commit | Env |
|---|---|---|---|
| NGP | [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) | `018dea6` | `instant-ngp` |
| NGF | [iveevi/ngf](https://github.com/iveevi/ngf) | `1cfe0b8` | `ngf` |
| BACON | [computational-imaging/bacon](https://github.com/computational-imaging/bacon) | `c691c69` | `bacon` |
| CONs | [autonomousvision/convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks) | `838bea5` | `conv_onet` |
| MeshGPT (original) | [audi/MeshGPT](https://github.com/audi/MeshGPT) | `028648f` | `meshgpt` |

Create the environments with `conda env create -f envs/<env>.yml`. A few packages in the yml files
are commented out because they have to be built or installed from a specific index (`pytorch3d`,
`torch-scatter`, `nvdiffrast`, NGF's `ngfutil`). The install command is next to each one.

## 3. Train / reconstruct each baseline

Short version below; see `docs/<method>.md` for details.

All commands start from this `baselines/` folder. Every sweep writes its reconstructions inside its own
`Projects/<method>/`, which is where `metrics/distance.py` looks for them.

**NGP** (per-mesh fit, `log2_hashmap_size` ∈ {2,3,5,7,8,10,12,14,17,19})
```bash
conda activate instant-ngp
cd Projects/instant-ngp && cmake . -B build && cmake --build build --config RelWithDebInfo -j
./run_param_search.sh                 # -> parameter_sweep_results/log2_<T>/03001627/<hash>/
python time_inference.py              # -> parameter_sweep_results/inference_times.csv
```

**NGF** (per-mesh fit, `lod` ∈ {4 … 5000})
```bash
conda activate ngf && pip install ./Projects/ngf/extensions
cd Projects/ngf && python sweep.py    # -> source/results/{stl,torched}/<hash>-lod<N>-f40.*
python time_inference.py
```

**BACON** (per-mesh fit, levels 1–4 from one 8-layer model)
```bash
conda activate bacon
python preprocessing/obj_to_xyz.py    # writes model_normalized.xyz (+normals) next to each .obj
cd Projects/bacon/experiments && python train.py   # -> outputs/meshes/bacon_<hash>_8_<level>.obj
cd .. && python time_inference.py
```

**CONs** (shared model trained on `train.lst`, `c_dim` ∈ {4 … 32})
```bash
conda activate conv_onet
cd Projects/convolutional_occupancy_networks && python setup.py build_ext --inplace
(cd data && bash ../scripts/download_data.sh 03001627)    # Occupancy Networks preprocessed ShapeNet
cp ../../data/shapenet/train.lst data/ShapeNet/03001627/train.lst
cp ../../data/shapenet/val.lst   data/ShapeNet/03001627/val.lst
cp ../../data/shapenet/val.lst   data/ShapeNet/03001627/test.lst
python sweep.py --gt-mesh-dir ../../data/shapenet
python time_inference.py
```

**MeshGPT (original)**
```bash
conda activate meshgpt
cd Projects/MeshGPT_old
# Pretrained models (audi's Google Drive, linked in Projects/MeshGPT_old/README.md) -> pretrained/
# Uses pretrained/vocabulary/checkpoints/749-0.ckpt. No training: config sets eval_only=true.
python preprocess_shapenet_quadric767.py   # -> data/shapenet/processed_data.pkl (decimated to <=767 faces)
CUDA_VISIBLE_DEVICES=0 bash scripts/train_and_eval_meshgpt.sh   # val reconstructions -> runs/vocabulary/mesh_val/
python time_inference.py
```

## 4. Chamfer distances and compression ratios

```bash
conda activate distance
python metrics/distance.py --project ngp
python metrics/distance.py --project ngf
python metrics/distance.py --project cons
python metrics/distance.py --project bacon-flat
python metrics/distance.py --project meshgpt-orig
```

Each writes `distances/<method>/<setting>_distance.txt`, one line per val mesh:
`Chamfer Distance for <hash> with ratio <ratio>: <chamfer>`. `--configs a,b` scores a subset,
`--list-configs` lists the settings, `--out-dir` writes somewhere else.

**Chamfer:** 10,000 surface points are sampled from the GT mesh and from the reconstruction. Each
cloud is normalized to the unit cube, the reconstruction is ICP-aligned to the GT, and then
symmetric mean squared nearest-neighbour distance is computed with pytorch3d. GT samples are seeded
by the GT file's absolute path, so numbers move slightly (sampling noise) if `data/` lives elsewhere.

**Ratio:** Σ per-mesh payload bits / Σ uncompressed GT mesh bits over the scored meshes. Shared
weights are free; per-mesh latents and weights are charged. Definitions are in
`metrics/bit_accounting.py`. MeshGPT's payload uses the decimated face counts cached in
`metrics/meshgpt_decimated_counts.json`. To regenerate that cache, or to re-stamp ratios without
recomputing Chamfer:

```bash
conda activate meshgpt
python metrics/meshgpt_decimated_counts.py    # needs Projects/MeshGPT_old/data/shapenet/processed_data.pkl
conda activate distance
python metrics/fix_all_distance_ratios.py     # --dry-run to preview; provided curves are skipped
```

Then redraw the figures (step 0).

### Using an existing workspace

All metrics scripts read `Projects/` and `data/` from `$BENCH_WORKSPACE`, which defaults to this
repo. To score reconstructions that live elsewhere:

```bash
BENCH_WORKSPACE=/path/to/workspace python metrics/distance.py --project ngp
```

## Notes on the committed numbers

| Files | How they were produced | Reproduced by step 4 |
|---|---|---|
| `MeshGPT_orig1` | seeded sampling + ICP (this script) | yes: medians identical, most values to the last digit |
| `ngp`, `CONs`, `ngf`, `bacon` | earlier version of the script: same pipeline but unseeded sampling | up to sampling noise (NGP: ~3% median per mesh) |

`--no-icp` turns off the ICP alignment step.

## License

Our own code (scripts, metrics, figures, docs, and the overlays we wrote) is under the MIT License;
see [`LICENSE`](LICENSE). The patches modify third-party code, so each stays under the license of
its upstream project. Instant-NGP and MeshGPT are **non-commercial only**, and NGF and BACON
publish no license. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
