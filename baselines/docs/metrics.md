# Metrics, distance files and figures

## Chamfer distance (`metrics/distance.py`)

Every method is scored the same way, against the same GT points:

1. Load the GT `data/shapenet/03001627/<hash>/models/model_normalized.obj` and the reconstruction with
   trimesh (`force="mesh"`). The GT is always the mesh, never a method's own input such as BACON's
   `.xyz` or CONs' `pointcloud.npz`.
2. Sample **10,000** surface points from each. GT samples are seeded with
   `crc32(<absolute GT path>)`, so every method sees identical GT points. Reconstruction samples use
   seed 0.
3. Normalise each cloud independently: subtract the centroid, divide by the max absolute coordinate.
4. **ICP-align** the reconstruction to the GT (`trimesh.registration.icp`, max 50 iterations).
   `--no-icp` skips this step.
5. Chamfer = mean squared nearest-neighbour distance in both directions, summed
   (`pytorch3d.loss.chamfer_distance`, exact nearest neighbours, GPU if available).

A failed load or an empty mesh is written as a `[FAIL]` line and excluded from statistics.

Because the GT seed is derived from the absolute path, moving `data/` changes the GT samples. Results
then move by sampling noise only (a few percent per mesh, less in aggregate).

```bash
conda activate distance
python metrics/distance.py --project {ngp|ngf|cons|bacon-flat|meshgpt-orig}
  [--configs log2_14,log2_17]   # subset of settings (run several in parallel)
  [--list-configs]              # print the settings found and exit
  [--out-dir DIR] [--out-name NAME] [--input-dir DIR] [--num-points N] [--no-icp]
```

Inputs are read from `$BENCH_WORKSPACE/Projects/...` and `$BENCH_WORKSPACE/data/shapenet`.
`BENCH_WORKSPACE` defaults to this repo.

## Compression ratio (`metrics/bit_accounting.py`)

One aggregate ratio per setting (per distance file), in bits:

```
ratio = Σ per-mesh payload bits / Σ GT mesh bits      over the meshes scored in that file
```

- **GT mesh bits** = `3V × 32` (fp32 coordinates) + `3F × ceil(log2 V)` (vertex indices), with V
  and F from `trimesh.load(model_normalized.obj, force="mesh")`.
- **Payload**: the non-shared parameters the decoder needs for that mesh. Weights, codebooks and
  anything else shared across meshes are free. Structural connectivity (NGF quad layout, MeshGPT
  face lists) is required to decode but is not a model parameter, so it is not charged by default.
  Floats are charged at the precision they are stored in.

| Method | Payload per mesh |
|---|---|
| NGP | hash-grid entries + MLP weights as allocated by tiny-cuda-nn, fp16 |
| NGF | MLP weights + patch points + features, fp32 (from the `.pt`) |
| CONs | `32³ × c_dim` fp32 feature grid |
| BACON | layers, head and Fourier filters needed for level L, fp32 |
| MeshGPT (original) | `3 F_dec × 2` VQ indices × 14 bits = 84 bits/face |

Every line of a distance file carries the same aggregate ratio. To re-stamp ratios without
recomputing Chamfer (for example after changing a payload definition):

```bash
conda activate distance         # torch (NGF .pt files) + trimesh (GT meshes)
python metrics/fix_all_distance_ratios.py --dry-run
python metrics/fix_all_distance_ratios.py
python metrics/fix_all_distance_ratios.py --charge-connectivity   # reference variant, not plotted
```

Our method's curves (VecSet, MeshGPT ours) are never rewritten.

## Distance file format

```
distances/<method>/<setting>_distance.txt
Chamfer Distance for <mesh_hash> with ratio <ratio>: <chamfer>
[FAIL] Could not compute Chamfer Distance for <mesh_hash>
```

| Directory | Files | Setting |
|---|---|---|
| `ngp/` | `log2_<T>_distance.txt` | hash table size `2^T` |
| `ngf/` | `lod_<N>_distance.txt` | level of detail |
| `bacon/` | `level_<L>_distance.txt` | output level |
| `CONs/` | `cdim_<C>_distance.txt` | feature dimension |
| `MeshGPT_orig1/` | `distance.txt` | single point |
| `VecSet_HAE/` (provided) | `ratio_L<k>.txt` | tokens kept |
| `VecSet_GT/` (provided) | `ratio.txt` | single point |
| `MeshGPT_HAE/` (provided) | `keepk_<k>_distance.txt` | tokens kept |

## Figures (`graphs/`)

```bash
conda activate distance
python graphs/graph_paper_figures.py [--stat mean|median] [--time-markers] [--common-meshes] [--free-axes]
python graphs/graph_median.py
python graphs/graph_mean.py
```

- `graph_paper_figures.py` draws the two paper panels with shared axes, writing to
  `figures/paper_figures/fig_rd_implicit.{png,pdf}` (NGP, BACON, CONs, VecSet-HAE, plus a VecSet
  star) and `fig_rd_mesh.{png,pdf}` (NGF, MeshGPT-ours, plus a MeshGPT-original star). Each point is
  one setting: x = ratio, y = mean (default) or median Chamfer over the scored meshes. Both axes are
  logarithmic. The dashed line at ratio 1 marks a payload as large as the raw mesh.
- `--time-markers` scales marker area by cumulative decode time over the ladder `1..k`, read from
  each method's `inference_times.csv`.
- `--common-meshes` restricts every curve to meshes scored by all methods. Currently that set is
  empty: NGP's lowest settings (`log2_2`/`3`/`5`) keep only 11–16 largely disjoint meshes, and
  VecSet-HAE L1/L2 have no finite value. (NGF alone still shares 44 meshes across its LODs.)
- `graph_median.py` / `graph_mean.py` draw all methods on one plot (`figures/graph_*_cd.png`).
