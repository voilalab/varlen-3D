# Provided curves (VecSet / VecSet-HAE, MeshGPT ours)

These curves come from our method's code at the top level of this repository and are included
here as finished distance files. The baselines package does not retrain or re-score them. The figure scripts read them
directly, and `metrics/fix_all_distance_ratios.py` leaves them untouched.

| Directory | Method | Files | Points |
|---|---|---|---|
| `distances/VecSet_HAE/` | VecSet + ours (hyper-autoencoder on a frozen VecSet base; any prefix of k tokens decodes) | `ratio_L<k>.txt`, k = 1…1024; `inference_times.csv` | 11 files, 9 plotted (see below) |
| `distances/VecSet_GT/` | VecSet, original base model (1024 × 32 latent) | `ratio.txt` | single point |
| `distances/MeshGPT_HAE/` | MeshGPT + ours (nested-dropout token sequence on the vocabulary; `keep_k` tokens kept) | `keepk_<k>_distance.txt`, k = 1…256; `inference_times.csv` | 9-point curve |

## Format and comparability

Same line format as every other distance file:

```
Chamfer Distance for <mesh_hash> with ratio <ratio>: <chamfer>
```

- **Same evaluation set:** the 100 meshes in `data/shapenet/val.lst`. MeshGPT (ours) has 99, because
  `8ec247d0e133ad3ce922ceea1248b9b0` cannot be decimated below its face budget.
- **Ratios** follow the same convention as the rest of the benchmark: per-mesh latents in fp32,
  shared weights free. VecSet: `L × 32` fp32 values per mesh (L = 1024 for the base model). MeshGPT
  (ours): `keep_k × 576` fp32 latents per mesh.
- **Chamfer:** MeshGPT (ours) was scored with this package's `metrics/distance.py` (10k points,
  normalisation, ICP) on its exported reconstructions. The VecSet files were written by VecSet's own
  inference script.
- **NaN entries (VecSet-HAE):** a NaN Chamfer value marks a reconstruction that could not be scored
  (VecSet's script writes `nan` where `metrics/distance.py` would write a `[FAIL]` line).
  L1 and L2 are NaN for all 100 meshes, L4 for 19 and L8 for 2. The figure scripts drop NaN values,
  so L1/L2 are not drawn and the L4/L8 points are the mean/median over the 81/98 meshes that
  decoded. Those two points are therefore optimistic compared with the other settings.
- `inference_times.csv` (`L`/`keep_k`, `inference_time_s`) is only used for `--time-markers` marker
  sizes. MeshGPT (ours)' file has 4 rows per `keep_k` besides the 99 val meshes (`sample` = `_0`…`_3`,
  no synset prefix); they are averaged in with the rest.

## Where they come from

The code behind these curves is in this repository:

- VecSet-HAE: [`../../SDF/`](../../SDF/), built on [VecSetX](https://github.com/1zb/VecSetX); the
  base is the released `learnable_vec1024x32_dim1024_depth24_nb` model.
- MeshGPT (ours): [`../../meshgpt/`](../../meshgpt/), built on
  [audi/MeshGPT](https://github.com/audi/MeshGPT). See [`../../meshgpt/README.md`](../../meshgpt/README.md) for training.
