# Documentation

Step-by-step guides for reproducing every number and figure in this package. Start here, then open
the page for the method you are running.

| Page | What it covers |
|---|---|
| [ngp.md](ngp.md) | Instant-NGP: per-mesh hash-grid SDF fits, `log2_hashmap_size` sweep |
| [ngf.md](ngf.md) | Neural Geometry Fields: per-mesh fits, level-of-detail sweep |
| [bacon.md](bacon.md) | BACON: per-mesh band-limited SDF, 4 output levels from one model |
| [cons.md](cons.md) | Convolutional Occupancy Networks: one model per `c_dim`, trained on 1000 chairs |
| [meshgpt_original.md](meshgpt_original.md) | MeshGPT (original): released vocabulary model, evaluated on our val split |
| [metrics.md](metrics.md) | Chamfer distance, compression ratio, distance-file format, figures |
| [provided_curves.md](provided_curves.md) | VecSet / VecSet-HAE and MeshGPT (ours): supplied as finished distance files |

## The pipeline

```
ShapeNet chairs (data/shapenet)
   │
   ├─ setup_methods.sh ── clones upstream @ pinned commit, applies patches/, adds overlays/
   │
   ├─ per-method sweep ── Projects/<method>/...  reconstructed meshes for the 100 val shapes,
   │                                             one set per capacity setting
   │
   ├─ metrics/distance.py ── distances/<method>/<setting>_distance.txt  (Chamfer + ratio per mesh)
   │
   └─ graphs/*.py ── figures/
```

Every method is scored by the same script (`metrics/distance.py`) against the same GT meshes. The
methods differ only in how they produce the reconstructed meshes. Method-internal metrics
(training losses, in-trainer Chamfer numbers) are not used anywhere.

## Common setup (do this once)

1. **Data.** Download [ShapeNetCore v2](https://shapenet.org/) and put the chairs (synset
   `03001627`) listed in `data/shapenet/train.lst` and `data/shapenet/val.lst` at
   `data/shapenet/03001627/<hash>/models/model_normalized.obj`. Only these 1100 meshes are used.
   - `val.lst`: 100 meshes, the evaluation set for every method.
   - `train.lst`: 1000 meshes, used only by methods with a shared model (CONs). Per-mesh methods
     (NGP, NGF, BACON) never see it.
2. **Method code.** `bash setup_methods.sh` (or `bash setup_methods.sh <method> ...`).
3. **Environments.** `conda env create -f envs/<env>.yml` for each method you run, plus
   `envs/distance.yml` for scoring and plotting. Packages that need a special build are commented
   out in the yml, with the install command next to each.

## Compute budget (from our runs)

| Method | What runs | GPU time (approx.) |
|---|---|---|
| NGP | 10 settings × 100 meshes, 10k steps each | ~11 h |
| NGF | 12 LODs × 100 meshes | several hours (dominated by high LODs) |
| BACON | 100 meshes × 200k steps (all 4 levels from one run) | ~1 h per mesh, ~95 h total |
| CONs | 8 models, each capped at 2 h of training | ~16 h + generation |
| MeshGPT (original) | evaluation only, no training | minutes |
| Scoring | `metrics/distance.py`, 10k points per mesh | minutes per method |

All runs used a single NVIDIA GPU with 48 GB of memory. Per-mesh methods can be parallelised by
splitting `val.lst`.

## What "reproduce" means here

- **Figures** redraw exactly from the committed `distances/` (section 0 of the top-level README).
- **Compression ratios** are deterministic: they depend only on the payload definitions and the
  GT meshes, and recompute exactly.
- **Chamfer distances** are re-measured from reconstructions. Training is stochastic (random
  initialisation, sampling), so retrained models give close but not bit-identical numbers. Each
  method page has the committed per-setting numbers to compare against.
