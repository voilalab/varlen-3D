# BACON (Band-limited Coordinate Networks)

## What it is

[BACON](https://github.com/computational-imaging/bacon) is a multiplicative filter network whose
outputs are band-limited. The multiscale variant used here has 8 hidden layers and an output head
after layers 2, 4, 6 and 8. Each head is a progressively finer reconstruction of the same shape.
Each mesh is fitted independently from a point cloud with normals.

**Capacity knob:** output level L ∈ {1, 2, 3, 4}. All four levels come from a single trained
model: evaluating level L needs only the layers up to its head. So the sweep is one training run
per mesh, not four.

**Payload charged per mesh (fp32):** everything needed to evaluate level L, namely
`linear[0..2L-1]`, the level-L output head, and the L Fourier filters those layers reuse. The
filter weights are frozen but sampled randomly per mesh with no seed, so they count as per-mesh data
too.

| Level | Layers | Learnable params | Charged (incl. frozen filters) |
|---|---|---|---|
| 1 | 2 | 132,097 | 132,865 |
| 2 | 4 | 263,937 | 265,473 |
| 3 | 6 | 395,777 | 398,081 |
| 4 | 8 | 527,617 | 530,689 |

## Upstream and our changes

- Upstream: [computational-imaging/bacon](https://github.com/computational-imaging/bacon) @ `c691c69`
- `patches/bacon.patch`:
  - `dataio.py` (`MeshSDF`): out-of-bounds samples are clamped to the domain instead of wrapped
    around. Wrapping teleports points to the far side with a wrong SDF. A quarter of each batch
    is also drawn uniformly over the domain with nearest-surface distances, so the network sees
    off-surface space.
  - `experiments/render_sdf.py`: takes checkpoints, names and point clouds on the command line;
    exports every level separately (non-adaptive) with marching cubes at N = 512; writes
    `<name>_params.txt` with per-level parameter counts.
  - `experiments/train_sdf.py`: a Pillow ≥ 10 compatibility shim (`ANTIALIAS`), and the
    `--coarse_weight` command-line default is 0.1. The config file sets `coarse_weight = 1e-2`,
    which takes precedence.
- `overlays/bacon/` (our scripts):
  - `experiments/train.py`: the driver (one config per mesh, train, then export all levels).
  - `time_inference.py`: decode timing.
- `preprocessing/obj_to_xyz.py` (repo root): builds BACON's input point clouds.

## Environment

```bash
conda env create -f envs/bacon.yml && conda activate bacon
```

## Checkpoints / downloads

None. Every mesh is fitted from scratch.

## Run

```bash
conda activate bacon
python preprocessing/obj_to_xyz.py
cd Projects/bacon/experiments
python train.py
```

1. `obj_to_xyz.py` samples 100,000 surface points with normals from each val mesh (seed 0) and writes
   `data/shapenet/03001627/<hash>/models/model_normalized.xyz` (6 columns: xyz + normal).
2. `train.py` writes `config/sdf/bacon_<hash>_8.ini` from upstream's `bacon_armadillo.ini`. The
   settings are 8 hidden layers, hidden size 256, 200,000 steps, lr 0.01, max frequency 384,
   10,000 on-surface points per batch, and multiscale outputs. It trains with `train_sdf.py`
   (log in `../logs/bacon_<hash>_8/train.log`, checkpoint `checkpoints/model_final.pth`).
3. When all meshes are trained, `train.py` calls `render_sdf.py`, which exports four meshes per
   shape with marching cubes on a 512³ grid.

One mesh takes about 1 hour on a free 48 GB GPU, so the full sweep is about 95 GPU-hours. To spread
it over several GPUs, run copies of the driver on disjoint parts of `val.lst`.

**Output** (what `metrics/distance.py` reads):
```
experiments/outputs/meshes/bacon_<hash>_8_<L>.obj    # level-L reconstruction
experiments/outputs/meshes/bacon_<hash>_8_params.txt # level list (used to find the files)
```

**Timing** (for `--time-markers` only):
```bash
cd .. && python time_inference.py     # -> logs/inference_times.csv (one run = all 4 levels)
```

## Score

```bash
cd ../.. && conda activate distance
python metrics/distance.py --project bacon-flat     # -> distances/bacon/level_<L>_distance.txt
```

BACON is scored against the GT mesh (`model_normalized.obj`), like every other method, not against
its input point cloud.

## Expected results (committed `distances/bacon/`)

| Level | ratio | meshes scored | median CD | mean CD |
|---|---|---|---|---|
| 1 | 2.8806 | 100 | 1.861e-03 | 3.788e-03 |
| 2 | 5.7557 | 100 | 1.770e-03 | 3.834e-03 |
| 3 | 8.6308 | 100 | 1.855e-03 | 3.854e-03 |
| 4 | 11.5058 | 100 | 1.984e-03 | 4.033e-03 |

All 100 meshes succeed at every level.
