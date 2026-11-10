# NGF (Neural Geometry Fields)

## What it is

[Neural Geometry Fields](https://github.com/iveevi/ngf) represent a mesh as a coarse quad patch layout
plus per-vertex feature vectors and a small MLP that displaces the patches into the fine surface. Each
mesh is fitted independently: there is no shared model.

**Capacity knob:** level of detail `lod`, the number of quad patches. The input mesh is decimated to
about `2 × lod` faces and quadrangulated before fitting. Swept values:
`lod` ∈ {4, 8, 16, 32, 64, 125, 250, 500, 1000, 2000, 4000, 5000}, with 40 features per vertex.

**Payload charged per mesh:** MLP weights + patch vertex positions (`points`) + per-vertex features,
all fp32, counted from the saved `.pt`. The quad connectivity (`complexes`) is required to decode,
but it is structure rather than a model parameter, so it is not charged. See
[metrics.md](metrics.md).

## Upstream and our changes

- Upstream: [iveevi/ngf](https://github.com/iveevi/ngf) @ `1cfe0b8` (with its submodules)
- `patches/ngf.patch`:
  - `source/train.py`: the quadrangulation timeout is a parameter (`--quad-timeout`, upstream
    hard-coded 60 s). ShapeNet meshes need longer. `--display` is opt-in, for headless servers.
  - `source/util/miscellaneous.py`: skips degenerate (zero-area or coplanar) face clusters when
    placing cameras, with a single-view fallback. Upstream divides by a zero norm and crashes on
    some ShapeNet meshes.
- `overlays/ngf/` (our scripts):
  - `sweep.py`: the sweep driver.
  - `time_inference.py`: decode timing.

## Environment

```bash
conda env create -f envs/ngf.yml && conda activate ngf
pip install git+https://github.com/NVlabs/nvdiffrast@v0.4.0
pip install ./Projects/ngf/extensions          # builds the ngfutil CUDA extension
```

## Checkpoints / downloads

None. Every mesh is fitted from scratch.

## Run

```bash
cd Projects/ngf
python sweep.py                 # all LODs, val split
# python sweep.py --lods 500,250 --max-meshes 5    # quick test
```

For each LOD and mesh, the driver:
1. converts `model_normalized.obj` to STL with trimesh (NGF's meshio loader crashes on ShapeNet OBJs
   with inconsistent `vt`/`vn` records),
2. runs `source/train.py --mesh <stl> --lod <N> --features 40 --batch 10 --quad-timeout 600`,
3. records the parameter count and copies the results into `parameter_sweep_results/lod_<N>/<hash>/`.

**Output** (what `metrics/distance.py` reads):
```
source/results/stl/<hash>-lod<N>-f40.stl       # reconstruction (scored)
source/results/torched/<hash>-lod<N>-f40.pt    # model (payload is counted from this)
parameter_sweep_results/lod_<N>/<hash>/        # copies + info.json, used by time_inference.py
```

**Timing** (for `--time-markers` only):
```bash
python time_inference.py        # -> parameter_sweep_results/inference_times.csv
```

## Score

```bash
cd ../.. && conda activate distance
python metrics/distance.py --project ngf      # -> distances/ngf/lod_<N>_distance.txt
```

`metrics/distance.py` loads each `.pt` with `torch` to count the payload. The `.pt` pickles NGF's
`MLP` class; `metrics/bit_accounting.py` substitutes a plain `nn.Module` for it while loading, so
NGF's own source and its dependencies (`ngfutil`, `meshio`, ...) are not needed in the `distance` env.

## Expected results (committed `distances/ngf/`)

| lod | ratio | meshes scored | median CD | mean CD |
|---|---|---|---|---|
| 4 | 0.3030 | 96 | 2.277e-01 | 2.422e-01 |
| 8 | 0.3116 | 99 | 2.514e-01 | 2.573e-01 |
| 16 | 0.3191 | 100 | 2.622e-01 | 2.946e-01 |
| 32 | 0.3211 | 98 | 2.311e-01 | 2.652e-01 |
| 64 | 0.3201 | 92 | 2.164e-01 | 2.419e-01 |
| 125 | 0.3503 | 92 | 1.862e-01 | 1.897e-01 |
| 250 | 0.3692 | 82 | 8.115e-02 | 1.229e-01 |
| 500 | 0.4558 | 81 | 5.014e-02 | 1.240e-01 |
| 1000 | 0.6186 | 81 | 4.097e-02 | 1.281e-01 |
| 2000 | 0.8858 | 83 | 5.263e-02 | 1.406e-01 |
| 4000 | 1.2690 | 79 | 1.566e-01 | 1.733e-01 |
| 5000 | 1.3741 | 77 | 1.518e-01 | 1.720e-01 |

## Known behaviour

- **Quadrangulation times out on a different subset of meshes at each LOD** (more often at high LOD),
  so 77–100 meshes are scored per setting. Raising `--quad-timeout` recovers some of them.
- At low LOD the ratio barely changes, because the MLP weights dominate the payload.
- The committed Chamfer values come from an earlier version of `metrics/distance.py` that sampled
  points without a seed; re-scoring gives values within sampling noise.
