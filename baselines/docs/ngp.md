# Instant-NGP (hash-grid SDF)

## What it is

[Instant-NGP](https://github.com/NVlabs/instant-ngp) fits a small MLP plus a multi-resolution hash-grid
encoding to one mesh's signed distance field. Every mesh is optimised independently: there is no
shared model and no training set.

**Capacity knob:** `log2_hashmap_size` T. Each of the 16 levels of the hash grid holds up to `2^T`
feature vectors (2 features each), so larger T means more parameters and higher fidelity.
Swept values: T ∈ {2, 3, 5, 7, 8, 10, 12, 14, 17, 19}.

**Payload charged per mesh:** every hash-grid entry and MLP weight tiny-cuda-nn allocates, in fp16.
The counts are read from each run's `train.log` (`total_encoding_params`, `total_network_params`). Coarse
levels are dense grids smaller than `2^T`, so `L × 2^T × F` would overcount for T ≥ 14.

## Upstream and our changes

- Upstream: [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) @ `018dea6`
- `patches/instant-ngp.patch` changes `scripts/run.py`:
  - The marching-cubes threshold default is `None` instead of `2.5`, so an explicit `0` (the SDF
    zero level set) is honoured. Upstream's `args.x or 2.5` turned `0` into `2.5`.
  - Adds optional `--save_slices` / `--slice_res` for SDF slice images (diagnostic, not used for
    results).
- `overlays/instant-ngp/` (our scripts):
  - `run_param_search.sh`: the sweep driver.
  - `ngp_helpers.py`: config writing, MLP parameter counting via `pyngp.Testbed`.
  - `time_inference.py`: timing from `train.log` timestamps.

## Environment

```bash
conda env create -f envs/instant-ngp.yml && conda activate instant-ngp
```

You also need the CUDA toolkit, CMake and a C++ compiler to build instant-ngp; see the upstream
README for exact versions. Default build options are used. tiny-cuda-nn builds with `TCNN_HALF_PRECISION=ON`, which is why the payload
is charged in fp16.

```bash
cd Projects/instant-ngp
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
```

This produces `build/pyngp*.so`. The sweep script adds `build/` to `PYTHONPATH` itself.

## Checkpoints / downloads

None. Every mesh is fitted from scratch.

## Run

```bash
cd Projects/instant-ngp
./run_param_search.sh
```

For each T and each mesh in `data/shapenet/val.lst`, the script:
1. writes `configs/sdf/base.json` with `log2_hashmap_size = T` (16 levels, 2 features/level, base
   resolution 16; 2×64 FullyFusedMLP; MAPE loss, Adam lr 1e-4 with EMA and exponential decay),
2. trains on the mesh's SDF for `N_STEPS` = 10,000 steps (`scripts/run.py`),
3. extracts the zero level set with marching cubes at resolution `MC_RES` = 128, threshold 0.

Overrides (environment variables): `LOG2_VALUES`, `N_STEPS`, `MC_RES`, `MC_THRESH`, `VAL_LST`,
`SHAPENET_DIR`, `OUT_DIR`, and `SKIP_EXISTING=1` to resume an interrupted sweep.

**Output:**
```
parameter_sweep_results/log2_<T>/03001627/<hash>/
  reconstructed_mesh.obj   # scored by metrics/distance.py
  train.log                # also the source of the payload count
  network.json, info.txt, original_mesh.obj
```

**Timing** (only for marker sizes with `--time-markers`):
```bash
python time_inference.py     # -> parameter_sweep_results/inference_times.csv
```

## Score

```bash
cd ../.. && conda activate distance
python metrics/distance.py --project ngp      # -> distances/ngp/log2_<T>_distance.txt
```

## Known behaviour

- **7 of 100 meshes fail at every T** (SDF training diverges on degenerate geometry):
  `374bec02e71fe06528b4c5ec471dc963`, `4011b7d293cbeae11890f3b4ec900fa`, `5bfee410a492af4f65ba78ad9601cf1b`,
  `730fc3ae87640d952c528d33bca1ac2`, `98cb91189f35e5f57c311bdc1462c34a`, `cd5ad4afabaed0d3e762624dc3c8fa2a`,
  `e9f36f74e22ca0897298e452825d1f0c`.
- **At T ≤ 7 most meshes have no usable reconstruction**, so those settings are averaged over
  11–72 meshes instead of 93.
