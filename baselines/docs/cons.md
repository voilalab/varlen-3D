# CONs (Convolutional Occupancy Networks)

## What it is

[Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks)
encode a point cloud into a 3D feature grid with a PointNet plus a 3D U-Net, then decode occupancy at
query points with a small MLP. One model is trained on the 1000 training chairs and then applied to the
100 val chairs. The configuration is `configs/pointcloud/shapenet_grid32.yaml`: a 32³ feature grid.

**Capacity knob:** feature dimension `c_dim` ∈ {4, 8, 12, 16, 20, 24, 28, 32}. Changing `c_dim`
alone barely changes the parameter count, so the sweep scales the encoder hidden dimension, the U-Net
channels (`f_maps`, `in_channels`, `out_channels`) and the decoder width together with `c_dim`.

**Payload charged per mesh:** the per-shape feature grid, `32³ × c_dim` fp32 values. The
encoder, U-Net and decoder are shared across shapes, so they are free.

## Upstream and our changes

- Upstream: [autonomousvision/convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks) @ `838bea5`
- `patches/convolutional_occupancy_networks.patch` ports the 2020 code to Python 3.10 / PyTorch 2.1:
  - `environment.yaml`, `setup.py`: modern pins. `pykdtree` comes from pip instead of being built
    locally, and Cython runs with language level 3.
  - `torch.gesv` → `torch.linalg.solve`; legacy `torch.FloatTensor(...)` constructors →
    `torch.tensor(...)`; `yaml.load` → `yaml.safe_load` (`src/common.py`, `src/config.py`,
    `src/conv_onet/generation.py`, `src/data/core.py`, `src/encoder/*`, `src/utils/*`).
  - `src/data/core.py`: accepts split lists in our `03001627/<hash>` format, and gives a clear error
    when a batch is empty because preprocessed files are missing.
  - `src/encoder/unet3d.py`: GroupNorm uses 4 groups instead of 8, so the smallest `c_dim` = 4 is
    still divisible.
  - `scripts/download_data.sh`: downloads the Occupancy Networks ShapeNet data (below), optionally
    for a single category.
  - `train.py`: also prints the per-mesh (non-shared) latent size next to the total parameter count.
- `overlays/convolutional_occupancy_networks/` (our scripts):
  - `sweep.py`: the sweep driver.
  - `src/param_count.py`: payload counting.
  - `time_inference.py`: decode timing.
  - `scripts/preprocess_shapenet_obj_to_npz.py`: builds the same training files from raw OBJs. It
    was not used for the reported runs.

## Environment

```bash
conda env create -f envs/conv_onet.yml && conda activate conv_onet
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
cd Projects/convolutional_occupancy_networks && python setup.py build_ext --inplace
```

## Data / downloads

CONs trains on the **preprocessed ShapeNet release from Occupancy Networks** (per shape: `points.npz`
with occupancy labels and `pointcloud.npz` with surface points and normals), not on the raw OBJs.

```bash
cd Projects/convolutional_occupancy_networks/data
bash ../scripts/download_data.sh 03001627     # downloads dataset_small_v1.1.zip, extracts ShapeNet/03001627
```

Then replace the release's split lists with the benchmark splits. Our runs used `test.lst` (the
generation split) and `val.lst` both equal to the benchmark val list:

```bash
S=../../../data/shapenet
cp $S/train.lst ShapeNet/03001627/train.lst
cp $S/val.lst   ShapeNet/03001627/val.lst
cp $S/val.lst   ShapeNet/03001627/test.lst
```

## Run

```bash
cd Projects/convolutional_occupancy_networks
python sweep.py --gt-mesh-dir ../../data/shapenet
```

For each `c_dim` (default order 32 → 4), the driver:
1. writes `parameter_sweep_results/cdim_<c>/config.yaml` (from `default.yaml` +
   `pointcloud/shapenet_grid32.yaml` with the scaled dimensions; batch size 32),
2. trains with `train.py` and keeps `model_best.pt`,
3. generates meshes for `test.lst` with `generate.py`,
4. writes `results.yaml`, with an internal ratio that uses a different denominator and is not the
   one plotted.

**Training length:** `train.py` exits through its `--exit-after` timer, and the driver additionally
stops each run after **2 hours of wall-clock time**. So the number of iterations depends on GPU
speed and model size. Our runs reached 40k–90k iterations: larger `c_dim` trained fewer iterations
(c_dim 32: 40k; c_dim 4–16: 70k–90k). Checkpoints are validated every 10k iterations on
`val.lst`, and `model_best.pt` is the one with the highest IoU. `--max-iters` sets the nominal budget.

**Output** (what `metrics/distance.py` reads):
```
parameter_sweep_results/cdim_<c>/training/generation/meshes/03001627/<hash>.off
```

**Timing** (for `--time-markers` only):
```bash
python time_inference.py      # -> parameter_sweep_results/inference_times.csv
```

## Score

```bash
cd ../.. && conda activate distance
python metrics/distance.py --project cons     # -> distances/CONs/cdim_<c>_distance.txt
```

## Expected results (committed `distances/CONs/`)

| c_dim | ratio | meshes scored | median CD | mean CD |
|---|---|---|---|---|
| 4 | 2.8418 | 100 | 6.237e-02 | 8.237e-02 |
| 8 | 5.6835 | 100 | 6.325e-02 | 8.248e-02 |
| 12 | 8.5253 | 100 | 6.248e-02 | 8.084e-02 |
| 16 | 11.3671 | 100 | 6.310e-02 | 8.221e-02 |
| 20 | 14.2088 | 100 | 6.269e-02 | 8.103e-02 |
| 24 | 17.0506 | 100 | 6.366e-02 | 8.314e-02 |
| 28 | 19.8924 | 100 | 6.255e-02 | 8.091e-02 |
| 32 | 22.7341 | 100 | 6.286e-02 | 8.018e-02 |

All 100 val meshes are generated for every `c_dim`. The committed Chamfer values come from an
earlier version of `metrics/distance.py` that sampled points without a seed; re-scoring gives values
within sampling noise.
