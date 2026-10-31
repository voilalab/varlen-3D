# VarLen — Variable-Length 3D Shape Encoding

This repository contains the training and evaluation code for **VarLen-3D**, a variable-length shape tokenizer that adapts a frozen base Point-clout to SDF autoencoder with a HyperAutoEncoder (HAE). The HAE learns to produce an ordered sequence of latents such that any prefix of length *L* reconstructs the shape at a fidelity that increases with *L* to achieve a continuous rate-distortion tradeoff with no retraining of the base model.

## Architecture overview

```
Input point cloud
       │
       ▼
 Base autoencoder (frozen)
  ├── Encoder  →  shape latent  (1024 × 32)
  └── Decoder  ←  shape latent
       │
       ▼
 HyperAutoEncoder (trained)
  ├── HyperEncoder  →  hyper-tokens  (1024 × 32)  [causal self-attention + RoPE]
  └── HyperDecoder  ←  prefix [:L]                [sinusoidal PE queries]
       │
       ▼
 Reconstructed shape latent  →  Base decoder  →  SDF volume  →  Mesh
```

Training is done in **two stages**:

| Stage | Loss | Purpose |
|-------|------|---------|
| 1 — Latent supervision | MSE between reconstructed and ground-truth latents | Fast convergence, stable ordering |
| 2 — SDF finetuning | Stage-1 loss + SDF regression (vol + near-surface + Eikonal + surface) | Signal-space quality |

---

## 1. Environment setup

We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda create -n varlen python=3.11 -y
conda activate varlen

# PyTorch with CUDA 12.4 (adjust if your CUDA version differs)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Flash-Attention (requires a compatible CUDA compiler)
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Remaining dependencies
pip install einops==0.8.2 numpy==2.3.5 scipy==1.17.1 \
    matplotlib==3.10.8 trimesh==4.11.3 pymcubes==0.1.6 \
    tensorboard==2.20.0 tqdm==4.67.3

# Optional: Weights & Biases logging
pip install wandb==0.26.0
```

Alternatively, create the environment from `environment.yaml` (after installing PyTorch and flash-attn manually as above, since they require special index URLs):

```bash
conda env create -f environment.yaml
```

---

## 2. Data preparation

Each shape is stored as a `.npz` file containing:

| Key | Shape | Description |
|-----|-------|-------------|
| `surface_points` | (N, 3) | Surface point cloud samples |
| `vol_points` | (M, 3) | Volumetric query points |
| `vol_sdf` | (M,) | SDF values at volumetric points |
| `near_points` | (K, 3) | Near-surface query points |
| `near_sdf` | (K,) | SDF values at near-surface points |

We use the **chairs** subset of [ShapeNet](https://shapenet.org/). Place all `.npz` files in a single flat directory (e.g., `./data/`). The train/test split (6100 / 678 shapes) is provided in `data/train.txt` and `data/test.txt`.

> Data preprocessing follows [this repository](https://github.com/1zb/sdf_gen) using the sphere normalization scheme. 

---

## 3. Base model checkpoint

The HAE operates on top of a pretrained base shape autoencoder (`learnable_vec1024x32_dim1024_depth24_nb`). Download the base model checkpoint and place it at:

```
checkpoints/base_model.pth
```

> We use the base checkpoint provided by the [VecSetX](https://github.com/1zb/VecSetX/) repository. Specifically, we use [this checkpoint file](https://huggingface.co/Zbalpha/VecSetX/blob/main/learnable_vec1024x32_dim1024_depth24_sdf_nb/checkpoint-125.pth) that uses learnable queries.

---

## 4. Training

### Stage 1 — Latent supervision only

Train the HAE to reconstruct shape latents from a prefix of tokens. The SDF branch is disabled (`--sdf_loss_weight 0`).

```bash
torchrun --nproc_per_node=4 main_hae.py \
    --base_pth checkpoints/base_model.pth \
    --data_path ./data/ \
    --output_dir ./output/hae_stage1/ \
    --working_dim 512 \
    --encoder_depth 8 \
    --decoder_depth 8 \
    --epochs 400 \
    --batch_size 6 \
    --blr 5e-5 \
    --sdf_loss_weight 0 \
    --latent_loss_weight 1.0 \
    --min_prefix 64
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--working_dim` | 512 | Internal attention dimension of the HAE |
| `--encoder_depth` | 8 | Self-attention layers in the HyperEncoder |
| `--decoder_depth` | 8 | Self-attention layers in the HyperDecoder |
| `--min_prefix` | 64 | Minimum prefix length *L* sampled during training |
| `--sdf_loss_weight` | 1.0 | Set to `0` for latent-only stage |
| `--bias_small_prefix` | off | Sample *L* with exponential bias toward small values |

### Stage 2 — SDF finetuning

Resume from Stage 1 and add SDF regression losses. Using `--reset_optimizer` starts the optimizer fresh, which is recommended when changing loss weights.

```bash
torchrun --nproc_per_node=4 main_hae.py \
    --base_pth checkpoints/base_model.pth \
    --data_path ./data/ \
    --output_dir ./output/hae_stage2/ \
    --working_dim 512 \
    --encoder_depth 8 \
    --decoder_depth 8 \
    --epochs 100 \
    --batch_size 6 \
    --blr 5e-5 \
    --sdf_loss_weight 1.0 \
    --latent_loss_weight 1.0 \
    --min_prefix 64 \
    --resume ./output/hae_stage1/checkpoint-399.pth \
    --reset_optimizer \
    --start_epoch 0
```

Checkpoints are saved every 5 epochs to `output_dir/checkpoint-{epoch}.pth`.

---

## 5. Evaluation

Evaluate a trained HAE on the test set, computing Chamfer distance at each prefix length (powers of 2 from 1 to 1024):

```bash
python eval_hae_rd.py \
    --base_pth checkpoints/base_model.pth \
    --hae_pth ./output/hae_stage2/checkpoint.pth \
    --data_path ./data/ \
    --working_dim 512 \
    --encoder_depth 8 \
    --decoder_depth 8 \
    --n_shapes 100 \
    --resolution 64 \
    --output_dir ./eval_out/
```

Outputs: `eval_out/rd_curve.png` (RD curve plot) and `eval_out/results.csv` (per-shape, per-prefix Chamfer distances).

You can download our pre-trained checkpoint [here](https://utexas.box.com/s/pnjhb14gx58o55lg5acyu92b33v887n9).

---

## 6. Inference

### HAE (rate-distortion curve for a single shape)

Reconstruct a single shape at every prefix length and plot the RD curve:

```bash
python infer_hae.py \
    --base_pth checkpoints/base_model.pth \
    --hae_pth ./output/hae_stage2/checkpoint-99.pth \
    --input path/to/shape.obj \
    --output_dir ./infer_out/ \
    --working_dim 512 \
    --encoder_depth 8 \
    --decoder_depth 8 \
    --resolution 128
```

Outputs: `infer_out/baseline.obj`, `infer_out/L{L:04d}.obj` for each prefix length, `infer_out/rd_curve.png`, and `infer_out/results.csv`.

---
}
```
