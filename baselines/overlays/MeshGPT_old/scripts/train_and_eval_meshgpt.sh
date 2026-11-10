#!/usr/bin/env bash
#
# Run the MeshGPT vocabulary over the val split (the name is historical).
#
# With patches/MeshGPT_old.patch applied, config/meshgpt.yaml sets eval_only=true and
# resume=pretrained/vocabulary/checkpoints/749-0.ckpt, so this does NOT train: it runs
# one validation pass that reconstructs every val mesh to runs/vocabulary/mesh_val/.
# (It also writes an in-trainer Chamfer file to <repo>/distances/MeshGPT_orig/; that
# file uses a different metric and is not used. Score with metrics/distance.py.)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_and_eval_meshgpt.sh
#   RESUME=path/to.ckpt ...   # evaluate a different checkpoint
#
set -euo pipefail

MESHGPT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MESHGPT_ROOT"
export PYTHONPATH="$MESHGPT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# GPU setup: use free GPUs; the trainer auto-enables DDP when >1 GPU is visible
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

MAX_EPOCH="${MAX_EPOCH:-750}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RESUME="${RESUME:-}"

echo "=============================================="
echo "  MeshGPT: Train + Validate (Chamfer vs GT)"
echo "=============================================="
echo "Root:       $MESHGPT_ROOT"
echo "Max epoch:  $MAX_EPOCH"
echo "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Resume:     ${RESUME:-none}"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo ""

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    echo ">>> Using checkpoint $RESUME"
    RESUME_ARG="resume=$RESUME"
else
    echo ">>> Using the checkpoint set in config/meshgpt.yaml (resume=...)"
fi

python trainer/train_vocabulary.py \
    experiment=vocabulary \
    only_chairs=True \
    max_epoch="$MAX_EPOCH" \
    batch_size="$BATCH_SIZE" \
    gradient_accumulation_steps="$GRAD_ACCUM" \
    wandb_main=False \
    sanity_steps=0 \
    num_val_samples=16 \
    $RESUME_ARG

echo ""
echo ">>> Done."
