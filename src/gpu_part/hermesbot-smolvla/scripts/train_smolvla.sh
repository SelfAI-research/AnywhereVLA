#!/usr/bin/env bash
set -euo pipefail

# ===== USER VARS =====
HF_USER="VorArt"
DATASET="hermesbot_dataset"

DATASET_DIR="$HOME/workspace/robotics/Hermes/hf_home/lerobot/${HF_USER}/${DATASET}"
OUT_DIR="$HOME/workspace/robotics/Hermes/hermesbot-smolvla/model/train/smolvla_hermesbot"

DEVICE="cuda"   # or "cpu"
BATCH_SIZE=8
STEPS=1000
JOB_NAME="smolvla_hermesbot"
WANDB="false"   # set "true" if you want W&B logging

mkdir -p "$OUT_DIR"

# ===== TRAINING CMD =====
CMD=(
  lerobot-train
  --policy.path=lerobot/smolvla_base
  --policy.repo_id="${HF_USER}/smolvla_hermesbot"
  --policy.device="$DEVICE"
  --dataset.repo_id="${HF_USER}/${DATASET}"
  --dataset.root="$DATASET_DIR"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --output_dir="$OUT_DIR"
  --job_name="$JOB_NAME"
  --wandb.enable="$WANDB"

  # Optimizer config
  --optimizer.type=adamw
  --optimizer.lr=0.0001
  --optimizer.weight_decay=1e-10
  --optimizer.grad_clip_norm=10.0

  # Scheduler config
  --scheduler.type=cosine_decay_with_warmup
  --scheduler.num_warmup_steps=1000
  --scheduler.num_decay_steps=20000
  --scheduler.peak_lr=0.0001
  --scheduler.decay_lr=2.5e-6
)

echo "[RUN] ${CMD[@]}"
"${CMD[@]}"
