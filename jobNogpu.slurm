#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
ulimit -n 131072
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores

# ===== PATH PROGETTO =====
cd /leonardo/home/userexternal/cgiacche/unicc/scale-mae
#mkdir -p logs output_dirVEG/logs

# ===== SCRATCH / TMP / W&B =====
SCR_BASE="${CINECA_SCRATCH:-${SCRATCH:-/leonardo_scratch/$USER}}"
RUN_DIR="$SCR_BASE/scale-mae/runs/${SLURM_JOB_ID}"
OUTPUT_DIR="$RUN_DIR/output"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$LOG_DIR"

export TMPDIR="$SCR_BASE/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

export WANDB_MODE=offline
export WANDB_DIR="$SCR_BASE/wandb"
export WANDB_CACHE_DIR="$SCR_BASE/.cache/wandb"
export WANDB_CONFIG_DIR="$SCR_BASE/.config/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# ===== Stabilità memoria (host e device) =====
export MALLOC_ARENA_MAX=2                  # riduce l'uso di heap glibc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== NCCL / Torch =====
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ===== RENDEZVOUS MULTI-NODO =====
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}

# GPU per nodo (sanitizza eventuali formati "4(A100-...)")
GPUS_PER_NODE_RAW="${SLURM_GPUS_ON_NODE:-4}"
GPUS_PER_NODE=$(echo "$GPUS_PER_NODE_RAW" | awk -F'[( ]' '{print $1}')
NNODES="${SLURM_NNODES:-1}"

# ===== LANCIO =====
srun --ntasks-per-node=1 bash -lc "
  source myenv/bin/activate
  python mae/main_pretrain.py \
      --config mae/config/veg.yaml \
      --eval_train_fnames /leonardo_scratch/fast/IscrC_UMC/NWPU-RESISC45/NWPU-RESISC45 \
      --eval_val_fnames   /leonardo_scratch/fast/IscrC_UMC/NWPU-RESISC45/NWPU-RESISC45 \
      --output_dir ${OUTPUT_DIR} \
      --log_dir    ${LOG_DIR} \
      --dataset_only \
      --device cpu \
      --num_workers 0 \
"
