#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 3
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores

cd /leonardo/home/userexternal/cgiacche/unicc/scale-mae
mkdir -p logs

# === percorso condiviso tra i job della catena ===
SCR_BASE="${CINECA_SCRATCH:-${SCRATCH:-/leonardo_scratch/$USER}}"
EXP_NAME="veg_teacher_scratch_v1"                         # <— cambialo tu
OUT_BASE="$SCR_BASE/scale-mae/runs/$EXP_NAME"
OUTPUT_DIR="$OUT_BASE/output"                             # fisso
LOG_DIR="$OUT_BASE/logs/job_${SLURM_JOB_ID}"              # diverso per job
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$SCR_BASE/tmp/$SLURM_JOB_ID" "$SCR_BASE/.cache/torch"

export TMPDIR="$SCR_BASE/tmp/$SLURM_JOB_ID"
export TORCH_HOME="$SCR_BASE/.cache/torch"
export WANDB_MODE=offline
export WANDB_DIR="$SCR_BASE/wandb"
export WANDB_CACHE_DIR="$SCR_BASE/.cache/wandb"
export WANDB_CONFIG_DIR="$SCR_BASE/.config/wandb"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONWARNINGS="ignore:::timm\.optim\.optim_factory:FutureWarning"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))
GPUS_PER_NODE=$(echo "${SLURM_GPUS_ON_NODE:-4}" | awk -F'[( ]' '{print $1}')
NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"

srun --nodes=${NNODES} --ntasks=${NNODES} --ntasks-per-node=1 --cpu-bind=cores --kill-on-bad-exit=1 \
bash -lc "
  source myenv/bin/activate
  torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id=${SLURM_JOB_ID} \
    mae/main_pretrain.py \
      --config mae/config/geo.yaml \
      --eval_train_fnames /leonardo_scratch/fast/IscrC_UMC/NWPU-RESISC45/NWPU-RESISC45 \
      --eval_val_fnames   /leonardo_scratch/fast/IscrC_UMC/NWPU-RESISC45/NWPU-RESISC45 \
      --output_dir $OUTPUT_DIR \
      --log_dir    $LOG_DIR \
      --num_workers \${DL_NUM_WORKERS:-2}
"
