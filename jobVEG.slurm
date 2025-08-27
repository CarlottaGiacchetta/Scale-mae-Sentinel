#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
# #SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --gpus-per-node=4          
#SBATCH --ntasks-per-node=1        
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=24:00:00
# #SBATCH -t 4-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
ulimit -n 131072
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores

# ===== PATH PROGETTO =====
cd /leonardo/home/userexternal/cgiacche/unicc/scale-mae
mkdir -p logs

# ===== SCRATCH / DIR DI RUN =====
SCR_BASE="${CINECA_SCRATCH:-${SCRATCH:-/leonardo_scratch/$USER}}"
RUN_DIR="$SCR_BASE/scale-maePROVAlogs/runs/${SLURM_JOB_ID}"
OUTPUT_DIR="$RUN_DIR/output"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# ===== TMP / CACHE / W&B =====
export TMPDIR="$SCR_BASE/tmp/$SLURM_JOB_ID"; mkdir -p "$TMPDIR"
export TORCH_HOME="$SCR_BASE/.cache/torch"; mkdir -p "$TORCH_HOME"

export WANDB_MODE=offline
export WANDB_DIR="$SCR_BASE/wandb"
export WANDB_CACHE_DIR="$SCR_BASE/.cache/wandb"
export WANDB_CONFIG_DIR="$SCR_BASE/.config/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# ===== Evita warning / imposta thread =====
export MALLOC_ARENA_MAX=2
unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

export PYTHONWARNINGS="ignore:::timm\.optim\.optim_factory:FutureWarning"

# ===== Dataloader =====
export DL_NUM_WORKERS=${DL_NUM_WORKERS:-2}

# ===== RENDEZVOUS MULTI-NODO =====
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR
# porta univoca per job (range 15000-34999)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))

# GPU per nodo (sanitizza eventuali stringhe tipo "4(A100...)")
GPUS_PER_NODE_RAW="${SLURM_GPUS_ON_NODE:-4}"
GPUS_PER_NODE=$(echo "$GPUS_PER_NODE_RAW" | awk -F'[( ]' '{print $1}')
NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"

echo "RUN_DIR: $RUN_DIR"
df -h "$RUN_DIR" || true
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT GPUS_PER_NODE=$GPUS_PER_NODE NNODES=$NNODES"

# ===== LANCIO: 1 task per nodo -> 1 torchrun per nodo -> 4 proc locali =====
srun \
  --nodes=${NNODES} \
  --ntasks=${NNODES} \
  --ntasks-per-node=1 \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
bash -lc "
    source myenv/bin/activate
    torchrun \
      --nnodes=${NNODES} \
      --nproc_per_node=${GPUS_PER_NODE} \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      --rdzv_id=${SLURM_JOB_ID} \
      mae/main_pretrain.py \
        --config mae/config/veg.yaml \
        --eval_train_fnames /leonardo_scratch/fast/IscrC_UMC/NWPU-RESISC45/NWPU-RESISC45 \
        --eval_val_fnames   /leonardo_scratch/fast/IscrC_UMC/NWPU-RESISC45/NWPU-RESISC45 \
        --output_dir $OUTPUT_DIR \
        --log_dir    $LOG_DIR \
        --num_workers ${DL_NUM_WORKERS} \
        --MAE_TIMINGS=1
"
