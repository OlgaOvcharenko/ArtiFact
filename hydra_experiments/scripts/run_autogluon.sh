#!/bin/bash
#SBATCH --job-name=autogluon
#SBATCH --partition=gpu-2d
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100"
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/autogluon-%j.out

BASE_DIR="/home/luloduarte/vision_experiment"
DATASET_DIR="/home/space/datasets-sqfs/luloduarte_images"
CONTAINER="$BASE_DIR/containers/vllm.sif"
SCRIPT="/scripts/experiment_autogluon.py"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export APPTAINERENV_PYTHONPATH="/scripts"

export APPTAINERENV_PYTHONPYCACHEPREFIX="/tmp/pycache_${SLURM_JOB_ID}"
mkdir -p /tmp/pycache_${SLURM_JOB_ID}
export APPTAINERENV_PIP_CACHE_DIR="/tmp/pip_cache_${SLURM_JOB_ID}"
mkdir -p /tmp/pip_cache_${SLURM_JOB_ID}
export APPTAINERENV_HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
mkdir -p /tmp/hf_cache_${SLURM_JOB_ID}

# hide slurm variables from pytorch lightning to prevent multi-node hang
export APPTAINERENV_SLURM_JOB_NAME="autogluon"
export APPTAINERENV_SLURM_NTASKS=1
export APPTAINERENV_SLURM_PROCID=0
export APPTAINERENV_SLURM_LOCALID=0
export SLURM_NTASKS=1
export SLURM_PROCID=0
export SLURM_LOCALID=0

ulimit -n 65536

echo "Starting AutoGluon + CleanLab Experiment..."


JOB_ID=${SLURM_JOB_ID:-"local"}
LIBS_DIR="/tmp/ag_libs_${JOB_ID}"
mkdir -p $LIBS_DIR

echo "1. Installing AutoGluon..."
apptainer exec \
    -B $LIBS_DIR:/libs \
    $CONTAINER \
    pip install --target=/libs autogluon.multimodal cleanlab pandas scikit-learn


export APPTAINERENV_PYTHONPATH="/libs:/scripts"

SEED_ARG=""
if [[ "$*" != *"--seed"* ]]; then
    if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        SEED_ARG="--seed $SLURM_ARRAY_TASK_ID"
        echo "Auto-assigning seed from SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    fi
fi

echo "2. Running Experiment (Arguments: $@ $SEED_ARG)..."
apptainer exec --nv \
    -B "$BASE_DIR/data":/data \
    -B "$BASE_DIR/scripts":/scripts \
    -B "$BASE_DIR/results":/results \
    -B "$DATASET_DIR/images_part_00.sqfs":/dataset/chunk_00:image-src=/ \
    -B "$DATASET_DIR/images_part_01.sqfs":/dataset/chunk_01:image-src=/ \
    -B "$DATASET_DIR/images_part_02.sqfs":/dataset/chunk_02:image-src=/ \
    -B "$DATASET_DIR/images_part_03.sqfs":/dataset/chunk_03:image-src=/ \
    -B "$DATASET_DIR/images_part_04.sqfs":/dataset/chunk_04:image-src=/ \
    -B "$DATASET_DIR/images_part_05.sqfs":/dataset/chunk_05:image-src=/ \
    -B "$DATASET_DIR/images_part_06.sqfs":/dataset/chunk_06:image-src=/ \
    -B "$DATASET_DIR/images_part_07.sqfs":/dataset/chunk_07:image-src=/ \
    -B "$DATASET_DIR/images_part_08.sqfs":/dataset/chunk_08:image-src=/ \
    -B "$DATASET_DIR/images_part_09.sqfs":/dataset/chunk_09:image-src=/ \
    -B "$DATASET_DIR/images_part_10.sqfs":/dataset/chunk_10:image-src=/ \
    -B "$DATASET_DIR/images_part_11.sqfs":/dataset/chunk_11:image-src=/ \
    -B "$DATASET_DIR/images_part_12.sqfs":/dataset/chunk_12:image-src=/ \
    -B "$DATASET_DIR/images_part_13.sqfs":/dataset/chunk_13:image-src=/ \
    -B "$DATASET_DIR/images_part_AIC.sqfs":/dataset/chunk_AIC:image-src=/ \
    -B $LIBS_DIR:/libs \
    -B $HOME:$HOME \
    $CONTAINER \
    python3 $SCRIPT "$@" $SEED_ARG

echo "Done."
rm -rf $LIBS_DIR
