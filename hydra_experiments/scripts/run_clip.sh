#!/bin/bash
#SBATCH --job-name=clip_zero
#SBATCH --partition=gpu-2h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/clip-%j.out

BASE_DIR="/home/luloduarte/vision_experiment"
DATASET_DIR="/home/space/datasets-sqfs/luloduarte_images"
CONTAINER="$BASE_DIR/containers/vllm.sif"
SCRIPT="/scripts/experiment_clip.py"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0
export APPTAINERENV_PYTHONPATH="/scripts"

echo "Starting CLIP Zero-Shot on the Full Set..."

apptainer exec --nv \
    -B "$BASE_DIR/data":/data \
    -B "$BASE_DIR/models":/models \
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
    -B $HOME:$HOME \
    $CONTAINER \
    python3 $SCRIPT

echo "Done."
