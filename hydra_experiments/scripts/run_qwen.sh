#!/bin/bash
#SBATCH --job-name=qwen_exp
#SBATCH --partition=gpu-5h
#SBATCH --gres=gpu:h100:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/qwen-%j.out

BASE_DIR="/home/luloduarte/vision_experiment"
DATASET_DIR="/home/space/datasets-sqfs/luloduarte_images"
MODEL_SQFS="$BASE_DIR/models/Qwen2.5-VL-72B-AWQ.sqfs"
CONTAINER="$BASE_DIR/containers/vllm.sif"
SCRIPT="/scripts/experiment_qwen.py"

SERVED_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
PORT=8000

unset CUDA_VISIBLE_DEVICES
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0
export APPTAINERENV_PYTHONPATH="/scripts"
export PORT="$PORT"
export MODEL_NAME_OVERRIDE=$SERVED_NAME

echo "1. Starting vLLM Server..."

apptainer exec --nv \
    -B "$MODEL_SQFS":/model:image-src=/ \
    -B $HOME:$HOME \
    $CONTAINER \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /model \
    --served-model-name $SERVED_NAME \
    --tensor-parallel-size 1 \
    --quantization awq_marlin \
    --dtype float16 \
    --max-model-len 8192 \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --enforce-eager \
    --max-num-seqs 32 \
    --limit-mm-per-prompt '{"image": 1}' &

SERVER_PID=$!

echo "Waiting for server..."
while ! curl -s localhost:$PORT/v1/models > /dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "CRITICAL: vLLM Server died unexpectedly!"
        exit 1
    fi
    sleep 5
done
echo "Server Ready!"

echo "2. Running Experiment (Arguments: $@)..."

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
    -B $HOME:$HOME \
    $CONTAINER \
    python3 $SCRIPT "$@"

kill $SERVER_PID
