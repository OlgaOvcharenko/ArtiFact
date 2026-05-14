#!/bin/bash
#SBATCH --job-name=prep_72b
#SBATCH --partition=cpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/prep_72b-%j.out

EXP_DIR="/home/luloduarte/vision_experiment"
TEMP_DIR="/tmp/model_download_72b"
FINAL_SQFS="$EXP_DIR/Qwen2.5-VL-72B-AWQ.sqfs"
CONTAINER_PATH="$EXP_DIR/vllm.sif"

rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

echo "1. Checking Container..."
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Container not found at $CONTAINER_PATH"
    exit 1
fi
echo "Container found."

echo "2. Starting Download (72B AWQ) to $TEMP_DIR..."
apptainer exec "$CONTAINER_PATH" huggingface-cli download \
    Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
    --local-dir "$TEMP_DIR" \
    --local-dir-use-symlinks False \
    --exclude "*.pth" "*.bin" "*.pt"

echo "3. Download successful. Squashing..."
if [ -z "$(ls -A $TEMP_DIR)" ]; then
   echo "Error: Download directory is empty!"
   exit 1
fi

squash-dataset "$TEMP_DIR" "$FINAL_SQFS"

echo "4. Done! Created $FINAL_SQFS"
ls -lh "$FINAL_SQFS"
