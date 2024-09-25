#!/bin/bash

# Set variables
EXPERIMENT_NAME1=$1
TRAIN_FUNC=$2
MODEL="deit_small_patch16_224"
BATCH_SIZE=256
DATA_PATH="/home/couser/imagenet/data"
OUTPUT_DIR="/home/couser/lbsm_results"
NUM_GPUS=4
IS_RESUME=$3  # New variable to store the resume path

mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME1}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME1}/outputs.txt"

# Prepare resume argument
RESUME_ARG=""
if [ -n "$IS_RESUME" ]; then
    echo "Resume training from ${OUTPUT_DIR}/${EXPERIMENT_NAME1}/checkpoint.pth"
    RESUME_ARG="--resume ${OUTPUT_DIR}/${EXPERIMENT_NAME1}/checkpoint.pth"
fi

# Run the distributed training
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port 29501 \
    --use_env \
    main.py \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --data-path $DATA_PATH \
    --output_dir "${OUTPUT_DIR}/${EXPERIMENT_NAME1}"\
    --smoothing 0.0 \
    --seed 0 \
    --dist-eval \
    $RESUME_ARG \
    --train-func $TRAIN_FUNC \
    | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME1}/outputs.txt"