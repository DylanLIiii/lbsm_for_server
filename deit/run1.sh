#!/bin/bash

# Set variables
EXPERIMENT_NAME1=$1
TRAIN_FUNC=$2
MODEL="deit_small_patch16_224"
BATCH_SIZE=12
DATA_PATH="/home/couser/imagenet/data"
OUTPUT_DIR="/home/couser/lbsm_results"
NUM_GPUS=4

mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME1}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME1}/outputs.txt"

# Run the distributed training
python CUDA_VISIBLE_DEVICES=4,5,6,7 -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --use_env \
    main.py \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --data-path $DATA_PATH \
    --output_dir "${OUTPUT_DIR}/${EXPERIMENT_NAME1}"\
    --smoothing 0.0 \
    --seed 0 \
    --train-func $TRAIN_FUNC \
    | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME1}/outputs.txt"