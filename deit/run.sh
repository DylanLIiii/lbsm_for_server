#!/bin/bash

# Set variables
EXPERIMENT_NAME1=$1
MODEL="deit_small_patch16_224"
BATCH_SIZE=256
DATA_PATH="/datadrive/mount/hengl/imagenet1k/ILSVRC/Data/CLS-LOC/"
OUTPUT_DIR="/datadrive/mount2/hengl/lbsm_results"
NUM_GPUS=4

mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME1}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME1}/outputs.txt"

# Run the distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --use_env \
    --master_port 29590 \
    main.py \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --data-path $DATA_PATH \
    --output_dir "${OUTPUT_DIR}/${EXPERIMENT_NAME1}"\
    --smoothing 0.0 \
    --seed 0 \
    | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME1}/outputs.txt"