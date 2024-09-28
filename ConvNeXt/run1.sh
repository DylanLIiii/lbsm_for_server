EXPERIMENT_NAME=$1
DATA_PATH="/home/couser/imagenet/data"
OUTPUT_DIR="/home/couser/lbsm_results"
NUM_GPUS=4
TRAIN_FUNC=$2

mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model convnext_tiny --drop-path 0.1 \
--batch-size 512 --lr 4e-3 --update-freq 2 \
--model-ema true --model-ema-eval true \
--data-path "${DATA_PATH}" \
--output-dir "${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
--use-amp True \
| tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"