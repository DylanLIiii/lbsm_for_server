EXPERIMENT_NAME=$1
DATA_PATH="/home/couser/imagenet/data"
OUTPUT_DIR="/home/couser/lbsm_results"
NUM_GPUS=4
mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346  main.py \
    --cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
    --data-path "${DATA_PATH}" \
    --batch-size 256 \
    --tag "${EXPERIMENT_NAME}" \
    --output "${OUTPUT_DIR}" \
    --fused_window_process \
    --smoothing 0.0 \
    | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"