EXPERIMENT_NAME=$1
DATA_PATH="/home/couser/imagenet/data"
OUTPUT_DIR="/home/couser/lbsm_results"
NUM_GPUS=4
mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
    --cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
    --data-path "${DATA_PATH}" \
    --batch-size 256 \
    --tag "${EXPERIMENT_NAME}" \
    --output "${OUTPUT_DIR}" \
    | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"