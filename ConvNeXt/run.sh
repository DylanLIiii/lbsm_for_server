EXPERIMENT_NAME=$1
DATA_PATH="/home/couser/imagenet/data"
OUTPUT_DIR="/home/couser/lbsm_results"
NUM_GPUS=4
TRAIN_FUNC=$2


python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --model convnext_tiny --drop_path 0.1 \
  --batch_size 128 --lr 4e-3 --update_freq 8 \
  --model_ema true --model_ema_eval true \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
  | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"