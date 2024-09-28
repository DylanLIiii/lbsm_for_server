# Set variables
EXPERIMENT_NAME1=$1
DATA_PATH="/home/couser/imagenet/data"
OUTPUT_DIR="/home/couser/lbsm_results"
NUM_GPUS=4
TRAIN_FUNC=$2

mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"

torchrun --standalone --nnodes=1 --nproc_per_node=4
  --model resnet50 \
  --amp \
  --data-path "${DATA_PATH}" \
  --workers 12 \
  --batch-size 256 \
  --epochs 90 \
  --opt sgd \
  --momentum 0.9 \
  --lr 0.1 \
  --lr-scheduler steplr \
  --lr-step-size 30 \
  --lr-gamma 0.1 \
  --weight-decay 1e-4 \
  --interpolation bilinear \
  --val-resize-size 256 \
  --val-crop-size 224 \
  --train-crop-size 224 \
  --print-freq 100 \
  --output-dir "${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
  --seed 0 \
  --label-smoothing 0.0 \
  --train-func $TRAIN_FUNC \
  | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"