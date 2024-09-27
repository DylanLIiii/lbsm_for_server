# Set variables
EXPERIMENT_NAME1=$1
DATA_PATH="/home/couser/imagenet/data"
OUTPUT_DIR="/home/couser/lbsm_results"
NUM_GPUS=4
TRAIN_FUNC=$2

mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"

CUDA_VISIBLE_DEVICES=3 python train.py \
  --model resnet50 \
  --cache-dataset \
  --amp \
  --data-path "${DATA_PATH}" \
  --workers 12 \
  --batch-size 1024 \
  --epochs 90 \
  --opt sgd \
  --momentum 0.9 \
  --lr 0.5 \
  --lr-scheduler cosineannealinglr \
  --lr-warmup-epochs 5 \
  --lr-warmup-method linear \
  --lr-warmup-decay 0.01 \
  --weight-decay 1e-4 \
  --interpolation bilinear \
  --val-resize-size 256 \
  --val-crop-size 224 \
  --train-crop-size 224 \
  --print-freq 10 \
  --output-dir "${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
  --seed 0 \
  --label-smoothing 0.0 \
  --train-func $TRAIN_FUNC \
  | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"