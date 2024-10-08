# Set variables
EXPERIMENT_NAME1=$1
TRAIN_FUNC=$2
DATA_PATH="/datadrive/hengl/imagenet1k/ILSVRC/Data/CLS-LOC/"
OUTPUT_DIR="/datadrive2/hengl/lbsm_results"
NUM_GPUS=4
RESUME=$3
mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"

if [ "$RESUME" = "resume" ]; then
  RESUME="${OUTPUT_DIR}/${EXPERIMENT_NAME}/checkpoint.pth"
else
  RESUME=""
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 /home/hengl/lbsm_for_server/conv/train.py \
  --model resnet50 \
  --amp \
  --data-path "${DATA_PATH}" \
  --workers 12 \
  --batch-size 256 \
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
  --train-func "${TRAIN_FUNC}" \
  --resume "${RESUME}" \
  | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"