EXPERIMENT_NAME=$1
OUTPUT_DIR='/datadrive/mount/hengl/lbsm_results'
DATA_DIR='/datadrive/mount/hengl/imagenet1k/ILSVRC/Data/CLS-LOC/'

mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
touch "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 vision/train.py \
  --model resnet50 \
  --data-path "${DATA_DIR}" \
  --workers 16 \
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
  --print-freq 500 \
  --output-dir "${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
  --save_frequency 100 \
  --seed 1 \
  | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}/outputs.txt"