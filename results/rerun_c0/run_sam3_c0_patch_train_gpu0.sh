#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt
ROOT=work/rerun_c0_c0/stage1_feature_knn_b7_ft1pct
SRC=work/rerun_c0/stage1_feature_knn_b7_s224
mkdir -p "$ROOT/anchor_conditioned_patch_correspondence"
ln -sfn "$(pwd)/$SRC/anchor_conditioned_patch_correspondence/train_pool0_stage1" "$ROOT/anchor_conditioned_patch_correspondence/train_pool0_stage1"
"$PY" scripts/eval_route_propagation_quality.py \
  --checkpoint "$CKPT" --mode anchor_conditioned_patch_correspondence \
  --root "$ROOT" --split train --canvas 512 --resume
echo "C0_PATCH_TRAIN_DONE"
