#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt
ROOT=work/rerun_c0_c0/stage1_feature_knn_b7_ft1pct
SRC=work/rerun_c0/stage1_feature_knn_b7_s224
for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  mkdir -p "$ROOT/$mode"
  for split in validation test; do
    ln -sfn "$(pwd)/$SRC/$mode/${split}_pool0_stage1" "$ROOT/$mode/${split}_pool0_stage1"
  done
done
echo "===== target validation ====="
"$PY" scripts/eval_route_propagation_quality.py --checkpoint "$CKPT" --mode anchor_conditioned_target_pooling --root "$ROOT" --split validation --canvas 512 --resume
echo "===== patch validation ====="
"$PY" scripts/eval_route_propagation_quality.py --checkpoint "$CKPT" --mode anchor_conditioned_patch_correspondence --root "$ROOT" --split validation --canvas 512 --resume
echo "===== patch test ====="
"$PY" scripts/eval_route_propagation_quality.py --checkpoint "$CKPT" --mode anchor_conditioned_patch_correspondence --root "$ROOT" --split test --canvas 512 --resume
echo "C0_REMAINING_GPU0_DONE"
