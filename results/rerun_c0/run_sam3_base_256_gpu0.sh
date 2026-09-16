#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
ROOT=work/rerun_c0_256_base/stage1_feature_knn_b7_base
SRC=work/rerun_c0/stage1_feature_knn_b7_s224
for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  mkdir -p "$ROOT/$mode"
  for split in train validation test; do
    ln -sfn "$(pwd)/$SRC/$mode/${split}_pool0_stage1" "$ROOT/$mode/${split}_pool0_stage1"
  done
done
for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  for split in train validation test; do
    echo "===== $mode $split base256 ====="
    "$PY" scripts/eval_route_propagation_quality.py \
      --checkpoint "$CKPT" --mode "$mode" --root "$ROOT" \
      --split "$split" --canvas 256 --resume
  done
done
echo "C0_256_BASE_ALL_PROPAGATION_DONE"
