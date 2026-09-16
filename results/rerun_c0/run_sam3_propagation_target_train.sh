#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt
ROOT=work/rerun_c0/stage1_feature_knn_b7_ft1pct
# link routes from generated KNN root if not already present
mkdir -p "$ROOT/anchor_conditioned_target_pooling"
for split in train validation test; do
  src=work/rerun_c0/stage1_feature_knn_b7/anchor_conditioned_target_pooling/${split}_pool0_stage1
  dst="$ROOT/anchor_conditioned_target_pooling/${split}_pool0_stage1"
  if [ ! -e "$dst" ] && [ -e "$src" ]; then
    ln -s "$(pwd)/$src" "$dst"
  fi
done
echo "===== propagation target_pooling train b3-b6 ====="
"$PY" scripts/eval_route_propagation_quality.py \
  --checkpoint "$CKPT" --mode anchor_conditioned_target_pooling \
  --root "$ROOT" --split train --canvas 512 --resume
echo "TARGET_TRAIN_PROPAGATION_DONE"
