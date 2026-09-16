#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
# wait for the previous GPU1 target val/test job to finish
while [ ! -f work/rerun_c0/TARGET_VALTEST_PROPAGATION_DONE ]; do
  sleep 20
done
export CUDA_VISIBLE_DEVICES=1
PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt
ROOT=work/rerun_c0/stage1_feature_knn_b7_ft1pct
mkdir -p "$ROOT/anchor_conditioned_patch_correspondence"
for split in train validation test; do
  src=work/rerun_c0/stage1_feature_knn_b7/anchor_conditioned_patch_correspondence/${split}_pool0_stage1
  dst="$ROOT/anchor_conditioned_patch_correspondence/${split}_pool0_stage1"
  if [ ! -e "$dst" ]; then ln -s "$(pwd)/$src" "$dst"; fi
done
for split in train validation test; do
  echo "===== propagation patch_correspondence $split ====="
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CKPT" --mode anchor_conditioned_patch_correspondence \
    --root "$ROOT" --split "$split" --canvas 512 --resume
done
echo "PATCH_PROPAGATION_DONE"
