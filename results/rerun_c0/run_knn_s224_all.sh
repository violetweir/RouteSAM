#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=work/rerun_c0/stage1_feature_knn_b7_s224
PROTO=work/kvasir_1pct_anchors/protocol
for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  for split in train validation test; do
    if [ "$split" = train ]; then
      minb=3; maxb=6
    else
      minb=0; maxb=6
    fi
    echo "===== $mode $split b${minb}-b${maxb} ====="
    "$PY" scripts/stage1_feature_knn_routes.py \
      --mode "$mode" --split "$split" --min-bridge "$minb" --max-bridge "$maxb" \
      --beam-width 32 --feature-size 224 \
      --protocol-root "$PROTO" --output-root "$ROOT"
  done
done
echo "ALL_KNN_S224_DONE"
