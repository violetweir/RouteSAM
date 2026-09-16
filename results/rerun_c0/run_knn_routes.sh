#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=work/rerun_c0/stage1_feature_knn_b7
PROTO=work/kvasir_1pct_anchors/protocol
for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  echo "===== $mode train b3-b6 ====="
  "$PY" scripts/stage1_feature_knn_routes.py --mode "$mode" --split train --min-bridge 3 --max-bridge 6 --beam-width 32 --protocol-root "$PROTO" --output-root "$ROOT"
  echo "===== $mode validation b0-b6 ====="
  "$PY" scripts/stage1_feature_knn_routes.py --mode "$mode" --split validation --min-bridge 0 --max-bridge 6 --beam-width 32 --protocol-root "$PROTO" --output-root "$ROOT"
  echo "===== $mode test b0-b6 ====="
  "$PY" scripts/stage1_feature_knn_routes.py --mode "$mode" --split test --min-bridge 0 --max-bridge 6 --beam-width 32 --protocol-root "$PROTO" --output-root "$ROOT"
done
echo "ALL_KNN_ROUTES_DONE"
