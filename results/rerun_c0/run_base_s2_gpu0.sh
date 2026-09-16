#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export SC_SAM_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/third_party/SC-SAM
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt
"$PY" scripts/run_t24_student.py --data-path $DATA --labeled-list $LABELS --pseudo-manifest work/rerun_c0_256_base/pseudo_manifest_original.jsonl --output-dir work/rerun_c0_256_base/students/S2 --experiment S2 --seed 2026 --max-iterations 40000 --val-interval 200 --num-workers 4
echo S2_BASE_DONE
