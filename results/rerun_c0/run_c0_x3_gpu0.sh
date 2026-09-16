#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export SC_SAM_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/third_party/SC-SAM
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt
"$PY" scripts/run_s27_student.py --data-path "$DATA" --labeled-list "$LABELS" --pseudo-manifest work/rerun_c0_c0/pseudo_manifest_x3.jsonl --output-dir work/rerun_c0_c0/students/X3 --experiment X3 --seed 2026 --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 --max-iterations 40000 --val-interval 200 --num-workers 4
echo X3_DONE
