#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=1
export SC_SAM_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/third_party/SC-SAM
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
$PY scripts/run_t24_student.py --data-path work/kvasir_1pct_anchors/baseline_data --labeled-list work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt --pseudo-manifest work/rerun_c0_256/S3_consensus/pseudo_consensus.jsonl --output-dir work/rerun_c0_256/students/S3 --experiment S3 --seed 2026 --max-iterations 40000 --val-interval 200 --num-workers 4
echo S3_256_DONE
