#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=1
export SC_SAM_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/third_party/SC-SAM
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_c0
for entry in "S2 student_best.pth predictions/S2_valbest" "S2 student_final.pth predictions/S2_final" "S3 student_final.pth predictions/S3_final"; do
  set -- $entry
  run=$1; ckpt=$2; out=$3
  echo "===== export $out ====="
  "$PY" scripts/export_t25_student_predictions.py \
    --run-dir "$PHASE/students/$run" \
    --checkpoint "$PHASE/students/$run/$ckpt" \
    --output-root "$PHASE/$out"
done
echo EXPORT_S2S3_DONE
