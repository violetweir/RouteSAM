#!/usr/bin/env bash
# SynFoC baseline on the frozen BUSI split (train517 / val64 / test66).
#
# The protocol adapter is built by scripts/prepare_busi_synfoc_protocol.py.
# SynFoC train.py rewrites CUDA_VISIBLE_DEVICES from --gpu, so pass the physical
# GPU id there as well.
set -euo pipefail

PV_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PROTOCOL=$PV_ROOT/work/busi_1pct_protocol
DATA=$PROTOCOL/data
LABELS=$PROTOCOL/busi_train1pct_labeled_images.txt
SYNFOC=$PV_ROOT/third_party/SynFoC-T20
CKPT=/Data_8TB/lht/models/medsam_vit_b.pth
GPU=0

run_train() {
  local out=$1 max_iter=$2 eval_iter=$3 save_name=$4
  mkdir -p "$out"
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$SYNFOC"
  "$PY" -u train.py \
    --dataset ClinicDB \
    --dataset_label BUSI \
    --data_path "$DATA" \
    --labeled_list "$LABELS" \
    --output_dir "$out" \
    --save_name "$save_name" \
    --model MedSAM \
    --ckpt "$CKPT" \
    --max_iterations "$max_iter" \
    --num_eval_iter "$eval_iter" \
    --label_bs 4 \
    --unlabel_bs 4 \
    --test_bs 8 \
    --img_size 256 \
    --base_lr 0.03 \
    --seed 2026 \
    --AdamW \
    --warmup \
    --save_model \
    --overwrite \
    --gpu "$GPU"
}

case "${1:-}" in
  smoke) run_train "$PROTOCOL/synfoc_smoke" 400 200 busi_1pct_smoke ;;
  full) run_train "$PROTOCOL/synfoc" 40000 500 busi_1pct_seed2026 ;;
  *) echo "usage: $0 smoke|full" >&2; exit 2 ;;
esac
