#!/bin/bash
# TN3K BUSI-replication: run every remaining stage in order.
# Stage 1 (base-routes) is launched separately; this script starts at `pool`.
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
PIPE=new_project/tn3k_busi_factorial.py
R=new_project/experiments/tn3k_busi_factorial_20260915
STAMP() { date '+%F %T'; }

echo "[$(STAMP)] === pool ==="
"$PY" "$PIPE" pool

echo "[$(STAMP)] === smoke (2 targets, GPU0) ==="
"$PY" "$PIPE" smoke 0 2

echo "[$(STAMP)] === freeze-inputs ==="
"$PY" "$PIPE" freeze-inputs

echo "[$(STAMP)] === propagate validation (GPU0) + test (GPU1) in parallel ==="
"$PY" "$PIPE" propagate validation 0 >> "$R/logs/chain_validation.log" 2>&1 &
VPID=$!
"$PY" "$PIPE" propagate test 1 >> "$R/logs/chain_test.log" 2>&1 &
TPID=$!
echo "[$(STAMP)] validation pid=$VPID  test pid=$TPID"
set +e
wait "$VPID"; VRC=$?
wait "$TPID"; TRC=$?
set -e
echo "[$(STAMP)] validation propagate rc=$VRC  test propagate rc=$TRC"
[ "$VRC" -eq 0 ] || exit 1
[ "$TRC" -eq 0 ] || exit 1

echo "[$(STAMP)] === validate ==="
"$PY" "$PIPE" validate

echo "[$(STAMP)] === evaluate ==="
"$PY" "$PIPE" evaluate

echo "[$(STAMP)] === ALL DONE ==="
