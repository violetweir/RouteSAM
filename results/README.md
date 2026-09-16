# `results/` — curated per-run summaries

This directory is a **filtered extract of the original `work/` tree**. It exists
so that every number quoted in `reports/`, `paper/` and `mainline/` stays
auditable without shipping the raw outputs.

## What is kept

Only small text artifacts, with the original directory structure preserved:

| Pattern | Why |
|---|---|
| `*.md` | per-run experiment reports and analysis notes |
| `*.py` | the driver / analysis script that produced that run |
| `*.json` | run summaries, configuration snapshots, frozen protocol records |
| `*.csv` | small metric tables |
| `*.txt`, `*.sh`, `*.yaml` | command records and run settings |

## What is dropped

- mask dumps, visualisations and prediction PNGs (≈ 700 k files in the original)
- feature caches and checkpoints (`*.npz`, `*.npy`, `*.pt`, `*.pth`, `*.pkl`)
- raw logs (`*.log`, several GB)
- JSON/CSV files larger than 100 KB (mostly full route manifests and long metric
  ledgers; the paper-ready extracts of those live under `paper/`)

## Groups

**Historical S27 X3 + B7 ladder** (see [`../docs/s27_x3_b7_line.md`](../docs/s27_x3_b7_line.md))

- `reproduction_v1/`, `reproduction_reports/` — the fixed CVC + Kvasir
  reproduction protocol and its reports
- `rerun_c0/`, `rerun_c0_c0/`, `rerun_c0_256/`, `rerun_c0_256_base/`,
  `rerun_c0_256_sam3knn_s256_base/` — C0 / canvas-256 reruns and B7 LoRA runs
- `rerun_check/`, `rerun_c0_256_bbox_mask_sam3only/`,
  `rerun_c0_256_det_gtmask_sam3only/` — prompt-form ablations
- `rerun_c0_256_round2a_fixed_knn_e33/`,
  `rerun_c0_256_round2b_e33_topology_factorial/`,
  `rerun_c0_256_round2c_lesion_anchor_validation/`,
  `rerun_c0_256_round2d_anchor_only_full_route_validation/` — route topology,
  anchor and lesion-correspondence rounds
- `rerun_c0_256_propagation_aware_routing/`,
  `rerun_c0_256_propagation_risk_analysis/`,
  `rerun_c0_256_risk_gated_routing/` — propagation-aware / risk-gated routing
- `rerun_c0_256_round3_tracker_stage4/`,
  `rerun_c0_256_round3_tracker_lora_t3_t4/` — SAM3 tracker + LoRA stage

**Kvasir-SEG 1% anchors — the mainline's largest arm** (inventory: [`../docs/kvasir_program.md`](../docs/kvasir_program.md))

- `kvasir_1pct_anchors/` — route construction, weighted selectors, the
  candidate-invariant and propagation-quality routers, fine-tuning sweeps
- `rerun_kvasir_sam3base_test_20260906/`,
  `rerun_kvasir_sam3base_direct_text_ablation_20260906/` — SAM3-base direct and
  text-prompt ablations
- `kvasir_pc_*`, `kvasir_tp_*`, `kvasir_rethink_*`, `kvasir_b7_gap_*`,
  `kvasir_x3_pool_v2_*`, `x3_pool_v2_bundle_20260909/` — the September
  target-pooling / patch-correspondence / pseudo-pool studies
- `clinicdb_external_kvasir8/` — cross-dataset evaluation with Kvasir-8 anchors

**ISIC2018 and BUSI**

- `isic2018_1pct/`, `isic18_1pct_protocol/`, `isic18_cluster_1pct_k26/`,
  `isic18_sam3knn_s256_base/`, `isic18_pseudovideo_full/`,
  `isic18_round1_round2a_from_pseudovideo_full/`
- `busi_1pct_protocol/`

## Notes

- Absolute paths inside these files refer to the original server
  (`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/...`). They are historical records.
- Ready-to-read narrative versions of many of these runs are in `reports/`,
  and the paper-facing tables are in `paper/`.
