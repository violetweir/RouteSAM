# `mainline/` — experiment index

This tree **is the mainline**: the cross-dataset 1% anchor study on Kvasir-SEG,
ISIC2018 and BUSI (plus a TN3K arm). It replaces the multi-family pseudo-video
routes with **single-family Target-Pooling (TP) `b0–b6` candidates + an
independent router**, and asks where the bottleneck is under a 1%-anchor budget
on each dataset.

The Kvasir arm of this tree is the **V1 → V7 ladder** (V1–V4 = original 8 anchors
+ route/student evolution; V5–V7 = automatic anchors, single TP + Router, V7 adds
score calibration). The ladder table and per-version results are in
[`../docs/kvasir_versions.md`](../docs/kvasir_versions.md) (per-version walkthrough),
[`../docs/kvasir_program.md`](../docs/kvasir_program.md) §"The V1 → V7 version
ladder", and the root README.

Mainline overview: [`../docs/cross_dataset_1pct.md`](../docs/cross_dataset_1pct.md).
The Kvasir-SEG-only arm is catalogued in
[`../docs/kvasir_program.md`](../docs/kvasir_program.md).

Start with [`README.md`](README.md) — the method note *"先筛合格候选，再由 Router
选优"* (filter admissible candidates first, then let the router choose), which
documents how the first-pass pseudo-label pool grew from 448 to 580 images.

```text
mainline/
  README.md                          method note (Chinese)
  *.md                               top-level design & result notes
  *.py                               top-level experiment drivers
  code/                              shared trainer / prepare / test code
  experiments/<name>_<date>/         one directory per experiment (see below)
  reproduction_guides/               self-contained reproduction bundles
  stage1_old_new_comparison_20260912/
  stage23_old_new_comparison_20260912/
  kvasir_labeled_8/                  the 8 fixed Kvasir anchors (images + masks)
```

Each experiment directory keeps `code/`, the frozen-protocol JSON (frozen
predictions, weights, hashes) and its report. Heavy outputs (masks, weights,
feature caches) are not included.

## Top-level notes

| File | Topic |
|---|---|
| `README.md` | TP-only pool: filter-then-select, 448 → 580 pool |
| `single_student_training.md` | confirmed single-student recipe (588 images, bs 12, 816 epochs) |
| `single_student_experiment.md` | hard vs soft label training setup |
| `single_student_experiment_results.md` | hard 0.8196/0.8486 vs soft 0.8273/0.8512 (val/test) |
| `pseudo_label_construction.md` | `Y = 0.75*M + 0.25*P` label construction |
| `tp_student_rescreen_experiment.md` | full 792-image rescreen protocol |
| `tp_student_rescreen_results.md` | rescreen result: best/final test 0.858330 / 0.869478 |
| `tp_tracker_endpoint_experiment.md` | tracker-endpoint study design |
| `tp_tracker_pilot_results.md` | tracker-endpoint pilot results |
| `round2_pool_a_report.md`, `round2_pool_a_vs_580.md` | round-2 pool A |
| `round2_selection_ablation.md` | round-2 pool A selection ablation |
| `new_b7_results.md` | updated B7 results |
| `A0_second_training_*.md`, `A1_A2_*.md`, `A0_A1_A2_two_seed_results.md` | round-2 training configuration sweeps |
| `automatic_anchor_findings_and_top2_plan.md` | automatic anchor selection findings and plan |

## Experiments

### Anchor selection and the TP router

| Directory | Report |
|---|---|
| `automatic_anchor_selection_pilot_20260911` | 自动参考样本选择首轮实验 |
| `auto8_tp_validation_20260911` | 自动 8 张参考图 TP b0–b6 验证结果 |
| `automatic_anchor_tp_router_20260913` | 自动选图 + SAM3-base 单 TP b0–b6 Router 实验 |
| `automatic_anchor_tp_test_20260913` | 自动选图 + SAM3-base KNN + 单 TP b0–b6：Test 实验 |
| `isic2018_auto21_tp_validation_20260911` | ISIC2018 自动选 21 张：TP 验证结果 |

### Single-student training and pseudo-pool screening

| Directory | Report |
|---|---|
| `single_student_hard_soft_20260909` | 单学生等权训练：硬标签与合格候选软标签对照 |
| `tp448_single_student_soft_20260912` | 原 448 张池 + 新版单学生软标签训练结果 |
| `tp448_s2_s3_test_20260912` | 448 张 TP-only：S2/S3 独立 test 补测 |
| `tp_student_rescreen_20260910` | TP + 单学生统一重筛：实验完成 |
| `tp_tracker_endpoint_20260910` | tracker endpoint study (results only, no report) |

### Round-2 pool and selection ablations

These are the **second LoRA round** of the Kvasir line: SAM3 LoRA is re-trained
from the base on the round-2 B7-rescreened student-audited pool and evaluated as a
direct single-image segmenter (fixed `colon polyp` text, no TP/Router/B7). All
10-epoch unless noted. Round-1 LoRA is `lora_1pct_e20` / `lora_p491_e20`
(20 epochs, 8 GT and 8 GT + 491 pseudo) — see
[`../docs/kvasir_program.md`](../docs/kvasir_program.md) §D.4.

| pool (pseudo + GT) | epochs | best epoch | Val Dice | Test Dice |
|---|---:|---:|---:|---:|
| A0 · 428 + 8 | 50 (no clip) | 19 | 0.893717 | 0.901950 |
| A0 · 428 + 8 | 50 (final) | 50 | 0.873586 | 0.894245 |
| A0 · 428 + 8 | 10 | 6 | 0.880909 | 0.901755 |
| A1 · 596 + 8 | 10 | 4 | 0.887257 | 0.903294 |
| A2 · 503 + 8 | 10 | 1 | 0.888680 | 0.912932 |

Two seeds (10 epochs): A0 0.901755 / 0.892288, A1 0.903294 / 0.886602,
A2 0.912932 / 0.884289 — **the ordering reverses between seeds**, so no pool is
stably best and round 2 does not beat round 1.

| Directory | Report |
|---|---|
| `round2_pool_a_20260910` | Round2：B7 重新筛选池 A |
| `round2_selection_ablation_20260910` | Round2 池 A 筛选消融与 SAM3 单图微调对比 |
| `round2_A0_cosine_clip_original1008_20260910` | A0 第二轮训练：原图 1008 + cosine + 梯度裁剪 |
| `round2_A1_A2_cosine_clip_original1008_20260910` | A1/A2：新设置重新训练 |
| `round2_A2_low_lr_two_seeds_20260910` | A2 降低学习率：双种子对照 |
| `round2_A2_repeat_seed2027_20260910` | A2 随机种子复现实验 |
| `round2_A0_A1_repeat_seed2027_20260910` | A0/A1 seed2027 重复训练 |
| `round2_A0_e50_no_clip_seed2026_20260911` | A0 seed2026：50 epochs，无梯度裁剪 |
| `round2_A2_low_lr_no_clip_seed2026_20260911` | A2 关闭梯度裁剪：seed2026 |
| `round2_A2_restored_lr_no_clip_seed2026_20260911` | A2 seed2026 恢复学习率、关闭梯度裁剪 |

### Cross-dataset 1% study: BUSI, TN3K, Kvasir / ISIC2018 calibration

See **[../docs/cross_dataset_1pct.md](../docs/cross_dataset_1pct.md)** for the full
write-up. Headline test Dice (frozen SAM3, 1% anchors, legacy-28 Ridge):

| Dataset | raw top-1 | cal top-1 | raw top-2 | cal top-2 | bias ratio |
|---|---:|---:|---:|---:|---:|
| Kvasir (100) | 0.870848 | 0.860331 | **0.891226** | 0.862761 | 0.68 |
| ISIC2018 (260) | 0.868228 | 0.871717 | 0.869276 | **0.875864** | 1.10 |
| BUSI (66) | 0.565990 | 0.719992 | 0.656084 | **0.752198** | 5.42 |
| TN3K (dry-run 25) | 0.525470 | 0.518560 | **0.604559** | 0.497413 | n/a (mean-span 0.55) |

| Directory | Report |
|---|---|
| `automatic_anchor_selection_pilot_20260911` | coverage anchor selection first pilot |
| `automatic_anchor_tp_router_20260913` | automatic anchors + SAM3-base single-TP b0–b6 + independent Router (Kvasir, ISIC2018) |
| `automatic_anchor_tp_test_20260913` | same, full test split |
| `busi_auto5_tp_1pct_20260913` | BUSI: automatic 1% anchors + TP b0–b6 (protocol + selection features) |
| `busi_auto5_tp_router_20260913` | BUSI automatic 1%: Router result |
| `busi_mask_prompt_ablation_20260913` | BUSI: full-mask prompt, with/without text |
| `busi_failure_diagnosis_20260913` | BUSI low-performance diagnosis (code, data, validation) |
| `busi_calibrated_multi_anchor_20260913` | BUSI: score calibration + multi-anchor top-2 (test 0.752198) |
| `busi_calibration_factorial_20260914` | BUSI: calibration × anchor count, four arms + control |
| `cross_dataset_calibration_factorial_20260914` | Kvasir + ISIC2018: the same four-arm ablation, full validation and test |
| `isic2018_auto21_tp_validation_20260911` | ISIC2018 automatic 21 anchors: TP validation |
| `tn3k_busi_factorial_20260915` | TN3K: the same four-arm ablation. **Dry-run complete on a 23-val / 25-test subset; the full 576/614 run is still propagating**, so its numbers are preliminary and val/test disagree |

### Reproducibility audits

| Directory | Report |
|---|---|
| `stage1_reproducibility_audit_20260912_py312` | 环节 1：TP+PC 与 TP-only 可复现性核查 |
| `stage1_old_new_comparison_20260912` | old vs new stage-1 comparison (code + report) |
| `stage23_old_new_comparison_20260912` | old vs new stage-2/3 comparison (code + report) |

## `reproduction_guides/`

`automatic_selection_20260914/` is the self-contained bundle for the
cross-dataset study. Start with `PAPER_OUTLINE.md` (claims, tables, pending
experiments) and `三数据集自动选图与TP路线_中文复现指南.md` (end-to-end
reproduction guide with frozen commands and acceptance values).

- `E0_diagnostics_20260914/` — 0-GPU diagnostics over archived results: bias
  ratio, Oracle/Gap decomposition, score–quality correlations, seven figures,
  and the unified `per_candidate_*.csv` tables
- `P1_pilot_20260914/{kvasir,isic2018,preregistration}` — b0 calibration pilot
  with prediction written **before** the GPU run, plus `P1_REPORT.md`
- `P2_fullbank_20260914/kvasir/` — Kvasir 56-candidate full bank (8 anchors ×
  b0–b6), budget curve and ceiling-vs-`k` curve, plus `P2_REPORT.md`
- `verify_{kvasir,isic2018,busi}/` — anchor-selection replay verification
  (`selection_replay.json`, `descriptors.npz`, frozen anchor IDs)
- `verify_tp_{kvasir,isic2018,busi}/` — TP b0–b6 verification bundles with
  per-candidate metrics
- `verify_router_kvasir_isic/`, `verify_router_busi/` — router refit
  verification against `fixed_val_best`, `peer_consensus`, `baseline_legacy`,
  `cycle_only`, `selected_router`
- `reproduce_automatic_selection.py`, `verify_commands*.py`, `source_hashes.json`
  — reproduction entry point, command checks and input hashes

## `kvasir_labeled_8/`

The 8 fixed Kvasir 1% anchors as images + masks, with `manifest.json`,
`chain_source_report.md` and an HTML index. Small enough to keep in-tree; useful
as a qualitative figure source.
