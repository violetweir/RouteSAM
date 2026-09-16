# Cross-dataset 1% anchor study (Kvasir / ISIC2018 / BUSI / TN3K)

> Latest experiment line, 2026-09-13 → 09-15. Frozen SAM3, no large-model
> training, one pipeline and the same code for every dataset.
>
> Primary sources (Chinese, with all raw numbers and frozen artifacts):
> [`mainline/reproduction_guides/automatic_selection_20260914/PAPER_OUTLINE.md`](../mainline/reproduction_guides/automatic_selection_20260914/PAPER_OUTLINE.md),
> [`P1_REPORT.md`](../mainline/reproduction_guides/automatic_selection_20260914/P1_pilot_20260914/P1_REPORT.md),
> [`P2_REPORT.md`](../mainline/reproduction_guides/automatic_selection_20260914/P2_fullbank_20260914/P2_REPORT.md),
> [`E0_diagnostics.md`](../mainline/reproduction_guides/automatic_selection_20260914/E0_diagnostics_20260914/E0_diagnostics.md),
> [three-dataset reproduction guide](../mainline/reproduction_guides/automatic_selection_20260914/三数据集自动选图与TP路线_中文复现指南.md).

## Why this line exists

The `S27 X3 + B7` mainline answers *"how far can a student-audited pseudo-video
pipeline go on one merged dataset"*. This line asks a different question:
with a **1% annotation budget** and a **frozen** foundation model, **where is the
bottleneck on each dataset** — anchor coverage, candidate breadth/depth, or
verification?

To answer it the pipeline is decomposed so that every change moves either the
ceiling or the gap:

```text
Realized = Oracle(H) − Gap(H, V)
```

- `H` is the candidate set (anchors × bridge depth) — it determines the **Oracle**
- `V` is the verifier / router — it can only reduce the **Gap**, never the Oracle

## Pipeline (ACV: coverage → calibration → verification)

```text
[0] frozen SAM3; cache 1008 features (anchor selection) and 256 features (propagation)
[1] coverage anchor selection — train RGB only, GT never read
    S = 0.5·clip(GGᵀ,0,1) + 0.5·clip(HHᵀ,0,1)     G = global descriptor
                                                   H = 64 local visual words, IDF-weighted
    F(A) = (1/N) Σᵢ max_{a∈A} S(i,a)               monotone submodular → greedy, 1−1/e
    K = round(0.01 · N_train)
[2] human labels those K images                     ← the only manual step
[3] score calibration (annotation-free)
    μ_A = mean_{x ∈ train RGB} TP(A, x)
    centered(A,T) = TP(A,T) − μ_A
[4] candidate generation (breadth k × depth b0–b6)
    keep top-k anchors per target by centered score; each anchor → beam-32 path, 7 depths
    ⇒ 7k candidate masks per target
[5] verification: Ridge (legacy-28 features) with image-grouped 5-fold CV on validation
[6] report Oracle and realized Dice, and the Gap between them
```

`K ∈ {1,2,3,5}` and `alpha ∈ {1,10,100}` are selected by the same
validation protocol on each dataset. Prompt is the anchor's GT tight box, no
text, canvas 256.

## Protocol

| Dataset | train | validation | test | anchors (1%) |
|---|---:|---:|---:|---:|
| Kvasir-SEG | 800 | 100 | 100 | 8 (1.0000%) |
| ISIC2018 | 2075 | 259 | 260 | 21 (1.0120%) |
| BUSI | 517 | 64 | 66 | 5 (0.9671%) |
| TN3K | 2303 | 576 | 614 | 23 (0.9987%) — dry-run only so far |

Anchor lists are frozen before any GT is read and are stored in
`mainline/experiments/*/selection/SELECTIONS_FROZEN.json`.

## The diagnostic: bias ratio

Anchor-conditioned TP scores are **not comparable across anchors**: some
anchors score every target highly. Because `μ_A` needs no labels, the spread of
`μ_A` gives an annotation-free predictor of how much calibration will help:

```text
bias_ratio = std_A(μ_A) / std_T(max_A s(A,T))
```

| Dataset | anchors | std(μ_A) | **bias ratio** | test re-selection rate | concentration raw → cal |
|---|---:|---:|---:|---:|---|
| Kvasir | 8 | 0.0216 | **0.68** | 43% | 0.360 → 0.220 |
| ISIC2018 | 21 | 0.0449 | **1.10** | 65% | 0.235 → 0.158 |
| BUSI | 5 | 0.1448 | **5.42** | 85% | 0.985 → 0.348 |

On BUSI 98.5% of targets originally picked the same anchor, whose mean Dice is
only 0.51. On the clean, unselected BUSI validation bank, `rho(raw, Dice) ≈ 0.14`
but `rho(centered, Dice) ≈ 0.38`; calibrating the anchor choice alone lifts mean
Dice from 0.592 to 0.692 (+0.0998, 95% CI [0.0317, 0.1702]) with **no new labels
and no new model**.

## Main cross-dataset result (full test split)

Score calibration × anchor breadth, four pre-registered arms plus the historical
per-bridge control. All arms refit the same legacy-28 Ridge(alpha=1) on the same
image-grouped 5-fold validation.

| Method | cand/target | Kvasir (100) | ISIC2018 (260) | BUSI (66) |
|---|---:|---:|---:|---:|
| raw score top-1 | 7 | 0.870848 | 0.868228 | 0.565990 |
| **calibrated** top-1 | 7 | 0.860331 | 0.871717 | **0.719992** |
| raw score top-2 | 14 | **0.891226** | 0.869276 | 0.656084 |
| **calibrated** top-2 | 14 | 0.862761 | **0.875864** | **0.752198** |
| historical per-bridge + legacy Router | 7 | 0.871374 | 0.866573 | 0.566808 |

Candidate-pool ceiling and the remaining selection gap:

| Method | Kvasir Oracle | Gap | ISIC Oracle | Gap | BUSI Oracle | Gap |
|---|---:|---:|---:|---:|---:|---:|
| raw top-1 | 0.912017 | 0.0412 | 0.883608 | 0.0154 | 0.617141 | 0.0512 |
| calibrated top-1 | 0.906667 | 0.0463 | 0.890596 | 0.0189 | 0.775461 | 0.0555 |
| raw top-2 | 0.932897 | 0.0417 | 0.903881 | 0.0346 | 0.776347 | 0.1203 |
| calibrated top-2 | **0.937542** | 0.0748 | **0.911909** | 0.0360 | **0.823649** | 0.0715 |
| historical per-bridge | 0.917615 | 0.0462 | 0.884316 | 0.0177 | 0.617141 | 0.0503 |

**Reading it honestly**

- **BUSI is the strong positive case.** Calibration is worth +0.154 (top-1) and
  +0.185 over the historical control (top-2, 95% CI [0.104, 0.271]). Paired
  bootstrap intervals exclude 0.
- **ISIC2018 is directionally positive but not significant at test.**
  On the full validation split the gains *are* significant
  (cal top-1 − raw top-1 = +0.0157 [0.0004, 0.0318];
  cal top-2 − raw top-2 = +0.0121 [0.0009, 0.0249]), but every test CI includes 0
  (+0.0035 and +0.0066).
- **Kvasir is a negative result.** Calibration does not help; the calibrated
  top-2 pool has the *highest* Oracle (0.937542) yet the router realizes only
  0.862761, leaving a 0.075 gap. The candidates are better, the legacy scorer
  fails to cash them in — here selection, not coverage, is the bottleneck.
- So the calibration gain is **gated by the bias ratio**:
  `5.42 → large and significant`, `1.10 → small and not significant`,
  `0.68 → none`.

## Three levers that substitute for each other

Adding anchors (breadth `k`), adding bridge depth (`m`), and calibrating all
enlarge "effective evidence" — and they trade off:

| Dataset | k=1→2, b0 only | k=1→2, full depth |
|---|---:|---:|
| Kvasir test | +0.0689 | **+0.0148** [+0.007, +0.025] |
| Kvasir validation | +0.0899 | **+0.0370** [+0.015, +0.066] |
| BUSI validation | +0.1603 | **+0.1079** |
| ISIC2018 test | +0.0389 | not yet measured |

The same "keep one more anchor" move pays far more on shallow paths than on deep
ones. Depth also partially compensates for a bad anchor: BUSI's calibration gain
falls from +0.2063 at `b0` to +0.0788 at full depth.

Kvasir full-depth ceiling vs `k` (P2, 56 candidates/target):

| k | candidates | val raw | val cal | test raw | test cal |
|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 0.8895 | 0.9062 | 0.9180 | 0.9067 |
| 2 | 14 | 0.9265 | 0.9238 | 0.9328 | 0.9388 |
| 3 | 21 | 0.9328 | 0.9371 | 0.9432 | 0.9455 |
| 5 | 35 | 0.9400 | 0.9405 | 0.9525 | 0.9510 |
| 8 | 56 | 0.9458 | 0.9458 | 0.9542 | 0.9542 |

Adjacent-`k` Oracle gains are all significantly positive but decay fast
(`k=1→2` +0.0148, `2→3` +0.0104, `3→5` +0.0093, `5→8` +0.0017 on test).

## The 1% budget is conservative

Greedy-prefix budget curve on Kvasir, full-depth Oracle:

| labelled anchors | validation Oracle | test Oracle | vs K=1 |
|---:|---:|---:|---:|
| 1 (0.125% of train) | 0.8822 | 0.9142 | — |
| 2 | 0.9027 | 0.9239 | +0.0097 |
| 4 | 0.9367 | 0.9440 | +0.0299 |
| 8 (1%) | 0.9458 | 0.9542 | +0.0400 |

One labelled image already reaches a 0.9142 test ceiling; going to the full 1%
adds only +0.040. (This is a ceiling, not a deployable score.)

## Comparison with a trained low-label baseline (BUSI)

SynFoC (`third_party/SynFoC-T20`) trained on the same 5 labelled images and the
same 517/64/66 split produces one mask per image:

| Method | BUSI test Dice | Δ vs SynFoC |
|---|---:|---:|
| raw top-1 (pre-calibration) | 0.565990 | −0.0871 |
| raw top-2 | 0.656084 | +0.0030 |
| **SynFoC MedSAM+LoRA** (validation-selected branch) | **0.653120** | — |
| SynFoC UNet (other branch, not selected) | 0.683123 | +0.0300 |
| calibrated top-1 (no router) | 0.719992 | +0.0669 |
| **calibrated top-2 + router** | **0.752198** | **+0.0991** |
| calibrated top-2 Oracle (upper bound) | 0.823649 | +0.1705 |

Single seed, single run for both sides — do not over-interpret small differences.

## TN3K (preliminary, dry-run only)

A fourth dataset is being added on the same protocol
(`train 2303 / validation 576 / test 614`, 23 anchors). A **dry-run on a 23-target
validation / 25-target test subset** has completed; the full run is still
propagating, so these numbers are **not** final and val/test disagree:

| Method | cand/target | dry-run val OOF | dry-run test | test Oracle |
|---|---:|---:|---:|---:|
| raw top-1 | 7 | 0.529984 | 0.525470 | 0.630886 |
| calibrated top-1 | 7 | 0.505007 | 0.518560 | 0.609742 |
| raw top-2 | 14 | 0.480613 | 0.604559 | 0.729632 |
| calibrated top-2 | 14 | 0.664819 | 0.497413 | 0.688370 |
| historical per-bridge | 7 | 0.522799 | 0.525470 | 0.630886 |

What the dry-run does establish is that the *mechanism* replicates and is in fact
stronger on TN3K: the anchor train-mean span is 0.5502 (BUSI: 0.3498), the raw
target score is essentially uncorrelated with true Dice (Pearson +0.045), and
calibration changes the rank-1 anchor on **25/25** test targets. See
[`dryrun/analysis_extra.md`](../mainline/experiments/tn3k_busi_factorial_20260915/dryrun/analysis_extra.md).

## Diagnostics figures

All under `mainline/reproduction_guides/automatic_selection_20260914/E0_diagnostics_20260914/figures/`:

| Figure | Content |
|---|---|
| `fig1_bias_diagnostics.png` | bias ratio and re-selection rate per dataset |
| `fig2_anchor_score_vs_dice.png` | raw vs calibrated anchor score against true Dice |
| `fig3_depth_cycle_dice.png` | depth vs `q_cycle` and Dice |
| `fig4_oracle_gap.png` | Oracle / realized / Gap decomposition |
| `fig5_busi_ceiling_vs_k.png` | BUSI ceiling vs anchor breadth `k` |
| `fig6_busi_score_vs_dice.png` | BUSI score–quality relation |
| `fig7_busi_fullbank_mechanism.png` | BUSI full-bank mechanism check |

## Limitations

1. **Calibration is not a universal fix** — it is neutral at bias ratio ≈ 1 and
   useless below 1.
2. **ISIC2018 full-depth calibration is still unmeasured** (P3, ~12 h GPU).
3. **The router uses validation GT.** "1%" is the *anchor labelling* budget, not
   the total supervision of the system.
4. **Test has been looked at across several rounds** — this is not a blind test.
5. **Greedy coverage has no downstream-Dice evidence against random anchors yet**
   (P4 pending); the 100 historical random control sets only compared coverage
   proxies (`No propagation performed`).
6. **Score→quality rank correlation is only ~0.1–0.4**, which is the intrinsic
   ceiling of this family and the source of the Gap.
7. **BUSI validation is only 64 images**, so its selected hyper-parameters and
   intervals are wide.

## Open next steps

| Priority | Experiment | Gap it closes | Cost | Status |
|---|---|---|---|---|
| ~~P2~~ | Kvasir full anchor bank | second full-depth dataset | ~3 h | done |
| **P3** | ISIC2018 top-5 (35 cand/target) | third full-depth point; unified main table | ~12 h | pending |
| **P4** | random anchor control (10 sets) | downstream evidence for the coverage claim | ~4 h | pending |
| **P5** | Kvasir unified Router refit | makes the main table one method | 0 GPU (candidates ready) | pending |
| ~~P6~~ | budget curve K = 1,2,4,8 | "why 1%" | 0 | done (P2) |
| P7 | a never-diagnosed dataset | one-shot blind test | dataset-dependent | optional |

## Where the artifacts live

```text
mainline/reproduction_guides/automatic_selection_20260914/
  PAPER_OUTLINE.md                  paper outline and claim table
  E0_diagnostics_20260914/          bias / Oracle-Gap diagnostics + figures + candidate CSVs
  P1_pilot_20260914/                b0 calibration pilot, preregistration, predictions
  P2_fullbank_20260914/             Kvasir 56-candidate full bank
  verify_{kvasir,isic2018,busi}/    anchor-selection replay verification
  verify_router_{kvasir_isic,busi}/ router refit verification
  verify_tp_{kvasir,isic2018,busi}/ TP b0–b6 verification
  三数据集自动选图与TP路线_中文复现指南.md   end-to-end reproduction guide
mainline/experiments/cross_dataset_calibration_factorial_20260914/  Kvasir + ISIC2018 four-arm ablation
mainline/experiments/busi_calibration_factorial_20260914/           BUSI four-arm ablation
mainline/experiments/busi_calibrated_multi_anchor_20260913/         BUSI calibration + top-2
mainline/experiments/tn3k_busi_factorial_20260915/                  TN3K (dry-run + running)
results/busi_1pct_protocol/result.md                                SynFoC BUSI 1% baseline + head-to-head
```

## One-sentence summary

> Under a 1% anchor budget with a frozen foundation model, the ceiling is set by
> candidate breadth and depth; anchor-conditioned scores are not comparable
> across anchors, that bias is measurable **without any labels**, and it predicts
> the benefit of calibration — but calibration is the *weakest* of three
> mutually substitutable levers, so spending the budget on coverage and on
> breadth/depth beats making the scores more accurate.
