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

> For the frozen, step-by-step definition of Stage 0 (automatic anchor mining)
> and Stage 1 (routes → propagation → Router) with all hyper-parameters,
> verification evidence and the open items, see
> [`stage0_stage1.md`](stage0_stage1.md).

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
| TN3K | 2303 | 576 | 614 | 23 (0.9987%) |

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

| Method | cand/target | Kvasir (100) | ISIC2018 (260) | BUSI (66) | TN3K (614) |
|---|---:|---:|---:|---:|---:|
| raw score top-1 | 7 | 0.870848 | 0.868228 | 0.565990 | 0.519585 |
| **calibrated** top-1 | 7 | 0.860331 | 0.871717 | **0.719992** | **0.556705** |
| raw score top-2 | 14 | **0.891226** | 0.869276 | 0.656084 | **0.569973** |
| **calibrated** top-2 | 14 | 0.862761 | **0.875864** | **0.752198** | 0.560661 |
| historical per-bridge + legacy Router | 7 | 0.871374 | 0.866573 | 0.566808 | 0.515486 |

Candidate-pool ceiling and the remaining selection gap:

| Method | Kvasir Oracle | Gap | ISIC Oracle | Gap | BUSI Oracle | Gap | TN3K Oracle | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw top-1 | 0.912017 | 0.0412 | 0.883608 | 0.0154 | 0.617141 | 0.0512 | 0.606576 | 0.0870 |
| calibrated top-1 | 0.906667 | 0.0463 | 0.890596 | 0.0189 | 0.775461 | 0.0555 | 0.615749 | 0.0590 |
| raw top-2 | 0.932897 | 0.0417 | 0.903881 | 0.0346 | 0.776347 | 0.1203 | 0.699971 | 0.1300 |
| calibrated top-2 | **0.937542** | 0.0748 | **0.911909** | 0.0360 | **0.823649** | 0.0715 | **0.701492** | 0.1408 |
| historical per-bridge | 0.917615 | 0.0462 | 0.884316 | 0.0177 | 0.617141 | 0.0503 | 0.608208 | 0.0927 |

**Reading it honestly**

- **BUSI is the strong positive case.** Calibration is worth +0.154 (top-1) and
  +0.185 over the historical control (top-2, 95% CI [0.104, 0.271]). Paired
  bootstrap intervals exclude 0.
- **ISIC2018 is directionally positive but not significant at test.**
  On the full validation split the gains *are* significant
  (cal top-1 − raw top-1 = +0.0157 [0.0004, 0.0318];
  cal top-2 − raw top-2 = +0.0121 [0.0009, 0.0249]), but every test CI includes 0
  (+0.0035 and +0.0066).
- **TN3K has a third regime.** Calibrated top-1 is significantly better than raw
  top-1 (+0.0371 [0.0094, 0.0654]) and raw top-2 is significantly better than raw
  top-1 (+0.0504 [0.0319, 0.0691]), but calibrated top-2 is *not* better than raw
  top-2 (−0.0093, CI spans 0) and the interaction is significantly negative
  (−0.0464 [−0.0737, −0.0192]). Its headline scores are far lower than BUSI's, but
  the gap is mostly tail composition, not uniform quality loss — see the TN3K
  section below.
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

## TN3K — complete, and a different failure story

TN3K (official fold0; train 2303 / validation 576 / test 614; 23 automatic
anchors) finished on 2026-09-16 and is the **fourth complete dataset**. The same
four arms, the same legacy-28 Ridge(alpha=1), the same image-grouped folds:

| group | candidates | val OOF | test Dice | test IoU | test Oracle | test gap |
|---|---:|---:|---:|---:|---:|---:|
| raw top-1 | 7 | 0.510819 | 0.519585 | 0.432103 | 0.606576 | 0.086990 |
| **centered top-1** | 7 | 0.500433 | **0.556705** | 0.470921 | 0.615749 | 0.059044 |
| raw top-2 | 14 | 0.547491 | **0.569973** | **0.479870** | 0.699971 | 0.129997 |
| centered top-2 | 14 | **0.552716** | 0.560661 | 0.479035 | **0.701492** | 0.140831 |
| historical per-bridge | 7 | 0.511450 | 0.515486 | 0.428090 | 0.608208 | 0.092723 |

Paired test deltas (10 000-image bootstrap, seed 2026):

| comparison | delta | 95% CI | improved / worsened |
|---|---:|---|---:|
| centered top-1 − raw top-1 | **+0.037120** | **[0.009359, 0.065427]** | 304 / 240 |
| centered top-2 − raw top-2 | −0.009312 | [−0.035255, 0.016282] | 291 / 230 |
| raw top-2 − raw top-1 | **+0.050388** | **[0.031917, 0.069138]** | 215 / 165 |
| centered top-2 − centered top-1 | +0.003956 | [−0.015065, 0.023129] | 180 / 128 |
| centered top-2 − historical control | **+0.045175** | **[0.020218, 0.070233]** | 335 / 183 |
| interaction | −0.046432 | [−0.073690, −0.019191] | – |

This is a **third regime**: calibration helps at top-1 (significantly) but not at
top-2, breadth helps (top-2 > top-1, significantly), and the interaction is
significantly negative. It is neither BUSI (both help) nor Kvasir (neither helps).

### Why TN3K scores so much lower than BUSI

The gap is mostly a **tail-composition** effect, not a uniform quality drop
(`mainline/experiments/tn3k_busi_factorial_20260915/why_gap_vs_busi.md`):

| | BUSI | TN3K |
|---|---:|---:|
| test targets | 66 | 614 |
| 14-candidate Oracle | **0.8236** | **0.7015** |
| share of "reachable" targets (Oracle ≥ 0.5) | **92.4%** | **76.7%** |
| Oracle on reachable targets | **0.8794** | 0.8493 |
| Oracle on unreachable targets | 0.1430 | 0.2148 |
| final test Dice (centered top-2 Router) | 0.7522 | 0.5607 |

Of the 0.1221 Oracle gap, about **0.099** comes from the tail composition and only
**0.023** from quality on reachable targets.

### Root cause: anchor/target nodule **scale mismatch**

- Failure mode is **undersegmentation**, not oversegmentation: on unreachable
  targets the median prediction/GT area ratio is 0.211 and **76.2%** of failures
  are below GT (the earlier "oversegmentation" reading came from the mean, which
  a few outliers dominate; the report carries the errata).
- `corr(|log(anchor area / target area)|, Dice) = **−0.4776**`. Failure group:
  anchor/target area ratio median **0.158** vs **0.641** for reachable targets —
  the anchor nodule is ~6× smaller than the target. Anchors *larger* than the
  target are fine (Dice 0.865), so the failure is one-sided.
- Per-anchor failure rate is monotone in anchor nodule size (e.g. 677 px → 62%
  failure, 24501 px → 2%).
- **The obvious fix is falsified.** Enlarging the prompt box around its centre on
  validation monotonically hurts: s = 1.00 → 0.427296, 1.25 → 0.391002 (−0.0363),
  1.50 → 0.359899 (−0.0674), 2.00 → 0.246073 (−0.1812), 3.00 → 0.246137
  (`tn3k_boxscale_probe_20260916/probe_result.json`). The box is geometrically
  correct **on the anchor**; enlarging it distorts the anchor prompt. "Box too
  small" is a symptom, not the cause.
- The automatic selector optimises **appearance** coverage (global patch mean +
  64 local visual words) and has **no scale term**, so the chosen anchors are not
  scale-representative.

### Good candidates exist — the loss is ranking, not generation

| diagnostic (same 64 validation images, id-sorted random sample) | Dice |
|---|---:|
| target's **own** GT tight box, single frame, SAM3-base | **0.854998** |
| all-23-anchor `b0` Oracle | **0.828171** |
| raw top-2 `b0` Oracle | 0.638288 |
| centered top-2 `b0` Oracle | 0.617872 |
| raw top-1 `b0` | 0.535822 |

Only **1 of 64** targets has all 23 anchors' `b0` below 0.5. So the pool contains
good answers; reference **ranking / retention** is what loses them. A
reference-ranking pilot (`tn3k_reference_ranking_pilot_20260916/参考排序诊断.md`)
learns a 16-dimensional anchor–target matching score with a fixed-alpha ridge:
OOF Top-1 Dice **0.597020**, i.e. +0.061199 over raw TP, but the paired interval
[−0.020469, 0.143866] crosses 0. Pure re-centring, z-scoring, train-percentile or
global-cosine ranking all fail to beat raw TP, and an anchor-quality prior alone
scores only 0.478807 — so "always pick the on-average best anchor" is not the
answer.

### Why calibration itself does little on TN3K

Calibration needs the raw score to pick the **wrong** anchor. On BUSI the raw
score sends 65/66 `b0` targets to `malignant (187)` (systematically wrong), so
calibration repairs a real defect (+0.154 at top-1). On TN3K the raw score does
degenerate (train-mean span 0.5502 vs BUSI 0.3498), but the anchor it degenerates
to **is already the per-bridge best** (val 560/576, test 604/614) — there is
almost nothing to repair, so the calibration gain shrinks to +0.037 at top-1 and
vanishes at top-2.

### What this changes for the paper

1. Report **both the full set and the reachable subset**, or the method is
   understated.
2. The binding constraint on TN3K is **anchor ranking that is scale-blind**.
   Promising directions, in order of directness: (a) add a scale-compatibility
   term to the ranking (the pool already contains good answers); (b) make the
   automatic selector cover **scale** as well as appearance; (c) simply retain
   more anchors — the all-23 `b0` Oracle (0.828) is far above the top-2 `b0`
   Oracle (0.618), at linear cost in candidates and propagation.
3. **BUSI's +0.185 is not a universal law**: it depends on the raw score being
   systematically wrong, a precondition that does not hold on TN3K.

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
mainline/experiments/tn3k_busi_factorial_20260915/                  TN3K four-arm ablation (complete)
mainline/experiments/tn3k_failure_diagnosis_20260916/               TN3K failure localisation
mainline/experiments/tn3k_boxscale_probe_20260916/                  prompt-box scale probe (negative result)
mainline/experiments/tn3k_reference_ranking_pilot_20260916/         reference ranking / Top-K diagnostics
results/busi_1pct_protocol/result.md                                SynFoC BUSI 1% baseline + head-to-head
```

## One-sentence summary

> Under a 1% anchor budget with a frozen foundation model, the ceiling is set by
> candidate breadth and depth; anchor-conditioned scores are not comparable
> across anchors, that bias is measurable **without any labels**, and it predicts
> the benefit of calibration — but calibration is the *weakest* of three
> mutually substitutable levers, so spending the budget on coverage and on
> breadth/depth beats making the scores more accurate.
