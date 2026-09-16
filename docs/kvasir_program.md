# Kvasir-SEG 1% anchor program — complete experiment inventory

> The mainline (cross-dataset 1% anchor study) runs the same pipeline on
> Kvasir-SEG, ISIC2018 and BUSI. **Kvasir-SEG is by far the largest arm**: it
> carries the full ablation program behind the method. This page is the index of
> everything that was run on it.
>
> Mainline overview: [`cross_dataset_1pct.md`](cross_dataset_1pct.md).
> Task/router reference: [`kvasir_1pct_anchor.md`](kvasir_1pct_anchor.md).

## The V1 → V7 version ladder

> Per-version detail (motivation, exact pipeline, pool sizes, results tables,
> author caveats, artifact paths) lives in [`kvasir_versions.md`](kvasir_versions.md).

Kvasir-SEG was rebuilt seven times. V1–V4 keep the original 8 fixed anchors and
evolve the route families and the student pipeline; V5–V7 switch to automatic
coverage anchors. The shared shape of V1–V4 is the Kvasir-only analogue of
`S27 X3 + B7`.

| Version | Mainline | Main change |
|---|---|---|
| **V1** | original 8 anchors → DINOv3 kNN → SAM3 **TP + PC** → S2/S3 → committee → X3 → B7 | earliest complete pipeline; SAM3 LoRA added later |
| **V2** | original 8 anchors → **SAM3-base feature** kNN → TP + PC → S2/S3 → X3 → B7 | SAM3 features replace DINOv3; carries the Round-1 / Round-2A follow-ups |
| **V3** | original 8 anchors → **single TP** `b0–b6` + independent Router → 448 pseudo labels → S2/S3 → expansion, X3, B7 | drops PC, settles the single-TP baseline |
| **V4** | original 8 anchors → TP → **filter admissible candidates before the Router** → 580 → single student soft labels → full rescreen → new students, B7 | 580-pool S2/S3 control, single student hard/soft, 620 rescreen, and the later round-2 SAM3 fine-tune |
| **V5** | **automatic 8 anchors** → SAM3-base kNN → single TP `b0–b6` → Router | changes where the anchors come from |
| **V6** | automatic 8 anchors → **raw TP score top-2** anchors → `b0–b6` each → Router | 7 → 14 candidates per target |
| **V7** | automatic 8 anchors → **calibrated TP score top-2** anchors → `b0–b6` each → Router | annotation-free score calibration |

Test Dice (Kvasir, 100 targets):

| Version | headline | Oracle | supporting numbers |
|---|---|---:|---|
| V1 | X3-best + B7 **0.895432** | 0.921471 | Router 0.874083; X3 single 0.854101; 410 first-pass labels, committee A 94 / B 128, X3 pool 632 |
| V2 | X3 + B7 **0.899369** | 0.919805 | X3 single 0.8633; student pool 679 (491 original + 93 A + 95 B) |
| V3 | X3-best + B7 **0.892639** | 0.907426 | Router 0.885433; X3 single 0.867161; 448 labels, committee A 94 / B 113, X3 pool 655 |
| V4 | 620 rescreen final **0.869478** (best 0.858330); round-2 SAM3 LoRA 50 ep best **0.901950** | – | 580-pool S2/S3 S2 0.851530/0.851675, S3 0.841958/0.843769; single student hard 0.848592/0.854011, soft 0.851243/0.859530 |
| V5 | raw top-1 **0.870848** | 0.912017 | independent Router 0.871374; val OOF 0.848638 |
| V6 | raw top-2 **0.891226** | 0.932897 | val OOF 0.846724 |
| V7 | calibrated top-2 **0.862761** | **0.937542** | calibrated top-1 0.860331 (val 0.852949); val OOF 0.853773 |

Section map — which part of this document belongs to which version:

| Section | Version |
|---|---|
| A route generation (DINOv3 modes, and the old `vits16@224 + SAM3@512` reference table) | V1 |
| A route generation (SAM3-encoder variants, SAM3-enc @256/@1008) | V2 |
| B route-quality selectors (PQ router, candidate-invariant v2, ViT-B/256, pairwise) | V2–V3 (route engine) |
| C calibration × anchor breadth, and the P1/P2/E0 guides | V5–V7 |
| D SAM3 adaptation: fine-tune budget sweep, `ft_1pct` + PQ router | V1–V2 |
| D.4 the 50-epoch round-2 LoRA and the 10-epoch round-2 LoRA | V4 |
| D.5 round-2 full fine-tune variants and pseudo-label data quality | V2 |
| E RouteCo, F low-label baselines, G text/visual routing, H diagnostics | supporting evidence across V1–V4 |

Anchor-selection and calibration details for V5–V7 live in
[`cross_dataset_1pct.md`](cross_dataset_1pct.md) and
`mainline/reproduction_guides/automatic_selection_20260914/`.

## Protocol

| Item | Value |
|---|---|
| Split | `train=800 / validation=100 / test=100` |
| Anchors | `round(0.01 × 800) = 8`, frozen, selected by coverage greedy (`global_local_facility`) |
| Backbone | SAM3 base, `sam3.pt`, frozen unless a variant says otherwise |
| Prompt | anchor GT tight box, no text |
| Canvas | 256 (route-search descriptors at 1008) |
| Metric | per-image Dice/IoU, macro mean over all 100 test targets, no smoothing |
| Route notation | `bN` = `N` bridge frames between anchor and target; `direct` = `b0` |
| Selection rule | any selector may only pick among a **fixed candidate pool**, so the Oracle is unchanged by design |

Frozen anchor list (Kvasir): `cju2wxv0hxs2f09884w48v8fi`, `cju1c4fcu40hl07992b8gj0c8`,
`cju42xpi8lw4w0871ve317a1p`, `cju5vi4nxlc530817uoqm2m7a`, `cju3v664kh0px0818y4y7wolf`,
`cju8dpa89u6l80818dj6lldh9`, `cju2t62nq45jl0799odpufwx6`, `cju7ajnbo1gvm098749rdouk0`.

---

## A. Route generation — five feature modes × `b0–b7`

Frozen SAM3, forward-only, no selector. Each cell is mean test Dice for one fixed
route family. This is the experiment that established the stable propagation
region.

| feature mode | direct | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T18 corrected | 0.747869 | 0.775949 | 0.756490 | 0.785808 | 0.821736 | 0.816809 | 0.819200 | 0.143392 |
| DINO global pooling | 0.740844 | 0.748290 | 0.758326 | 0.786777 | 0.777536 | 0.818397 | 0.818565 | 0.092024 |
| DINO patch average | 0.728634 | 0.726893 | 0.756876 | 0.746756 | 0.799814 | 0.829179 | 0.826294 | 0.046968 |
| anchor-conditioned target pooling | 0.741306 | 0.760176 | 0.827209 | 0.840837 | 0.841258 | 0.848618 | **0.854627** | 0.032027 |
| anchor-conditioned patch correspondence | 0.757814 | 0.773501 | 0.822305 | 0.832819 | 0.833990 | 0.833577 | 0.842078 | 0.070807 |

Findings:

- The useful region is `b4–b6`; **`b7` collapses for every mode** (≤ 0.14), so the
  mainline keeps `b3–b6`.
- The two anchor-conditioned modes win; the best fixed route is
  `target pooling b6 = 0.854627`.
- Canvas 256 and canvas 512 produce **identical route hashes and ordering** for
  all five modes, so canvas differences come from SAM3 execution, not from KNN
  picking different paths.
- A ViT-B/256 rerun of the same sweep (DINOv3 features, grid 16) reproduced the
  same ordering; see `results/kvasir_1pct_anchors/stage1_feature_knn_vitb256/comparison_summary.md`.

Source: `docs/kvasir_1pct_anchor.md`, `results/kvasir_1pct_anchors/stage1_feature_knn*/`.

### SAM3-encoder route variants

Replacing the DINOv3 descriptor with SAM3's own encoder features:

| variant | descriptor | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAM3-enc @256, target pooling | patch_mean | 0.7981 | 0.8466 | 0.8700 | 0.8623 | 0.8739 | 0.8696 | 0.8740 |
| SAM3-enc @1008, target pooling | patch_mean | 0.8205 | 0.8762 | 0.8729 | 0.8701 | 0.8655 | **0.8874** | 0.8771 |
| SAM3-enc @256, patch correspondence | pooled | 0.7066 | 0.7458 | 0.8030 | 0.8652 | 0.8618 | 0.8505 | 0.8603 |
| SAM3-enc @1008, patch correspondence | pooled | 0.8202 | 0.8499 | 0.8311 | 0.8033 | 0.8451 | 0.8154 | 0.8408 |
| IMR @256 (pyramid + contrast + Qwen text) | fused | 0.7981 | 0.8572 | 0.8589 | 0.8674 | **0.8751** | 0.8679 | 0.8624 |

Sources: `results/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_{s256,s1008}/` and
`..._imr_s256/comparison_summary_*_base.md`.

---

## B. Route-quality selectors

### B.1 Propagation-quality router (frozen SAM3)

The router replaces candidate-relative weighting with an **absolute** per-route
score built from GT-free propagation features (`q_cycle`, area trajectory,
empty-mask count, connected components, centroid/box drift, adjacent-frame Dice,
SAM score trajectory).

| selector | candidate pool | Test Dice | Oracle | Oracle gap |
|---|---|---:|---:|---:|
| fixed best route | target pooling `b6` | 0.854627 | – | – |
| v2 absolute ridge router | target pooling `b0–b7` | 0.854437 | 0.898969 | 0.044531 |
| propagation-quality router | target pooling `b0–b7` | 0.872938 | 0.898969 | 0.026030 |
| propagation-quality router | target pooling `b0–b6` | 0.873599 | 0.898969 | 0.025370 |
| propagation-quality router | target + patch `b0–b6` | 0.873626 | 0.903846 | 0.030220 |
| **propagation-quality router** | **target + patch `b3–b6`** | **0.877299** | 0.896918 | 0.019619 |

Two diagnostics matter:

- Full `C0–C7` validation training did **not** close the oracle gap by itself —
  the bottleneck was not missing long-route training coverage.
- Propagation-quality features are useful: for target pooling, Top-1 Dice moves
  `0.854437 → 0.872938`.

### B.2 Candidate-invariant v2 router (per feature mode)

| feature | best fixed b | fixed Dice | selected (C7) | Oracle C7 | gain | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| t18_corrected | b4 | 0.821736 | 0.845581 | 0.897202 | +0.023845 | 0.401 |
| dino_global_pooling | b6 | 0.818565 | 0.791717 | 0.889342 | −0.026848 | 0.318 |
| dino_patch_average | b5 | 0.829179 | 0.815340 | 0.893806 | −0.013839 | 0.422 |
| anchor_conditioned_target_pooling | b6 | 0.854627 | 0.854437 | 0.898969 | −0.000189 | 0.449 |
| anchor_conditioned_patch_correspondence | b6 | 0.842078 | **0.851690** | 0.887997 | +0.009613 | 0.501 |

Unified (union) candidate-pool ceiling:

| union of modes | Oracle Dice | unique candidates/target |
|---|---:|---:|
| target pooling + patch correspondence | 0.903904 | 13.5 |
| target pooling + patch correspondence + t18 corrected | **0.923511** | 21.2 |

Source: `results/kvasir_1pct_anchors/candidate_invariant_router_v2_full_validation_b7_report/report.md`.

### B.3 ViT-B/256 route selector over an 11-variant pool

Pool = 11 route variants × `b0–b6`; propagation quality from the same checkpoint;
student audit scores reused from the 256-px X3/S2/S3 predictions; calibration and
fitting use validation only, test reported once.

| selector | base checkpoint | LoRA `p491_e20` |
|---|---:|---:|
| fixed `b6` | 0.8573 | 0.8978 |
| B7 geometric (3-student mean audit) | 0.8909 | 0.9014 |
| pure `q_model` (3-student mean) | 0.8945 | 0.8948 |
| ridge router | 0.8816 | 0.8933 |
| calibrated linear (3 signals, 3-student mean) | 0.8899 | **0.9035** |
| pool Oracle | 0.9291 | 0.9357 |

Source: `results/kvasir_1pct_anchors/route_selector_vitb256/route_selector_report_{base,lora_p491_e20}.md`.

### B.4 Pairwise route ranker

Target-grouped nested-CV audit on validation (`b3–b6`, LoRA checkpoint); the test
split was not loaded during fitting.

| method | validation OOF Dice | Δ vs `b6` | 95% CI |
|---|---:|---:|---|
| fixed `b6` | 0.844065 | — | — |
| `q_model_mean_only` | 0.847980 | +0.003915 | [−0.029, +0.035] |
| B7 mean | 0.852645 | +0.008580 | [−0.018, +0.032] |
| **pairwise quality** | **0.880443** | **+0.036378** | **[+0.011, +0.064]** |
| pairwise quality + fallback | 0.879641 | +0.035576 | [+0.011, +0.064] |
| pairwise + `q_model` | 0.877096 | +0.033031 | [+0.009, +0.060] |

Frozen-transfer test on ClinicDB (external dataset, no refit): selected Dice
**0.857032** vs fixed `b6` 0.815820, Δ **+0.041213** (95% CI [−0.005, +0.097]),
Oracle 0.912831.

Source: `results/kvasir_1pct_anchors/pairwise_ranker_vitb256/`, `results/clinicdb_external_kvasir8/`.

### B.5 Validation OOF selector matrix

Ten (canvas × feature) cells comparing the then-current selector against
route-wise z-score, explicit geometry, pairwise logistic and an "always b3"
baseline, on validation OOF only (logistic treated as a supervised diagnostic).

Representative rows (256 canvas):

| feature | current | z-score | geometry | logistic | always b3 | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| T18 corrected | **0.830202** | 0.827030 | 0.775154 | 0.812495 | 0.804913 | 0.861486 |
| DINO global | 0.805351 | 0.804522 | 0.788073 | 0.805877 | 0.791988 | 0.851277 |
| DINO patch avg | 0.806499 | 0.811376 | 0.812705 | **0.821732** | 0.806683 | 0.848212 |
| target pooling | 0.769089 | 0.764374 | 0.810430 | 0.804951 | 0.792900 | 0.858190 |
| patch correspondence | 0.783248 | 0.784134 | 0.817698 | 0.816068 | **0.828202** | 0.866338 |

Source: `results/kvasir_1pct_anchors/stage1_feature_knn_validation/OOF_selector_matrix_v2/decision_summary.md`.

---

## C. Score calibration × anchor breadth (the newest slice)

Frozen anchors, annotation-free calibration `centered(A,T) = TP(A,T) − μ_A` where
`μ_A` is the anchor's mean TP over train RGB; then `k` anchors per target, each
generating `b0–b6`. Full test results:

| config | cand/target | val OOF | test Dice | test Oracle | Oracle gap |
|---|---:|---:|---:|---:|---:|
| raw top-1 | 7 | 0.848638 | 0.870848 | 0.912017 | 0.041169 |
| calibrated top-1 | 7 | 0.852949 | 0.860331 | 0.906667 | 0.046337 |
| raw top-2 | 14 | 0.846724 | **0.891226** | 0.932897 | 0.041671 |
| calibrated top-2 | 14 | **0.853773** | 0.862761 | **0.937542** | 0.074782 |
| historical per-bridge + legacy Router | 7 | 0.851317 | 0.871374 | 0.917615 | 0.046241 |

What Kvasir shows:

- **Calibration is a negative result here** (bias ratio 0.68): the calibrated
  top-2 pool has the *highest* Oracle (0.937542) but the router realises only
  0.862761, a 0.0748 gap. The candidates improve; the legacy scorer fails to cash
  them in — on Kvasir the bottleneck is **selection**, not candidate quality.
- **Breadth is a real, significant lever**: `k=1→2` raises the full-depth ceiling
  by +0.0148 [+0.007, +0.025] (test) and +0.0370 [+0.015, +0.066] (validation),
  then decays (`2→3` +0.0104, `3→5` +0.0093, `5→8` +0.0017).
- **Depth substitutes for breadth**: the same `k=1→2` move is worth +0.0689 at
  `b0` but only +0.0148 at full depth.

Full-depth ceiling vs `k` (P2, 8 anchors × b0–b6 = 56 candidates):

| k | candidates | val raw | val cal | test raw | test cal |
|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 0.8895 | 0.9062 | 0.9180 | 0.9067 |
| 2 | 14 | 0.9265 | 0.9238 | 0.9328 | 0.9388 |
| 3 | 21 | 0.9328 | 0.9371 | 0.9432 | 0.9455 |
| 5 | 35 | 0.9400 | 0.9405 | 0.9525 | 0.9510 |
| 8 | 56 | 0.9458 | 0.9458 | 0.9542 | 0.9542 |

Budget curve (greedy anchor prefixes, full-depth Oracle):

| labelled anchors | val Oracle | test Oracle | vs K=1 |
|---:|---:|---:|---:|
| 1 | 0.8822 | 0.9142 | — |
| 2 | 0.9027 | 0.9239 | +0.0097 |
| 4 | 0.9367 | 0.9440 | +0.0299 |
| 8 (1%) | 0.9458 | 0.9542 | +0.0400 |

Bias diagnostics (E0, 0 GPU): `std(μ_A) = 0.0216`, bias ratio **0.68**, test
re-selection rate 43%, anchor concentration `0.360 → 0.220`. Correlation of
`q_cycle` with Dice decays with depth (b0 0.494 → b6 0.030).

Sources: `mainline/experiments/cross_dataset_calibration_factorial_20260914/kvasir/`,
`mainline/reproduction_guides/automatic_selection_20260914/{P1_pilot,P2_fullbank,E0_diagnostics}_20260914/`.

---

## D. SAM3 adaptation on the same 8 anchors

### D.1 Fine-tuning budget sweep (weighted 3-route selector)

| model | weighted Dice | direct | one_bridge | two_bridges | all-route mean | oracle |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.767542 | 0.819048 | 0.761372 | 0.782654 | 0.879040 |
| ft_1pct | 0.874330 | 0.831090 | 0.856687 | 0.820172 | 0.835983 | 0.902679 |
| ft_5pct | 0.882543 | 0.814541 | 0.871664 | 0.837800 | 0.841335 | 0.911454 |
| ft_10pct | 0.844562 | 0.768585 | 0.846834 | 0.815985 | 0.810468 | 0.899441 |
| ft_20pct (latest, after crash) | **0.914321** | 0.856197 | 0.901934 | 0.875345 | 0.877825 | 0.931738 |

### D.2 Longer route families (3–7 weighted routes)

| model | 3 routes | 4 | 5 | 6 | 7 | oracle 7 |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.856771 | 0.849015 | 0.861283 | 0.864143 | 0.913565 |
| ft_1pct | 0.874330 | 0.888891 | 0.894013 | 0.901913 | 0.900234 | 0.928200 |
| ft_5pct | 0.882543 | 0.891466 | 0.883936 | 0.886997 | 0.895276 | 0.938849 |
| ft_10pct | 0.844562 | 0.889791 | 0.884999 | 0.887835 | 0.901470 | 0.932109 |
| ft_20pct | **0.914321** | 0.914936 | 0.914313 | 0.914289 | 0.910493 | 0.942206 |

Fine-tuning helps; the 20% checkpoint is strongest and smaller budgets are not
strictly monotonic. Per-bridge Dice after the 1% fine-tune improves at **every**
bridge length (+0.02 to +0.05) for both route generators.

### D.3 1% fine-tune + propagation-quality router (the Kvasir best)

The `ft_1pct` checkpoint (merged video weights) propagated over the same frozen
KNN routes, with the ridge scorer retrained on ft_1pct validation features:

| scheme | selected Dice | Oracle | gap |
|---|---:|---:|---:|
| **`b3–b6` target + patch** | **0.894648** | 0.919805 | 0.025157 |
| `b3–b6` target pooling | 0.886241 | 0.903572 | 0.017331 |
| `b3–b6` patch correspondence | 0.884695 | 0.901863 | 0.017168 |
| `b0–b6` patch correspondence | 0.890865 | 0.907062 | 0.016198 |
| `b0–b6` target + patch | 0.877061 | 0.921800 | 0.044739 |

With a fine-tuned model the shorter routes also become useful, and the union
scorer is easier to confuse — hence `b3–b6` stays the setting.

### D.4 Two LoRA rounds of the SAM3 teacher

Kvasir line 1 went through **two rounds of LoRA-adapted SAM3**. Both rounds use
the same MedSAM3-style full-LoRA recipe — rank 16, alpha 32, dropout 0.1, LoRA
applied to the vision / text / geometry encoders, the DETR encoder and decoder
and the mask decoder, with the same detector loss — and differ in the training
pool and in the training length.

**Round 1 (20 epochs).** Two checkpoints that differ only in the data:

| version | LoRA training data | data dir | config |
|---|---|---|---|
| `lora_1pct_e20` | 8 GT anchors (1%) | `KvasirSEG_1pct_seed2026` | `configs/kvasir_1pct_lora.yaml` |
| `lora_p491_e20` | 8 GT + **491 pseudo labels** (first expansion pool, same quality gates) | `KvasirSEG_1pct_plus_pseudo491_seed2026` | `configs/kvasir_1pct_plus_pseudo491_lora.yaml` |

Merged video checkpoints: `work/kvasir_1pct_anchors/video_checkpoints/`
(`lora_1pct_e20_merged_video.pt`, `lora_p491_e20_merged_video.pt`; 383 of 1156
detector keys updated by the adapter merge in both cases).

Forward-only test Dice, 100 targets, canvas 256, frozen kNN routes:

| version | route mode | direct | b1 | b2 | b3 | b4 | b5 | b6 | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora_1pct_e20` | target pooling | 0.771193 | 0.797666 | 0.861002 | 0.859552 | **0.886556** | 0.874425 | 0.880978 | 0.847339 |
| `lora_1pct_e20` | patch correspondence | 0.765991 | 0.800123 | 0.844173 | 0.852210 | **0.880581** | 0.869514 | 0.864183 | 0.839539 |
| `lora_p491_e20` | target pooling | 0.790054 | 0.824923 | 0.870163 | 0.877370 | **0.887164** | 0.890885 | 0.878756 | 0.859902 |
| `lora_p491_e20` | patch correspondence | 0.799188 | 0.830049 | 0.865747 | 0.884548 | 0.883273 | 0.889516 | **0.911171** | 0.866213 |

Findings:

- Adding 491 pseudo labels to the LoRA training set improves **every bridge in
  both route modes** (+0.009 to +0.047). `patch correspondence b6` reaches
  **0.911171**, the best single fixed route in the Kvasir line.
- Both LoRA versions beat the frozen base and the full fine-tune `ft_1pct` at
  every bridge. `docs/knn_experiment_vitb256.md` has the full base-vs-LoRA delta
  table (per-cell +0.03 to +0.13, `direct` around +0.10).
- With SAM3-encoder kNN features the LoRA route is stronger still: @1008
  `target pooling cond` reaches **0.9154** (b1/b4), and @256
  `patch correspondence cond` reaches **0.9082** (b4).
- Ranking reversal: under the base checkpoint `t18_corrected` was strongest and
  anchor-conditioned modes lagged; under LoRA the anchor-conditioned
  target-pooling long bridges overtake `t18`, and the previously weakest `cond`
  kNN gains the most — the anchor-relevance signal becomes reliable after LoRA.
- The downstream line was re-run on the `lora_p491_e20` routes: fixed `b6`
  **0.8978**, B7 geometric (3-student mean) **0.9014**, calibrated linear
  **0.9035**, candidate Oracle 0.9357; pairwise-ranker validation OOF
  **0.880443**; frozen transfer to ClinicDB test **0.857032**
  (`results/clinicdb_external_kvasir8/`).
- **Caveat.** The B7 z-scores above reuse the 256-px S2/S3/X3 student predictions
  trained from the **base-SAM3** pseudo labels; the students were not re-trained
  on the LoRA teacher.

**Round 2 (10 epochs, plus one 50-epoch control).** SAM3 LoRA is re-trained from
the base on the **round-2 B7-rescreened hard pseudo labels** and then evaluated as
a direct single-image segmenter (fixed `colon polyp` text, no points/boxes, no
TP/Router/B7 at test; metrics macro-averaged per image at 256).

| pool (pseudo + GT) | epochs | best epoch | Val Dice | Test Dice (`colon polyp`) | Test Dice (empty text) |
|---|---:|---:|---:|---:|---:|
| A0 · 428 + 8 | 50 (no clip) | 19 | 0.893717 | 0.901950 | 0.890174 |
| A0 · 428 + 8 | 50 (final) | 50 | 0.873586 | 0.894245 | 0.884782 |
| A0 · 428 + 8 | **10** | 6 | 0.880909 | 0.901755 | 0.885264 |
| A1 · 596 + 8 | **10** | 4 | 0.887257 | 0.903294 | 0.870346 |
| A2 · 503 + 8 | **10** | 1 | 0.888680 | **0.912932** | 0.845776 |

Pool definitions (B7 re-screen of the TP candidates; the rule is frozen before
any training GT is read):

| pool | rule | B7 threshold | kept on validation | pool mean Dice |
|---|---|---:|---:|---:|
| A0 | B7 top candidate, subject to non-empty and total-score threshold | 0.97 | 34 | 0.934062 |
| A1 | require non-empty and return consistency `R >= 0.95` **before** picking the B7 best | 0.92 | 57 | 0.911098 |
| A2 | require non-empty and `R`, candidate consistency `C`, student consistency `S` all `>= 0.95` | 0.00 | 37 | 0.923180 |

Two-seed check at 10 epochs (same recipe, seeds 2026 / 2027):

| pool | seed2026 | seed2027 | mean |
|---|---:|---:|---:|
| A0 | 0.901755 | 0.892288 | 0.897022 |
| A1 | 0.903294 | 0.886602 | 0.894948 |
| A2 | 0.912932 | 0.884289 | 0.898611 |

Round-2 findings:

- **Round 2 does not beat round 1.** The three 10-epoch pools reach
  0.9018 / 0.9033 / 0.9129 on seed 2026, but the ordering fully reverses on
  seed 2027 (seed2026 `A2>A1>A0`, seed2027 `A0>A1>A2`). Two seeds cannot separate
  the pools; the best single number (A2 seed2026, 0.912932) is not a stable level.
- The 50-epoch A0 control peaks at epoch 19 (0.901950) and falls back to
  0.894245 by epoch 50 — the extra length buys nothing and the final weights are
  0.007705 worse than the validation best.
- The 50-epoch run also disables gradient clipping and changes the cosine
  schedule horizon, so it is **not** a clean epoch-length ablation against the
  10-epoch runs (stated in its own `experiment.md`).
- `empty text` is a pre-declared diagnostic, not a test-selected primary prompt;
  `colon polyp` is the primary metric.

Sources: `mainline/experiments/round2_*`, `mainline/round2_selection_ablation.md`,
`mainline/A0_second_training_results.md`, `mainline/A1_A2_new_training_results.md`,
`mainline/A0_A1_A2_two_seed_results.md`.

Sources: `configs/kvasir_1pct_lora.yaml`, `configs/kvasir_1pct_plus_pseudo491_lora.yaml`,
`work/kvasir_1pct_anchors/lora_experiment/test_eval/{lora_1pct_e20,lora_p491_e20}/`,
`docs/knn_experiment_vitb256.md` §7/§9, `docs/knn_experiment_sam3enc_imr_s256.md`,
`scripts/supervise_lora_p491.sh`, `scripts/merge_sam3_lora_video_checkpoint.py`.

### D.5 Round-2 full fine-tune variants and pseudo-label data quality

- Round-2 re-fine-tuning variants (`ftround2_bestval`, `ftround2_ckpt1`,
  `ftround2_ckpt6`) and the `1% GT + HQ pseudo` audit are documented in
  `docs/kvasir_1pct_anchor.md`. Headline: the HQ pseudo labels are
  protocol-clean (zero contamination) and 85% have Dice ≥ 0.9, but route-internal
  signals **cannot** filter the confidently-wrong ones (best single-feature AUROC
  ≈ 0.71); a clean same-LR HQ comparison is still pending.

Sources: `results/kvasir_1pct_anchors/summaries/`, `model_routes/`, `model_routes_max6/`.

---

## E. RouteCo-SAM3 v1

Adapter-only adaptation (memory-read adapter + mask-decoder LoRA, 400 steps):

- Roughly **neutral on the Oracle**: official-pipeline validation Oracle
  `0.8694 → 0.8614`; per-mode `−0.002 ~ −0.003`.
- Route-level behaviour is a robustness/precision trade: low-Dice routes improve,
  high-Dice routes degrade (Spearman between frozen Dice and delta ≈ −0.24).
- As a **second candidate pool** it has real upside: a perfect
  frozen-or-RouteCo per-target pick reaches `0.8778` vs frozen-only `0.8694`;
  43% of routes are better with RouteCo.
- The `S→T` gate is inert because the student is not wired in
  (`student_variance = 0`, `gate_s_to_t = 1.0`).

Source: [`routeco_sam3_v1.md`](routeco_sam3_v1.md).

---

## F. Low-label baselines on the same 8 anchors

| method | validation Dice | test Dice (100) |
|---|---:|---:|
| SynFoC UNet | 0.730004 | **0.768113** |
| SynFoC MedSAM+LoRA | 0.731031 | 0.761503 |
| SC-SAM (`sam_adpt`, vit_b) | see `results/kvasir_1pct_anchors/scsam_kvasir_1pct/` | not recorded in the retained log |
| frozen SAM3 fixed route (`target pooling b6`) | – | 0.854627 |
| frozen SAM3 propagation-quality router | – | 0.877299 |
| `ft_1pct` + propagation-quality router | – | **0.894648** |

Both baselines train a segmentation network on the same 8 labels, so they are
one-mask-per-image comparisons; the route selectors also emit one mask per image.

Source: `results/kvasir_1pct_anchors/synfoc_kvasir_1pct/summary.json`.

---

## G. Text and visual-word routing (exploratory)

These replace or augment the image descriptor with Qwen3.5-35B text and
mask-visual signals. They are kept as negative/exploratory evidence; the retained
artifacts are per-route JSON rather than a consolidated summary.

| group | what was varied | artifacts |
|---|---|---|
| Qwen text kNN | top-k text-similarity retrieval instead of image kNN | `qwen_text_knn_v1/`, `qwen_text_knn_routes/` |
| hybrid text kNN | `λ = 0.2` image + text fusion | `hybrid_text_knn_lambda02/` |
| joint original + mask + text | three-signal route scoring | `joint_original_mask_text01/` |
| mask-visual λ scan | `λ ∈ {0.0, 0.1, 0.2, 0.5, 1.0}` | `mask_visual_lambda_scan_*/` |
| original-visual λ scan | `λ ∈ {0.0, 0.1, 0.2, 0.5, 1.0}` | `original_visual_lambda_scan_*/` |
| triple visual (Qwen) | triple-route visual scan and direct tests | `triple_visual_qwen_scan_0.0/`, `triple_direct_test_mask*/` |
| conditional mask rank | mask-rank weighting `w ∈ {0, .1, .25, .5, .75, 1.0}` | `sam3enc_cond_mask_knn_v1/` |
| MSR re-route | bridge-5 re-routing variants B1–B5 | `msr_reroute_v1/` |

MSR re-route bridge-5 Dice: B1 0.873704, B2 0.875996, B3 0.868392,
**B4 0.880218**, B5 0.879385.

Background and conclusions: `docs/v2/04..08_*.md`.

---

## H. Diagnostics (0 GPU, over archived results)

| artifact | what it establishes |
|---|---|
| `c3_frozen_rules.json`, `c3_test_report.json`, `c3_validation_analysis.json` | C3 path-invariance / freeze-boundary rules |
| `e5_cycle_proxy.json`, `e5_transport_correlation.json`, `transport_proxies_v1.json`, `transport_proxies_v2_cycle.json` | how well GT-free transport proxies track real route quality |
| `consensus_medoid_analysis.json` | consensus/medoid label selection behaviour |
| `pseudo_vs_gt_lesion_paired_bootstrap.json` | paired pseudo-label vs GT lesion comparison with bootstrap intervals |
| `paper_figures/feature_focus/` | feature-focus analysis and figures `figA*`/`figB*`/`figC*` |

---

## I. Where the artifacts live

```text
docs/kvasir_program.md                       this inventory
docs/kvasir_1pct_anchor.md                   task + router reference, launch commands
docs/knn_experiment_*.md                     per-feature-mode kNN experiment notes
docs/route_selector_experiment_vitb256.md    ViT-B/256 route selector
docs/routeco_sam3_v1.md                      RouteCo v1
docs/v2/04..08_*.md                          Qwen text / mask-visual series

results/kvasir_1pct_anchors/summaries/       consolidated tables (3 files)
results/kvasir_1pct_anchors/stage1_feature_knn*/      routes + per-bridge summaries
results/kvasir_1pct_anchors/candidate_invariant_router_v*/  router summaries
results/kvasir_1pct_anchors/route_selector_vitb256/   selector reports
results/kvasir_1pct_anchors/pairwise_ranker_vitb256/  ranker reports
results/kvasir_1pct_anchors/model_routes{,_max6}/     fine-tune budget summaries
results/kvasir_1pct_anchors/paper_figures/            figures
results/clinicdb_external_kvasir8/                    external transfer with the same 8 anchors

mainline/experiments/cross_dataset_calibration_factorial_20260914/kvasir/   four-arm ablation
mainline/reproduction_guides/automatic_selection_20260914/                  P1 / P2 / E0 / guides
results/kvasir_1pct_anchors/                       raw summaries for everything above
```

## Settled vs open

**Settled.** `b4–b6` is the stable propagation region; anchor-conditioned route
generators dominate; propagation-quality features beat candidate-relative
weighting; fine-tuning helps at every bridge and the 20% budget is strongest;
anchor breadth helps with decreasing returns; depth substitutes for breadth; on
Kvasir score calibration is ineffective (bias ratio < 1) and the remaining
bottleneck is **selection**, not candidate quality.

**Open.** A unified router protocol that picks `k` on validation for every
dataset (Kvasir candidates are ready, ~0 GPU); a clean same-LR `1% GT + HQ
pseudo` comparison; wiring the student into the RouteCo `S→T` gate; and a
never-diagnosed blind test set for the coverage claim.
