# Stage 0 + Stage 1 — complete protocol, results and reproduction

> This document is the current definition of the **new** Stage 0 (automatic
> anchor mining) and Stage 1 (anchor-conditioned semantic routes → propagation →
> candidate selection) for the 1%-anchor cross-dataset study on **Kvasir-SEG,
> ISIC2018, BUSI and TN3K**.
>
> It records, for every step: what is computed, with which hyper-parameters, what
> was actually run, the results, the verification evidence, and what remains
> open. Status marker: **frozen** = decided and reproducible; **studied** =
> fully experimented but not adopted as a mandatory module; **open** = not done.
>
> Related pages: [`cross_dataset_1pct.md`](cross_dataset_1pct.md) (the same study
> told as a research narrative) · [`kvasir_versions.md`](kvasir_versions.md)
> (per-version history of the older Kvasir lines) ·
> [`kvasir_program.md`](kvasir_program.md) (full experiment inventory).

---

## 0. Status summary

| Stage | Component | Status | Evidence |
|---|---|---|---|
| 0 | Coverage-greedy automatic anchor mining | **frozen**, all 4 datasets | `SELECTIONS_FROZEN.json`, `selection_replay.json` |
| 0 | Budget rule `K = max(1, round(0.01·N_train))` | **frozen** | `policy.json` in each selection dir |
| 1 | Anchor-conditioned Target Pooling (TP) score | **frozen** | `stage1_feature_knn_routes.py` (frozen copy per experiment) |
| 1 | `b0–b6` semantic route construction, beam 32 | **frozen** | `routes.jsonl`, `ROUTES_FROZEN.json` |
| 1 | SAM3 propagation (anchor GT tight box, no text, canvas 256) | **frozen** | `propagation_quality_*.jsonl`, `*_PREDICTIONS_FROZEN.json` |
| 1 | Anchor ranking: raw TP, top-1 / top-2 | **frozen** (arms) | `pool_membership_frozen.json` |
| 1 | Anchor score calibration (`centered`) | **studied, not adopted** | `cross_dataset_calibration_factorial_20260914` |
| 1 | Candidate selection Router (legacy-28 Ridge α=1) | **frozen** | `models_frozen.json`, `test_choices_frozen.json` |
| 1 | ISIC2018 full-depth (top-5) calibration point | **open** (P3) | – |
| 1 | Random-anchor propagation control | **open** (P4) | – |
| 1 | Unified `k` refit on Kvasir | **open** (P5) | – |
| 2 | Round-1 pseudo labels → student training on the **auto-anchor** protocol | **open** | no pseudo pool / student manifest under any `automatic_anchor_*` or `tn3k_*` experiment; newest student checkpoints are 2026-09-12 and belong to the fixed-anchor protocol |

Everything below Stage 1 (student auditors, B7, SAM3 LoRA round-2) exists only on
the **old fixed-anchor protocol** — see [`kvasir_versions.md`](kvasir_versions.md)
V1–V4.

---

## 1. Datasets and budgets

| Dataset | train | validation | test | anchors `K` | actual fraction | anchor source |
|---|---:|---:|---:|---:|---:|---|
| Kvasir-SEG | 800 | 100 | 100 | 8 | 1.0000% | `automatic_anchor_selection_pilot_20260911/SELECTIONS_FROZEN.json` |
| ISIC2018 | 2075 | 259 | 260 | 21 | 1.0120% | `isic2018_auto21_tp_validation_20260911/selection/SELECTIONS_FROZEN.json` |
| BUSI | 517 | 64 | 66 | 5 | 0.9671% | `busi_auto5_tp_1pct_20260913/SELECTED_SUPPORT_FROZEN.json` |
| TN3K | 2303 | 576 | 614 | 23 | 0.9987% | `tn3k_busi_factorial_20260915/SELECTED_SUPPORT_FROZEN.json` |

Budget rule (frozen): `K = max(1, round(0.01 · N_train))`.

```text
K = max(1, round(0.01 * N_train))
Kvasir    : round(8.00)  =  8
ISIC2018  : round(20.75) = 21
BUSI      : round(5.17)  =  5
TN3K      : round(23.03) = 23
```

Splits are never re-drawn: Kvasir 800/100/100, ISIC2018 2075/259/260, BUSI
517/64/66, TN3K official fold0 2303/576/614 (TN3K test is the official 614-image
test set). Validation GT is **extra** supervision used only to fit the Router;
test GT is read only after the selection is frozen.

---

## 2. Stage 0 — automatic anchor mining

### 2.1 What may and may not be read

**Allowed:** train RGB only (plus the frozen SAM3-base checkpoint).

**Forbidden during selection:** GT masks, pseudo masks, anchor-conditioned
descriptors, validation/test features or metrics, class names / semantic labels.

The anchor list is frozen — with a SHA256 — **before** the selected images' GT is
opened. Each selection directory carries a `policy.json` asserting
`selected_labels_only_after_freeze: true`, `train_hidden_masks_used: false`, and
`test_used: false`.

### 2.2 Image descriptor (1008)

```python
# 1. RGB, bicubic + antialias resize to 1008x1008, scale to [0,1], then (x-0.5)/0.5
# 2. frozen SAM3-base: model.detector.backbone.vision_backbone.trunk, first output
#    72 x 72 patches, 1024 dims each
# 3. global descriptor
patches = l2_normalize(patches, axis=-1)         # per patch
g = l2_normalize(patches.mean(axis=0))           # mean of NORMALISED patches, then L2
# 4. local descriptor: 64 sampled patches per image
rows = np.linspace(4, 67, 8).round().astype(int)  # also used for cols
idx  = row * 72 + col                             # row-major
```

Selection resolution is 1008; it is **not** interchangeable with the canvas-256
used for KNN and propagation.

### 2.3 Local visual-word dictionary and IDF

```python
# all train images' 64 local vectors, numpy-L2-normalised, stacked to (N*64, 1024)
MiniBatchKMeans(n_clusters=64, random_state=2026, n_init=3,
                batch_size=2048, max_iter=100)      # all other params = sklearn defaults
hist[i] = counts of the 64 sampled vectors over the 64 clusters / 64
df[c]   = number of train images with hist[c] > 0
idf[c]  = log((N + 1) / (df[c] + 1)) + 1
local[i]  = l2_normalize(sqrt(hist[i] * idf[i]))
global[i] = l2_normalize(global_feature[i])
S = 0.5 * clip(global @ global.T, 0, 1) + 0.5 * clip(local @ local.T, 0, 1)
```

Kept in the original numpy dtype (no forced float32 on the IDF), because that can
flip near-tie decisions.

### 2.4 Greedy coverage objective

```text
train rows sorted by ID; candidates whose file SHA256 is identical keep only the
first occurrence
best = zeros(N, float32)
each step:
    gain[j] = mean_i( max(S[i, j] - best[i], 0) )
    j*      = argmax gain[j]   over unselected, non-duplicate candidates
    best[i] = max(best[i], S[i, j*])
repeat K times
```

`F(A) = (1/N) Σ_i max_{a∈A} S(i,a)` is monotone submodular, so greedy is
`(1 − 1/e)`-optimal for the coverage objective. `np.argmax` breaks ties to the
earliest index, so the train ID order matters. The coverage target includes
**all** train images, not only the unselected ones. Main scheme name:
`global_local_facility`.

### 2.5 Coverage results and random control

Coverage is measured over the non-selected images and is a **proxy, not Dice**.

Kvasir (792 non-selected images) —
`automatic_anchor_selection_pilot_20260911/report.md`:

| scheme | global | local | combined | max-anchor share | auto-stop |
|---|---:|---:|---:|---:|---:|
| existing 8 | 0.986559 | 0.688024 | 0.835373 | 31.37% | – |
| `global_facility` | 0.988960 | 0.694420 | 0.839891 | 20.12% | 14 |
| **`global_local_facility`** | 0.987532 | **0.721408** | **0.852560** | **15.50%** | 11 |
| random ×100 mean (seed 2026) | 0.986493 | 0.679539 | 0.830501 | 24.20% | – |

ISIC2018 (2054 non-selected images) —
`isic2018_auto21_tp_validation_20260911/selection/report.md`:

| scheme | global | local | combined | max-anchor share | auto-stop |
|---|---:|---:|---:|---:|---:|
| existing 21 | 0.982047 | 0.684837 | 0.829741 | 11.04% | – |
| `global_facility` | 0.985333 | 0.683178 | 0.830714 | 7.71% | 14 |
| **`global_local_facility`** | 0.982873 | **0.723292** | **0.850002** | 10.94% | 10 |
| random ×100 mean | 0.981518 | 0.681729 | 0.827747 | 12.44% | – |

BUSI: the frozen selection dir keeps the list and the descriptors; the budget
policy is `{"budget": 5, "train": 517, "actual_fraction": 0.009671..., "seed": 2026}`
with `"selection": "same train-only global_local_facility, RGB1008, 64 spatial
patch tokens, 64-cluster dictionary, equal global/local similarity"`.

Auto-stop rule (exploratory, unchanged across datasets): stop after 3 consecutive
additions whose coverage gain is below 1% of the remaining mean single-anchor
distance; at most 32 (Kvasir) / 64 (ISIC) anchors are observed. The threshold was
never validated on segmentation, so these numbers are **not** "the minimum
sufficient label count".

### 2.6 Frozen anchor lists

Kvasir (8, greedy order):

```text
kvasir-seg::cju2wxv0hxs2f09884w48v8fi
kvasir-seg::cju1c4fcu40hl07992b8gj0c8
kvasir-seg::cju42xpi8lw4w0871ve317a1p
kvasir-seg::cju5vi4nxlc530817uoqm2m7a
kvasir-seg::cju3v664kh0px0818y4y7wolf
kvasir-seg::cju8dpa89u6l80818dj6lldh9
kvasir-seg::cju2t62nq45jl0799odpufwx6
kvasir-seg::cju7ajnbo1gvm098749rdouk0
```

ISIC2018 (21, greedy order):

```text
ISIC_0007788  ISIC_0009941  ISIC_0013410  ISIC_0014026  ISIC_0001126
ISIC_0010023  ISIC_0015614  ISIC_0014438  ISIC_0002453  ISIC_0011229
ISIC_0013458  ISIC_0015995  ISIC_0009252  ISIC_0010263  ISIC_0014829
ISIC_0013399  ISIC_0012453  ISIC_0011129  ISIC_0015206  ISIC_0016048
ISIC_0010360
```

BUSI (5):

```text
BUSI::benign (3)   BUSI::benign (125)   BUSI::benign (305)
BUSI::malignant (195)   BUSI::malignant (187)
```

Benign/malignant names played no part in selection — the 3 + 2 split is a
post-hoc statistic. TN3K: 23 anchors, list in
`tn3k_busi_factorial_20260915/SELECTED_SUPPORT_FROZEN.json`.

### 2.7 Frozen-GT morphology audit

| dataset | scheme | GT area fraction range | median |
|---|---|---|---|
| Kvasir | existing 8 | 0.47% – 10.20% | 4.41% |
| Kvasir | `global_facility` | 6.13% – 47.38% | 18.92% |
| Kvasir | `global_local_facility` | 1.71% – 22.72% | 10.35% |
| ISIC2018 | existing 21 | 1.46% – 81.13% | 23.92% |
| ISIC2018 | `global_facility` | 1.71% – 66.76% | 14.32% |
| ISIC2018 | `global_local_facility` | 1.30% – 44.63% | 8.94% |

The audit is descriptive; it did **not** feed back into selection.

### 2.8 Reproduction and verification

Deterministic replay of the dictionary, IDF, greedy decisions and the final list
was performed on all three pilot datasets
(`mainline/reproduction_guides/automatic_selection_20260914/verify_*/selection_replay.json`):

| dataset | `exact_match` | order identical | `mask_pixels_read` | eligible | dictionary max abs | histogram max abs |
|---|---|---|---|---|---|---|
| Kvasir | **true** | yes | **0** | 800 | 0.0 | 0.0 |
| ISIC2018 | **true** | yes | **0** | 2075 | 0.0 | 0.0 |
| BUSI | **true** | yes | **0** | 516 | 0.0 | 0.0 |

BUSI's eligible count is 516 (not 517) because `benign (433)` and
`malignant (145)` are exact byte duplicates and only the first occurrence is
selectable; all train samples still contribute to the coverage objective and to
the dictionary/IDF fit. TN3K has the same property via `dataset_audit.json`.

Replay entry point: `mainline/reproduction_guides/automatic_selection_20260914/reproduce_automatic_selection.py`
(`extract` / `select` / `prepare` / `evaluate`).

---

## 3. Stage 1 — semantic routes, propagation and selection

### 3.1 Substrate: 256 features and Target Pooling

```text
KNN/TP input        : 256 x 256
patch grid          : 256 // 14 = 18  ->  18 x 18 = 324 patches, 1024 dims each
KNN similarity      : whole-image patch_mean cosine
                      (patch_mean = L2 of the mean of L2-normalised patches)
anchor prototype p_A: GT nearest-resized to the 18x18 grid, mean of the
                      foreground-position patches, L2-normalised
                      (empty foreground grid -> fall back to the whole-image
                      patch mean)
s_j                 : dot(p_A, x_j)
w_j                 : softmax(10 * s_j)          # exp((sims - max) * 10) form
z(A,T)              : L2_normalize(sum_j w_j * x_j)
TP(A,T)             : dot(p_A, z(A,T))
```

Only the `cond_target` (Target Pooling) family is used — the Patch
Correspondence family belongs to the older Kvasir V1/V2 lines and is **not** part
of this Stage 1.

### 3.2 Route construction (`b0–b6`)

```text
b0 : anchor A -> target T
b1 : A -> train bridge 1 -> T
...
b6 : A -> train bridge 1 -> ... -> train bridge 6 -> T

per b, independent beam search, beam width 32
node scores  : TP(bridge | A) for every intermediate node and the target
edge scores  : patch_mean cosine between consecutive nodes
path score   : (min(values), mean(values)) maximised lexicographically
```

Bridge frames come **only from the train split**, so a target can never appear in
its own bridge chain. The route records the KNN path only; SAM3 executes it.

### 3.3 Prompt and propagation

```text
prompt        : the anchor's GT tight box (human_pool(support, 512))
text          : none
canvas        : 256 (SAM3 internally still processes at its native resolution)
checkpoint    : frozen SAM3-base sam3.pt
              SHA256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
object choice : the historical highest-score object-selection rule
forward mask  : the target frame's binary mask is the candidate
return        : the PREDICTED target mask (never target GT) initialises the
                backward propagation to the anchor; Dice against the anchor GT
                is recorded as q_cycle and used only as a quality feature
```

Nothing is filtered by Dice or by `q_cycle` when building the test pool: every
test image is scored, and the macro mean covers all of them.

### 3.4 Anchor–target matching

Two ranking scores exist; both are computed **before** propagation and need no
labels:

```text
raw(A,T)      = TP(A,T)
mu_A          = mean_{x in train RGB} TP(A, x)          # no GT read
centered(A,T) = TP(A,T) - mu_A
```

The centred value is used **only to rank anchors**. Inside a path the original
TP and the original adjacent patch-mean cosine still drive the
`(bottleneck, mean)` lexicographic score — mixing near-zero centred values with
~0.8-scale edge scores would change the path objective.

### 3.5 Top-K anchor retention

| arm | ranking | anchors kept per target | candidates per target |
|---|---|---:|---:|
| `raw_top1` | raw TP | 1 | 7 |
| `centered_top1` | centred TP | 1 | 7 |
| `raw_top2` | raw TP | 2 | 14 |
| `centered_top2` | centred TP | 2 | 14 |
| `original_per_bridge` | historical: per `(target, b)` best path across **all** anchors | varies per b | 7 |

Ties are broken by ascending anchor ID; the Router's candidate order is
`(anchor rank, bridge length)` and ties in predicted score take the earliest
candidate. `raw_top1` and `original_per_bridge` are **different rules** and must
not be conflated.

### 3.6 Candidate pools actually built

| dataset | pool | validation candidates | test candidates | notes |
|---|---|---:|---:|---|
| Kvasir | five-arm union | 1942 | 1862 | plus the P2 full bank: 8 × 7 = 56 per target (11 200 propagations) |
| ISIC2018 | five-arm union | 5547 | 5437 | the 21-anchor full bank was not built |
| BUSI | seven-candidate arms | 448 | 462 | calibrated multi-anchor run recorded 1183 rows (66 targets) including reused old candidates; the selected pool is 66 × 14 = 924 |
| TN3K | propagated union | 15 554 (27.003/target) | 16 485 (26.849/target) | full pool would be 23 × 7 = 161 per target |

Propagation workload reference (Kvasir, historical `seconds` per target):
`raw_top2` 31.469 vs `original_per_bridge` 15.624 — 7 → 14 roughly doubles the
task count. This is a workload reference, not a controlled end-to-end timing.

### 3.7 The Router

```text
features (legacy-28)
  7 base   : bridge_count, path_bottleneck_similarity, path_mean_similarity,
             final_sam_score, final_candidate_count,
             bridge×bottleneck, bridge×path_mean
  21 prop  : q_cycle, cycle_success, cycle_sam_score,
             trace_area_min / _max / _final / _max_rel_delta,
             trace_empty_count, trace_component_max / _final,
             trace_centroid_max_step,
             trace_bbox_w_max_rel_delta, trace_bbox_h_max_rel_delta,
             trace_adjacent_dice_min / _mean / _last,
             trace_sam_score_min / _mean / _final,
             trace_candidate_count_max / _final
model      : standardised Ridge(alpha = 1) regressing candidate Dice
validation : image-grouped 5-fold, seed 2026; all 7 (or 14) candidates of one
             target always share a fold; standardisation refit inside each fold
selection  : one configuration chosen by OOF Dice, then refit on all validation
nested     : outer-5 / inner-4 re-run of the whole configuration choice
freeze     : models_frozen.json -> test_choices_frozen.json -> THEN read test GT
```

`rank_peer` (the alternative) adds 8 inter-candidate features (mean / min / max /
std of Dice against the other candidates, area, absolute difference from the
median area, log area ratio, candidate-consistency × return-consistency), centres
features and Dice labels **within each target's candidate set**, and compares
alpha = 1 / 10 / 100.

The Router may only choose among existing candidates, so it can never change the
Oracle.

---

## 4. Stage-1 results

### 4.1 Per-bridge test Dice (fixed rank-1 / historical per-bridge arms)

"fixed rank-1" means one anchor per target across all depths; the
`original_per_bridge` row is the historical per-depth rule. Values are macro means
over the full test split.

Kvasir (100), `verify_tp_kvasir/test_results.json`:

| | b0 | b1 | b2 | b3 | b4 | b5 | b6 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dice | 0.804710 | 0.863804 | 0.858386 | 0.862556 | 0.874749 | 0.865332 | **0.887069** | **0.917615** |
| IoU | 0.740799 | 0.802767 | 0.793706 | 0.798998 | 0.812229 | 0.801673 | 0.825453 | – |

ISIC2018 (260), `verify_tp_isic2018/test_results.json`:

| | b0 | b1 | b2 | b3 | b4 | b5 | b6 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dice | 0.837421 | 0.853724 | **0.856643** | 0.854948 | 0.852281 | 0.851927 | 0.852864 | **0.884316** |
| IoU | 0.758924 | 0.773610 | 0.777967 | 0.778093 | 0.775312 | 0.773758 | 0.774872 | – |

BUSI (66), `verify_tp_busi/test_results.json`:

| | b0 | b1 | b2 | b3 | b4 | b5 | b6 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dice | 0.458291 | 0.514197 | 0.514843 | 0.533733 | 0.517095 | 0.514702 | **0.539444** | **0.617141** |
| IoU | 0.375306 | 0.429142 | 0.431707 | 0.446990 | 0.432910 | 0.435170 | 0.454572 | – |

TN3K (614), `tn3k_busi_factorial_20260915/results.json` — the same target set,
two different single-anchor rankings:

| arm | b0 | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw rank-1 | 0.431451 | 0.452698 | 0.471609 | 0.439093 | 0.430539 | 0.443845 | 0.444718 |
| **centred rank-1** | 0.451997 | 0.483300 | 0.479944 | **0.497898** | 0.490611 | 0.496646 | 0.493195 |
| `original_per_bridge` | 0.431333 | 0.453160 | 0.471632 | 0.439899 | 0.431075 | 0.443004 | 0.441576 |

On TN3K the centred single-anchor ranking is better at **every** depth, unlike
Kvasir where it is worse at every depth.

Validation Oracle with the **frozen, unfitted** historical Router (development
descriptive only):

| dataset | anchors | val fixed b0–b6 | val Oracle |
|---|---|---|---:|
| Kvasir | original 8 | 0.773972 → 0.842460 (b1–b6: 0.808410 / 0.817428 / 0.808974 / 0.811338 / 0.820067 / 0.842460) | 0.869544 |
| Kvasir | **automatic 8** | 0.797777 / 0.829016 / 0.828908 / 0.837240 / 0.843508 / 0.844415 / 0.850480 | **0.889347** |
| ISIC2018 | original 21 | 0.721429 / 0.714929 / 0.713285 / 0.726206 / 0.718503 / 0.731877 / 0.738695 | 0.772915 |
| ISIC2018 | **automatic 21** | 0.847336 / 0.859682 / 0.866458 / 0.866253 / 0.865009 / 0.868265 / 0.868583 | **0.887467** |
| BUSI | automatic 5 | 0.494374 / 0.560873 / 0.590754 / 0.608827 / 0.620961 / **0.644838** / 0.604237 | 0.675247 |

Automatic anchors raise the ceiling on every dataset. On Kvasir the unfitted
router then realises less (0.851917 → 0.840154), which is why the Router is
always **refit** before reporting.

### 4.2 Independent Router, with configuration selection

`automatic_anchor_tp_router_20260913/{kvasir,isic2018}/results.json`,
`busi_auto5_tp_router_20260913/report.md`:

| dataset | selected configuration | val OOF | nested OOF | test Dice | test IoU | fixed-depth test | Oracle | Oracle gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Kvasir | `legacy_a1` | **0.851317** | 0.832701 | **0.871374** | 0.809810 | b6 0.887069 | 0.917615 | 0.046241 |
| ISIC2018 | `rank_peer_a100` | **0.870530** | 0.864335 | **0.866363** | 0.786488 | b6 0.852864 | 0.884316 | 0.017953 |
| BUSI | `legacy_a1` | **0.617634** | 0.609696 | **0.566808** | 0.480026 | val-b5 0.514702 | 0.617141 | 0.050333 |

Per-configuration OOF values:

- Kvasir: `legacy_a1` 0.851317, `rank_peer_a1` 0.829937, `rank_peer_a10` 0.834230,
  `rank_peer_a100` 0.835469 — the legacy configuration wins and the nested value
  is lower, i.e. configuration selection is unstable.
- ISIC2018: `legacy_a1` 0.862396, `rank_peer_a1` 0.862954, `rank_peer_a10`
  0.866155, `rank_peer_a100` 0.870530. The refit legacy Router scores 0.866573 on
  test, i.e. **+0.000210 above** the selected `rank_peer_a100` (0.866363):
  the pre-registered validation choice is reported, and no claim is made that the
  new ranking is better.

Other Kvasir test references: highest return consistency 0.872943, highest
candidate-pool consistency 0.876956, fixed validation-chosen `b6` 0.887069. The
selected Router is **1.5695 points below** fixed `b6`, with a paired interval
[−0.037478, +0.001524] that cannot prove a router benefit.

Additional BUSI references: highest return consistency 0.530451, highest
candidate-pool consistency 0.547502. Its Router beats the validation-chosen fixed
`b5` by 5.2106 points ([0.012606, 0.101015]) — but on validation the best fixed
depth (b5, 0.644838) beats the Router, so this is not a stable cross-split gain.

### 4.3 Four-arm calibration ablation (test Dice)

`cross_dataset_calibration_factorial_20260914`, `busi_calibration_factorial_20260914`,
`tn3k_busi_factorial_20260915`:

| Method | cand/target | Kvasir (100) | ISIC2018 (260) | BUSI (66) | TN3K (614) |
|---|---:|---:|---:|---:|---:|
| raw top-1 | 7 | 0.870848 | 0.868228 | 0.565990 | 0.519585 |
| centred top-1 | 7 | 0.860331 | 0.871717 | **0.719992** | **0.556705** |
| raw top-2 | 14 | **0.891226** | 0.869276 | 0.656084 | **0.569973** |
| centred top-2 | 14 | 0.862761 | **0.875864** | **0.752198** | 0.560661 |
| historical per-bridge + legacy Router | 7 | 0.871374 | 0.866573 | 0.566808 | 0.515486 |

Candidate-pool ceiling and the gap left to selection:

| Method | Kvasir Oracle / gap | ISIC Oracle / gap | BUSI Oracle / gap | TN3K Oracle / gap |
|---|---|---|---|---|
| raw top-1 | 0.912017 / 0.0412 | 0.883608 / 0.0154 | 0.617141 / 0.0512 | 0.606576 / 0.0870 |
| centred top-1 | 0.906667 / 0.0463 | 0.890596 / 0.0189 | 0.775461 / 0.0555 | 0.615749 / 0.0590 |
| raw top-2 | 0.932897 / 0.0417 | 0.903881 / 0.0346 | 0.776347 / 0.1203 | 0.699971 / 0.1300 |
| centred top-2 | **0.937542** / 0.0748 | **0.911909** / 0.0360 | **0.823649** / 0.0715 | **0.701492** / 0.1408 |
| historical per-bridge | 0.917615 / 0.0462 | 0.884316 / 0.0177 | 0.617141 / 0.0503 | 0.608208 / 0.0927 |

Validation OOF per dataset: Kvasir 0.848638 / 0.852949 / 0.846724 / 0.853773 /
0.851317; ISIC2018 0.861163 / 0.876904 / 0.868854 / 0.880950 / 0.862396; BUSI
0.616268 / 0.728772 / 0.665683 / 0.738958 / 0.617634; TN3K 0.510819 / 0.500433 /
0.547491 / 0.552716 / 0.511450.

### 4.4 Paired per-target comparisons

All intervals are 10 000-image paired bootstrap, seed 2026, exploratory, no
multiplicity correction.

| comparison | Kvasir (test) | ISIC2018 (test) | BUSI (test) | TN3K (test) |
|---|---|---|---|---|
| centred top-1 − raw top-1 | −0.010517 [−0.0315, +0.0047] | +0.003489 [−0.0077, +0.0147] | **+0.154001 [+0.0766, +0.2364]** | **+0.037120 [+0.0094, +0.0654]** |
| centred top-2 − raw top-2 | −0.028465 [−0.0658, +0.0045] | +0.006588 [−0.0036, +0.0186] | **+0.096114 [+0.0213, +0.1740]** | −0.009312 [−0.0353, +0.0163] |
| raw top-2 − raw top-1 | +0.020378 [−0.0011, +0.0471] | +0.001048 [−0.0123, +0.0131] | +0.090094 [+0.0177, +0.1655] | **+0.050388 [+0.0319, +0.0691]** |
| centred top-2 − centred top-1 | +0.002430 [−0.0246, +0.0298] | +0.004147 [−0.0067, +0.0148] | +0.032206 [−0.0147, +0.0811] | +0.003956 [−0.0151, +0.0231] |
| centred top-2 − historical control | −0.008614 [−0.0401, +0.0206] | +0.009291 [−0.0010, +0.0196] | **+0.185389 [+0.1043, +0.2709]** | **+0.045175 [+0.0202, +0.0702]** |
| interaction | −0.017948 [−0.0532, +0.0150] | +0.003099 [−0.0103, +0.0176] | −0.057888 [−0.1394, +0.0163] | **−0.046432 [−0.0737, −0.0192]** |

ISIC2018 **validation** is significant where test is not: centred top-1 − raw
top-1 **+0.015742 [+0.000417, +0.031808]**, centred top-2 − raw top-2
**+0.012096 [+0.000883, +0.024890]**, centred top-2 − historical control
**+0.018554 [+0.004823, +0.033475]**.

**Four regimes.** Calibration is not one module with one verdict:

| dataset | centred top-1 | centred top-2 | interaction | reading |
|---|---|---|---|---|
| BUSI | **+0.154 significant** | **+0.096 significant** | −0.058 (ns) | raw score systematically picks the wrong anchor |
| TN3K | **+0.037 significant** | −0.009 (ns) | **−0.046 significant** | calibration and breadth substitute |
| ISIC2018 | +0.003 (ns at test, **significant at validation**) | +0.007 (ns at test) | +0.003 | directionally positive, underpowered at test |
| Kvasir | −0.011 (ns) | −0.028 (ns) | −0.018 | no benefit; the bottleneck is selection |

### 4.5 The split anchor-bias diagnostic

```text
bias_ratio = std_A(mu_A) / std_T( max_A s(A,T) )
```

`mu_A` is computed from train RGB only, so the diagnostic needs **no labels**.

| dataset | anchors | std(mu_A) | **bias ratio** | test re-selection | concentration raw → cal | raw top-1 ties |
|---|---:|---:|---:|---:|---|---:|
| Kvasir | 8 | 0.0216 | **0.68** | 43% | 0.360 → 0.220 | 12% |
| ISIC2018 | 21 | 0.0449 | **1.10** | 65% | 0.235 → 0.158 | 9% |
| BUSI | 5 | 0.1448 | **5.42** | 85% | **0.985 → 0.348** | 0% |
| TN3K | 23 | not published | not published | 100% | not published | not published |

Validation re-selection rates: Kvasir 56%, ISIC2018 66%, BUSI 80%.

Calibration gain vs bias ratio:

| dataset | bias ratio | ΔDice (b0, k=1) | ΔDice (b0, k=2) | ΔDice (full depth, k=1) |
|---|---:|---:|---:|---:|
| BUSI | 5.42 | **+0.2063 [+0.116, +0.302]** | **+0.1159 [+0.051, +0.191]** | **+0.0788 [+0.022, +0.141]** |
| ISIC2018 | 1.10 | +0.0103 [−0.009, +0.030] | +0.0115 [−0.001, +0.025] | **open (P3)** |
| Kvasir | 0.68 | −0.0239 [−0.078, +0.029] | −0.0045 [−0.028, +0.022] | val +0.0167 / test −0.0113 (CI spans 0) |

The rule is monotone over three datasets but rests on **two full-depth points**
only, and TN3K shows that a large train-mean span is not sufficient: its span is
0.5502 (BUSI 0.3498) yet its top-2 calibration gain is not significant, because
the raw score degenerates onto the anchor that is already the per-bridge best.

### 4.6 Top-K anchor retention curves

Kvasir full bank, 8 anchors × `b0–b6` = **56 candidates per target**, 11 200
propagations (`P2_fullbank_20260914/P2_REPORT.md`); the original 700 test routes
reproduce bit-exactly (difference 0.00e+00):

| k | candidates | val raw | val cal | test raw | test cal |
|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 0.8895 | 0.9062 | 0.9180 | 0.9067 |
| 2 | 14 | 0.9265 | 0.9238 | 0.9328 | 0.9388 |
| 3 | 21 | 0.9328 | 0.9371 | 0.9432 | 0.9455 |
| 5 | 35 | 0.9400 | 0.9405 | 0.9525 | 0.9510 |
| 8 | 56 | 0.9458 | 0.9458 | 0.9542 | 0.9542 |

Adjacent-`k` ceiling gains (paired bootstrap): validation +0.0370 / +0.0063 /
+0.0072 / +0.0058 for k = 1→2 / 2→3 / 3→5 / 5→8; test +0.0148 / +0.0104 /
+0.0093 / +0.0017. All significantly positive, decaying fast.

Budget curve (greedy anchor prefixes, full-depth Oracle): K = 1 → val 0.8822 /
test **0.9142**; K = 2 → 0.9027 / 0.9239; K = 4 → 0.9367 / 0.9440; K = 8 →
0.9458 / 0.9542. One labelled image already reaches a 0.9142 test **ceiling**; the
full 1% adds +0.040.

TN3K `Top-K` retention (`tn3k_reference_ranking_pilot_20260916/参考排序诊断.md`,
64 validation images, 23 anchors, `b0` only):

| ranking | Top-1 Dice | Top-2 Oracle | Top-4 Oracle | Top-8 Oracle | Top-16 Oracle |
|---|---:|---:|---:|---:|---:|
| raw | 0.535822 | 0.638288 | 0.734715 | 0.788503 | 0.827042 |
| centred | 0.440338 | 0.617872 | 0.739983 | 0.787178 | 0.820704 |
| z-score | 0.493462 | 0.632730 | 0.740044 | 0.794240 | 0.818349 |
| train percentile | 0.520108 | 0.636906 | 0.729553 | 0.795967 | 0.818887 |
| global cosine | 0.399361 | 0.576907 | 0.693413 | 0.752892 | 0.805157 |
| half z-score + global | 0.510129 | 0.650797 | 0.746081 | 0.785715 | 0.816455 |
| anchor-quality prior (OOF) | 0.478807 | 0.661088 | 0.748840 | 0.791713 | 0.823418 |
| ridge10 numeric (OOF) | **0.597020** | 0.698505 | 0.763641 | 0.802549 | 0.822563 |
| ridge10 numeric + anchor one-hot | 0.550992 | 0.686576 | 0.757835 | 0.796070 | 0.824670 |
| all-23 `b0` Oracle | – | – | – | – | **0.828171** |

The learned 16-d matching score improves Top-1 by +0.061199 over raw TP, but the
paired interval [−0.020469, 0.143866] crosses 0. Every simple re-ranking
(re-centring, z-score, percentile, global cosine) fails to beat raw TP, and an
anchor-quality prior alone scores only 0.478807 — "always pick the on-average
best anchor" is not the answer.

### 4.7 Three levers substitute for each other

| dataset / split | k=1→2, b0 only | k=1→2, full depth |
|---|---:|---:|
| Kvasir test | +0.0689 | +0.0148 |
| Kvasir validation | +0.0899 | +0.0370 |
| BUSI validation | +0.1603 | +0.1079 |
| ISIC2018 test | +0.0389 | open (P3) |
| ISIC2018 validation | +0.0366 | open (P3) |

Depth also compensates for a bad anchor: BUSI's calibration gain falls from
+0.2063 at `b0` to +0.0788 at full depth (~40%), and decays to zero by k = 3.
Depth selection itself has only one genuine per-target signal, `q_cycle`: it wins
on ISIC (0.861732 vs 0.856643) and loses on Kvasir (0.872898 vs 0.887069) and
BUSI (0.530451 vs 0.539444). `path_bottleneck` barely varies with depth and
`path_mean` rises monotonically, so their argmax is effectively "always take b6".

---

## 5. Diagnostics that explain Stage-1 behaviour

### 5.1 BUSI — the positive case for calibration

The raw TP score sends **65 of 66 `b0` targets** (and 455 of 462 paths) to a
single anchor, `malignant (187)`; its train mean is the highest (0.8607) but its
`b0` validation Dice is the lowest (0.4817), while the nearly discarded
`benign (3)` (mean 0.5110) reaches 0.6054–0.6064. On the unselected BUSI
validation bank, `rho(raw, Dice) = +0.139` pooled vs `rho(centred, Dice) = +0.380`;
the anchor-choice step moves from 0.5923 to 0.6921 (+0.0998 [0.0317, 0.1702]).

Against a trained low-label baseline on the same 5 labels and split
(`results/busi_1pct_protocol/result.md`): SynFoC MedSAM+LoRA (validation-selected)
0.653120, SynFoC UNet (not selected) 0.683123, raw top-1 0.565990, raw top-2
0.656084, centred top-1 0.719992, **centred top-2 + Router 0.752198**, centred
top-2 Oracle 0.823649. The UNet branch is higher on test but validation did not
select it, so it is not switched on test.

### 5.2 Kvasir — calibration is null, selection is the bottleneck

The centred top-2 pool has the **highest Oracle of all four arms** (0.937542 vs
raw top-2's 0.932897) yet the legacy Router realises only 0.862761, leaving the
largest gap in the table (0.0748). The centred rank-1 single-anchor per-bridge
Dice is **lower than raw rank-1 at every depth**. Bias ratio 0.68 < 1.

### 5.3 ISIC2018 — significant on validation, not at test

Full validation (259 targets) supports calibration at both depths; every test
interval crosses 0 and the test lead over the historical Router is only
+0.9291 pp. The full-depth (top-5) point was never measured — this is P3.

### 5.4 TN3K — scale mismatch, and the pool does contain good answers

TN3K's absolute level is far below BUSI's, but the gap is mostly **tail
composition** (`why_gap_vs_busi.md`): 14-candidate Oracle 0.7015 vs BUSI 0.8236;
share of "reachable" targets (Oracle ≥ 0.5) **76.7% vs 92.4%**; Oracle on
reachable targets 0.8493 vs 0.8794 — only 0.030 apart. Of the 0.1221 Oracle gap,
≈0.099 is tail composition and ≈0.023 is quality on reachable targets.

- **Failure mode is undersegmentation**, not oversegmentation: on unreachable
  targets the median prediction/GT area ratio is 0.211 and 76.2% are below GT.
  (The earlier "oversegmentation" reading came from the mean; the report carries
  the errata.)
- **Root cause: anchor/target nodule scale mismatch.**
  `corr(|log(anchor area / target area)|, Dice) = −0.4776`; failure group
  anchor/target area ratio median **0.158** vs **0.641** for reachable targets.
  Anchors *larger* than the target are fine (Dice 0.865) — the failure is
  one-sided. Per-anchor failure rate is monotone in anchor nodule size.
- **The obvious fix is falsified.** Enlarging the prompt box around its centre
  monotonically hurts (`tn3k_boxscale_probe_20260916/probe_result.json`):
  s = 1.00 → 0.427296, 1.25 → 0.391002, 1.50 → 0.359899, 2.00 → 0.246073,
  3.00 → 0.246137. The box is geometrically correct **on the anchor**; enlarging
  it distorts the anchor prompt. "Box too small" is a symptom, not the cause.
- **The pool has good answers; ranking loses them.** On the same 64 validation
  images: target's **own** GT box → **0.854998**; all-23-anchor `b0` Oracle →
  **0.828171**; raw top-2 `b0` Oracle 0.638288; centred top-2 `b0` Oracle
  0.617872; raw top-1 `b0` 0.535822. Only **1 of 64** targets has all 23 anchors'
  `b0` below 0.5.
- The automatic selector optimises **appearance** coverage and has **no scale
  term**, so the chosen anchors are not scale-representative.

### 5.5 What Stage-1 diagnostics imply

1. Report **both the full set and the reachable subset**, or the method is
   understated.
2. The binding constraint on TN3K is a **scale-blind ranking**. Candidate fixes,
   in order of directness: (a) add a scale-compatibility term to the ranking;
   (b) make Stage 0 cover **scale** as well as appearance; (c) retain more
   anchors — the all-23 `b0` Oracle (0.828) far exceeds the top-2 `b0` Oracle
   (0.618), at linear cost in candidates and propagation.
3. **BUSI's +0.185 is not a universal law**: it requires the raw score to be
   systematically wrong, a precondition that does not hold on TN3K.

---

## 6. Frozen vs open

### Frozen

- Stage 0: the descriptor (1008 SAM3 trunk, 64 local words, IDF), the similarity
  `S`, the greedy facility-location rule, the budget rule, determinism, the
  duplicate/tie handling, and one anchor list per dataset (SHA256-recorded).
- Stage 1: TP as the only route family, `b0–b6`, beam 32, bridges from train
  only, anchor GT tight box, no text, canvas 256, `q_cycle` as a return-quality
  feature only, the five ranking arms, the legacy-28 Ridge(α=1) Router with
  image-grouped 5-fold validation, and the "freeze choices before reading test
  GT" order.

### Studied but not adopted

- Anchor score calibration as a **mandatory** Stage-1 module. It is a real,
  label-free, measurable bias (bias ratio 0.68 / 1.10 / 5.42; TN3K span 0.5502)
  with dataset-gated benefit, and it is reported as an ablation, not as part of
  the frozen pipeline.

### Open

| id | item | gap it closes | cost |
|---|---|---|---|
| **P3** | ISIC2018 top-5 (35 candidates/target, val+test) | the third full-depth calibration point; a unified main table | ~12 h GPU |
| **P4** | random-anchor control with real propagation (10 sets × 100 targets × b0, b6) | downstream-Dice evidence for the coverage claim — currently only proxy coverage | ~4 h |
| **P5** | Kvasir unified-router refit with `k` chosen on validation | makes the main table one method instead of four arms | 0 GPU (candidates ready) |
| **P7** | a never-diagnosed dataset | one-shot blind test | dataset-dependent |
| **S2** | Stage 2 closure on the auto-anchor protocol | pseudo labels → student training under Stage 0 + Stage 1 (see §7) | – |

Also missing: TN3K's bias-ratio / concentration / tie statistics; ISIC2018 and
BUSI full anchor banks; `all23` beyond `b0`.

---

## 7. The Stage 2 gap (the most important open item)

Stage 0 and Stage 1 are complete and reproducible, but **the student/training loop
has never been run on the auto-anchor protocol**. Evidence:

- No `automatic_anchor_*` or `tn3k_*` experiment directory contains a
  pseudo-label pool, a student manifest, a student trainer, or a student
  checkpoint — only `protocol/merged_manifest.jsonl`,
  `protocol/support_manifest.jsonl`, route caches and metrics.
- Every `student_best.pth` / `student_final.pth` in the tree is dated
  **2026-09-12 or earlier**, i.e. from the fixed-anchor protocol
  (`kvasir_tp_student_mainline_20260907`,
  `kvasir_tp_filterfirst_students_20260909`,
  `single_student_hard_soft_20260909`, `tp_student_rescreen_20260910`,
  `tp448_*`, `round2_*`).

So the chain

```text
Stage 1 candidate masks
  -> quality gates (q_multi / q_return)  or  Router Top-1
  -> Round-1 pseudo-label pool
  -> S2/S3 (or one student)
  -> committee audit -> X3 -> B7
```

exists only for the old fixed-anchor Kvasir protocol
(see [`kvasir_versions.md`](kvasir_versions.md) V1–V4), **not** for the new
Stage 0 + Stage 1.

Before wiring it up, §5.4 argues that the reference ranking should first gain a
scale term (or retain more anchors), otherwise the student's capacity is spent on
the ~23% unreachable tail rather than on the reachable majority.

---

## 8. Artifact index

```text
# Stage 0 — anchor mining
mainline/experiments/automatic_anchor_selection_pilot_20260911/   Kvasir pilot:
    report.md, SELECTIONS_FROZEN.json, coverage_metrics.json, policy.json,
    global_facility/, global_local_facility/, existing8/, greedy traces
mainline/experiments/isic2018_auto21_tp_validation_20260911/selection/  ISIC pilot
mainline/experiments/busi_auto5_tp_1pct_20260913/{SELECTED_SUPPORT_FROZEN.json,
    policy.json, selection/}
mainline/reproduction_guides/automatic_selection_20260914/
    reproduce_automatic_selection.py, source_hashes.json, verification_receipt.json,
    verify_{kvasir,isic2018,busi}/selection_replay.json + SELECTED_IDS_FROZEN.json,
    E0_diagnostics_20260914/ (bias ratios, Oracle/Gap, figures, per-candidate CSVs),
    P1_pilot_20260914/ (b0 calibration pilot, preregistration, predictions),
    P2_fullbank_20260914/ (Kvasir 56-candidate bank, k-curve, budget curve),
    PAPER_OUTLINE.md, 三数据集自动选图与TP路线_中文复现指南.md

# Stage 1 — routes, propagation, selection
mainline/experiments/automatic_anchor_tp_test_20260913/{kvasir,isic2018}/
    protocol/, code/ (frozen route + propagation scripts), quality_root/,
    PREDICTIONS_FROZEN.json, ROUTES_FROZEN.json, results.json, report.md, COMPLETE
mainline/experiments/automatic_anchor_tp_router_20260913/{kvasir,isic2018}/
    models_frozen.json, folds_frozen.json, validation_results.json,
    test_choices_frozen.json, results.json, completion_audit.json, per-method masks/
mainline/experiments/busi_auto5_tp_1pct_20260913/, busi_auto5_tp_router_20260913/
mainline/experiments/cross_dataset_calibration_factorial_20260914/
    README.md, results.json, {kvasir,isic2018}/{report.md, results.json,
    calibration_frozen.json, pool_membership_frozen.json, validation_results.json,
    models_frozen.json, test_choices_frozen.json, test_per_target.json,
    test_candidate_metrics.json, predefined_policy.json, completion_audit.json}
mainline/experiments/busi_calibration_factorial_20260914/,
    busi_calibrated_multi_anchor_20260913/
mainline/experiments/tn3k_busi_factorial_20260915/
    report.md, results.json, validation_results.json, analysis_extra.md,
    why_gap_vs_busi.md, completion_audit.json, dataset_audit.json, pool_audit.json
mainline/experiments/tn3k_failure_diagnosis_20260916/ (诊断报告.md, all-anchor b0)
mainline/experiments/tn3k_boxscale_probe_20260916/  (probe_result.json)
mainline/experiments/tn3k_reference_ranking_pilot_20260916/ (参考排序诊断.md)
mainline/compare_tn3k_busi.py, tn3k_seg_diagnosis.py, tn3k_boxscale_probe.py,
    tn3k_scale_mismatch.py, tn3k_verify_results.py
results/busi_1pct_protocol/result.md                (SynFoC head-to-head)
```

## 9. Reproduction entry points

```bash
cd <repo>
export SAM_PY=/home/violet/anaconda3/envs/sam3/bin/python
export CPU_PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
export PYTHONPATH=/Data_8TB/lht/sam3:$PWD/src

# Stage 0 — verify the frozen anchor lists (CPU, no GPU, no GT read)
for D in kvasir isic2018 busi; do
  $CPU_PY mainline/reproduction_guides/automatic_selection_20260914/\
reproduce_automatic_selection.py select --dataset $D --out /tmp/replay_$D
done
# acceptance: selection_replay.json has exact_match=true, mask_pixels_read=0

# Stage 1 — rebuild routes, propagate, and refit the Router for one dataset
$SAM_PY mainline/reproduction_guides/automatic_selection_20260914/\
reproduce_automatic_selection.py prepare --dataset busi --selection <sel> --out <out>
# then run <out>/code/stage1_feature_knn_routes.py and
# <out>/code/eval_route_propagation_quality.py per the four-dataset guide, and
# finally evaluate; see 三数据集自动选图与TP路线_中文复现指南.md §8 for exact flags
```

The four-dataset guide
(`mainline/reproduction_guides/automatic_selection_20260914/三数据集自动选图与TP路线_中文复现指南.md`)
carries the exact command lines, the expected candidate counts per split and the
acceptance values; all scripts are frozen copies inside each experiment directory
so that later edits to the shared `scripts/` tree cannot change the results.

## 10. Do not conflate

- Coverage numbers are **not** Dice; `Oracle` is not a deployable score.
- `bias_ratio` is not a performance metric; it only orders how much calibration
  can help.
- `raw_top1` (one fixed anchor per target) ≠ `original_per_bridge` (per-depth
  best path across all anchors).
- `q_cycle` is anchor-return consistency, not target Dice.
- `b0` is a two-frame anchor→target propagation, not prompt-free single-image
  segmentation.
- 7 → 14 candidates doubles the propagation task count; the recorded `seconds`
  are a workload reference, not a controlled timing.
- Test has been examined across many rounds on all four datasets: **no result
  here is a blind test**; the paired intervals are exploratory.
- The Router uses validation GT, so "1%" is the **anchor-labelling** budget only,
  not the total supervision of the system.
