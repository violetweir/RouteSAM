# Kvasir-SEG: the V1 → V7 ladder

> Kvasir-SEG is the largest arm of the mainline and it was rebuilt seven times.
> This page walks through **every version from start to finish** — why it
> existed, the exact pipeline and hyper-parameters, the pool sizes, all headline
> numbers, the caveats the original authors recorded, and where the artifacts
> live.
>
> Companion pages: [`cross_dataset_1pct.md`](cross_dataset_1pct.md) (V5–V7 as the
> cross-dataset arm) · [`kvasir_program.md`](kvasir_program.md) (full experiment
> inventory) · [`kvasir_1pct_anchor.md`](kvasir_1pct_anchor.md) (route/router
> reference).
>
> **Naming note.** The strings "V1"…"V7" do not appear in the server files; the
> numbering is the project owner's. Each section states the source-file names it
> corresponds to.

## The ladder at a glance

| Version | Mainline | Main change |
|---|---|---|
| **V1** | original 8 anchors → **DINOv3** kNN → SAM3 **TP + PC** → S2/S3 → committee → X3 → B7 | the earliest complete pipeline; SAM3 LoRA added later |
| **V2** | original 8 anchors → **SAM3-base feature** kNN → TP + PC → S2/S3 → X3 → B7 | SAM3 features replace DINOv3; carries Round-1 / Round-2A |
| **V3** | original 8 anchors → **single TP** `b0–b6` + independent Router → 448 pseudo labels → S2/S3 → expansion, X3, B7 | drops PC, settles the single-TP baseline |
| **V4** | original 8 anchors → TP → **filter admissible candidates before the Router** → 580 → single student soft labels → 620 rescreen → Round-2 SAM3 LoRA | 580-pool control, single equal-weight student, 620 rescreen, tracker-endpoint pilot, Round-2 fine-tunes |
| **V5** | **automatic 8 anchors** → SAM3-base kNN → single TP `b0–b6` → Router | changes where the anchors come from |
| **V6** | automatic 8 anchors → **raw TP score top-2** anchors → `b0–b6` each → Router | 7 → 14 candidates per target |
| **V7** | automatic 8 anchors → **calibrated TP score top-2** anchors → `b0–b6` each → Router | annotation-free score calibration |

### Headline test Dice (Kvasir, 100 targets)

| Version | variant | headline | Router | Oracle |
|---|---|---|---:|---:|
| V1 | original C0 (canvas 512, `ft_1pct`) | **X3 + B7 0.899369** | 0.894648 | 0.919805 |
| V1 | C0-256 (canvas 256, `ft_1pct`) | **X3 + B7 0.897146** | 0.891288 | 0.918109 |
| V1 | C0-256-base (canvas 256, base `sam3.pt`) | X3 + B7 0.886130 | 0.871822 | 0.903589 |
| V2 | SAM3-enc kNN @256, base propagation | **X3 + B7 0.895432** | 0.874083 (TP+PC) / 0.885433 (TP-only) | 0.921471 |
| V2 | + second full-module LoRA (e33) | **X3 + B7 0.906380** | – | 0.920206 |
| V3 | single TP, 448 pool | **X3 + B7 0.892639** | 0.885433 | 0.907426 |
| V4 | 580 pool (old S2/S3) | X3 + B7 0.890846 | 0.885433 | 0.907426 |
| V4 | 620 rescreen, single student | student best 0.858330 / final **0.869478**; B7 best **0.894641** / final 0.893723 | 0.885433 | 0.907426 |
| V4 | Round-2 SAM3 LoRA (10-epoch pools) | 0.901755 / 0.903294 / **0.912932** | – | – |
| V5 | automatic anchors, per-bridge routing | **Router 0.871374** | 0.871374 | 0.917615 |
| V6 | raw top-2 | **Router 0.891226** | 0.891226 | 0.932897 |
| V7 | calibrated top-2 | Router 0.862761 | 0.862761 | **0.937542** |

> **These numbers are not protocol-identical across the whole ladder.** V1/V2 use
> the dual-route (TP + PC) candidate set; V3/V4 use single-TP with a different
> pseudo-label pool and a different student recipe; V5–V7 use automatic anchors
> and a refitted router, and V5's historical rule (`original_per_bridge`) is not
> the same rule as the factorial's `raw_top1` arm. Compare V1↔V2, V2↔V3↔V4 and
> V5↔V6↔V7 pairwise, never across the whole table.

---

# V1 — DINOv3 dual-route (original C0)

Corresponds to `work/rerun_c0/`, `work/rerun_c0_256/`, `work/rerun_c0_256_base/`,
`work/kvasir_1pct_anchors/phase1/`, and `docs/method_cn.md` / `docs/method_en.md`.

## 1. Why this version exists

V1 migrates the public `S27 X3 + B7` mainline (merged CVC + Kvasir, 16 anchors,
frozen SAM3 0.8715 → X3 0.866738 → B7 0.895835, Oracle 0.907835) onto the WACV2027
protocol: **Kvasir-SEG alone, with only 1% of train labelled (8 fixed anchors)**.
A single-image student trained on 8 labels is far too weak (SC-SAM 0.6626/0.6373,
SynFoC 0.7615/0.7681), so the line keeps SAM3 for route propagation and uses
students as **auditors and selectors**. The frozen-route propagation-quality
router already reached 0.877299 (Oracle 0.896918), i.e. selection-limited, and
RouteCo-v1 (adapter-only) had proved roughly Oracle-neutral — so the transfer
target became the public mainline's student-audit loop.

## 2. Pipeline, end to end

Protocol: `train=800 / validation=100 / test=100`; 8 human anchors; the other 792
train images are the unlabelled bridge pool; validation/test GT is evaluated only.

```text
DINOv3 ViT-S/16 features at 224x224
  (CLS token, 14x14 = 196 patch tokens, patch mean; all L2-normalised)
  -> anchor prototype = mean of the patch tokens inside the anchor GT mask
  -> two anchor-conditioned route families:
       anchor_conditioned_target_pooling
           w_j = softmax(10 * cos(anchor_proto, x_j)); z = L2(sum_j w_j x_j)
           cond_score = dot(anchor_proto, z)
       anchor_conditioned_patch_correspondence
           cond_score = mean of the top-8 patch similarities
  -> beam search (width 32) prepends bridge frames one at a time; the path score
     is the (min, mean) lexicographic pair over node condition scores plus the
     patch-mean cosine between consecutive nodes; across the 8 anchors keep the
     best (anchor, bridge sequence)
     train: b3-b6 only; validation/test: b0-b6
  -> frozen routes, shared by all later experiments in this version
  -> SAM3 executes each route: only the anchor frame gets a normalised GT bbox
     prompt, text_str = None; later frames rely on multi-frame memory
     canvas 256 (the original C0 run used 512)
     checkpoint: ft_1pct (8 GT, COCO, 20 epochs, lr_scale 0.1, 160 train steps)
  -> 21 GT-free propagation-quality features per route
     (q_cycle, cycle_success, cycle_sam_score, area trajectory, empty-mask count,
      connected components, centroid/bbox drift, adjacent-frame Dice,
      SAM-score trajectory, candidate counts)
  -> ridge over validation features (21 + mode one-hot), Top-1 route per target
  -> pseudo pool gates: q_return(=q_cycle) >= 0.95 and q_multi >= 0.90
       q_multi = mean over the 21 pairwise Dice of the 7 candidates (image level)
       image weight = normalised q_multi clipped to [0.2, 1]
  -> S2 / S3 students (SC-SAM SamUnet: a U-Net with no SAM3 encoder)
       batch 12 = 6 GT + 6 pseudo, UNet-lr 0.01, momentum 0.9, wd 1e-4,
       40000 iterations, validation every 200, seed 2026
       S2: router-picked hard masks, loss = L_GT + 0.5*ramp*L_pseudo,
           ramp 0 -> 1 over the first 2000 iterations
       S3: consensus soft labels P = mean of the 7 candidate masks, pixel weight
           max(0.1, exp(-4 * variance)), weighted BCE + weighted soft Dice
  -> committee over the unselected targets; members S2 val-best, S2 final,
     S3 final
       q_route = 0.20*q_multi + 0.80*q_model_mean
       Tier A: q_multi>=0.90, q_model_mean>=0.90, min>=0.80, var<=0.01,
               q_route>=0.88, non-empty, >=2 students non-empty, area-safe
       Tier B: q_multi>=0.75, mean>=0.75, min>=0.60, var<=0.03
  -> new labels: soft = 0.75*M + 0.25*(four-student mean),
     pixel weight clip(exp(-5*V_route)*exp(-5*V_student)*(1-|M-S|), 0.05, 1)
  -> X3 three-stream student, fresh init, 40000 iterations
       batch 12 = 3 GT / 3 original / 6 new; stream weights 1.0 / 0.75 / 0.50;
       global pseudo 0.5 with a 2000-step ramp
  -> B7: Score_j = (q_return_j * q_multi_j^2 * q_model_j^2)^(1/5), floor 1e-6,
     Top-1, copy that SAM3 candidate mask (no pixel fusion)
```

## 3. The three V1 runs

| run | KNN features | propagation | canvas | pool | committee A/B/C | X3 pool | Router | B7 | Oracle |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| original C0 (`phase1`) | DINOv3 ViT-S/16 @224 | `ft_1pct` | 512 | 491 | 93 / 95 / 113 | 679 | 0.894648 | **0.899369** | 0.919805 |
| C0-256 (`rerun_c0_256`) | DINOv3 ViT-S/16 @224 | `ft_1pct` | 256 | 470 | 105 / 102 | 677 | 0.891288 | **0.897146** | 0.918109 |
| C0-256-base (`rerun_c0_256_base`) | DINOv3 ViT-S/16 @224 | base `sam3.pt` | 256 | 386 | 134 / 103 / 169 | 623 | 0.871822 | 0.886130 | 0.903589 |

C0-256 detail (`reports/C0_256_reproduction.md`):

| Router variant | selected Dice | Oracle | gap |
|---|---:|---:|---:|
| target pooling + patch correspondence | 0.891288 | 0.918109 | 0.026821 |
| anchor-conditioned target pooling | 0.875413 | 0.904479 | 0.029066 |
| anchor-conditioned patch correspondence | 0.891065 | 0.902411 | 0.011347 |
| **final B7** | **0.897146** | 0.918109 | 0.020963 |

Moving both the SAM3 side and the U-Net side to 256 cost almost nothing
(0.899369 → 0.897146, ≈ −0.0022) and removed the resize-alignment problem.
C0-256-base S2/S3 test: S2 val-best 0.8310 (iter 22200) / final 0.8245;
S3 val-best 0.8395 (iter 15600) / final 0.8264; X3-val-best + B7 0.886130,
X3-final + B7 0.875302 (diagnostic). A same-family repeat `work/rerun_c0_c0/`
gives Router 0.892477, B7 0.885416, pool 490, X3 pool 678.

Route-level reference, frozen base, forward-only test Dice
(`docs/method_en.md` §3.4) — old resolution (ViT-S/16 @224 + SAM3 @512); the five
feature modes here are **not** comparable with the @256 tables:

| feature mode | direct | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T18 corrected | 0.747869 | 0.775949 | 0.756490 | 0.785808 | 0.821736 | 0.816809 | 0.819200 | 0.143392 |
| DINO global pooling | 0.740844 | 0.748290 | 0.758326 | 0.786777 | 0.777536 | 0.818397 | 0.818565 | 0.092024 |
| DINO patch average | 0.728634 | 0.726893 | 0.756876 | 0.746756 | 0.799814 | 0.829179 | 0.826294 | 0.046968 |
| anchor-conditioned target pooling | 0.741306 | 0.760176 | 0.827209 | 0.840837 | 0.841258 | 0.848618 | **0.854627** | 0.032027 |
| anchor-conditioned patch correspondence | 0.757814 | 0.773501 | 0.822305 | 0.832819 | 0.833990 | 0.833577 | 0.842078 | 0.070807 |

`b7` collapses for every mode; the stable region is `b4–b6`; the best fixed route
is target pooling `b6` = 0.854627. Under the 256 protocol with a DINOv3 **ViT-B**
backbone the same sweep peaks at `t18_corrected b6` = 0.8573, and the
anchor-conditioned modes prefer `cls` kNN (patch 0.8355, target 0.8126)
(`docs/knn_experiment_vitb256.md` §6). With the LoRA teacher `lora_p491_e20` the
same 11 variants improve by +0.03…+0.13 per cell and the ranking reverses:
`target_pooling knn=cls b6` = **0.8978** becomes the best, and the previously
weakest `cond` kNN gains the most.

`ft_1pct` vs frozen per bridge (`docs/method_cn.md` §4.1):

| bridge | ft_1pct target | frozen target | ft_1pct patch | frozen patch |
|---|---:|---:|---:|---:|
| direct | 0.7708 | 0.7413 | 0.7979 | 0.7578 |
| b1 | 0.7922 | 0.7602 | 0.8161 | 0.7735 |
| b2 | 0.8574 | 0.8272 | 0.8578 | 0.8223 |
| b3 | 0.8522 | 0.8408 | 0.8599 | 0.8328 |
| b4 | 0.8609 | 0.8413 | **0.8843** | 0.8340 |
| b5 | 0.8731 | 0.8486 | 0.8658 | 0.8336 |
| b6 | 0.8643 | 0.8546 | 0.8797 | 0.8421 |

Student-side headline for the 512 original C0: X3 single image 0.8633 (IoU not
found), **X3 + B7 0.899369** (Oracle 0.919805, gap 0.0204), pool 491/792,
committee A 93 / B 95 / C 113 over the remaining 301 targets, X3 pool 679;
S2 val 0.80755 @34000, S3 val 0.80368 @16400.

## 4. Experiments that belong to V1

**Fine-tuning budget sweep** (weighted 3-route selector): base 0.830103,
ft_1pct 0.874330, ft_5pct 0.882543, ft_10pct 0.844562, ft_20pct **0.914321**
(strongest; small budgets are not monotonic). The 3–7-route weighted table gives
ft_20pct w7 0.910493 with Oracle 0.942206.

**First full-module LoRA.** Selection rule: X3-best + B7 on validation, keep
`B7 >= 0.94` → 494/792 train images (+8 anchors = 502 with validation). LoRA rank
16 / alpha 32 / dropout 0.1 over vision + text + geometry encoders, DETR
encoder/decoder and mask decoder; 383 modules, 17 883 264 trainable parameters
(2.08%); AdamW lr 5e-5, wd 0.01, batch 1, FP32, seed 2026, 50 epochs, every epoch
checkpointed. **e35** is the official validation-best: test B7 **0.898734**
(Oracle 0.904141); e45 0.897579 and e50 0.900173 are user-authorised test audits
only. Training loss falling does **not** imply downstream Dice rising (e45
regresses, e50 recovers but stays below e35).

**Round-2 (full fine-tune, not V2's Round-2A).** Student-audited consensus pool
455/792 (mean Dice 0.921, 5.1% below 0.7); 8 GT + 455 = 463 images; full
fine-tune `lr_scale=0.1`, 20 epochs; **best-val AP 0.6799 (epoch 13)**, final
0.6603 (epoch 19); checkpoints at 1/6/11/16/20. Dual-checkpoint evaluation: round2
final 0.868266, bestval 0.882049 — **both below `ft_1pct` 0.894648**. Regularity:
round 2 improves short routes (b1–b2) and degrades long routes (b4–b6), while the
router's Top-1 depends mainly on b4–b6. Early checkpoints are the golden zone: e1
beats `ft_1pct` on target 7/7 and patch 6/7 bridges, and the `b0-b6` pool ckpt6 =
0.883671 is the best of the whole round. Widened to `b0-b6`, round-2 bestval
(0.8791) beats `ft_1pct` (0.8771) — the conclusion depends on the candidate range.

**First-frame prompt comparison.** Pure GT-mask prompt −0.0131 (unusable);
`det_gtmask` (detector runs, tracker seeded with the GT mask) +0.0055 with a 59%
route win rate. Only forward Dice — not wired into `q_cycle` / router / B7.

**HQ pseudo-label audit (357 pool).** Protocol contamination zero; median Dice
0.956, 85% ≥ 0.9, but 10/357 < 0.5 and 16/357 < 0.7. These are "confidently
wrong": the best single route-internal feature reaches AUROC ≈ 0.71;
student-route agreement ≈ 0.78 and the disagreement combo ≈ 0.85. The only
completed HQ fine-tune used `lr_scale=0.025` (4× lower than the 1% GT repro), its
validation AP stayed flat (0.6386 → 0.6356) and it did not actually train — so its
low test Dice (0.8367) is **not** evidence against pseudo labels. All
`lr_scale=0.1` HQ runs crashed with a matcher NaN; a finite-guard patch exists but
a clean same-LR comparison is still pending.

**RouteCo-SAM3 v1.** Adapter-only (memory-read adapter + mask-decoder LoRA, 400
steps) is roughly Oracle-neutral (official-pipeline validation Oracle
0.8694 → 0.8614). As a **second candidate pool** it has real upside: a perfect
frozen-or-RouteCo pick reaches 0.8778 vs 0.8694, and 43% of routes are better.
The `S→T` gate is inert because the student is not wired in.

## 5. Caveats recorded for V1

- Old-resolution and 256-resolution tables change **two** variables at once
  (feature resolution and propagation resolution) and must not be compared.
- `b7_mean` is not Dice; `combined` is a two-mode mean, neither a per-target
  choice nor a mask ensemble; **b0 is not "direct"** — b0 is still a two-frame
  anchor+target propagation, while direct is single-image text segmentation.
- `effective256` is not a native 256 image model; the empty-text control is not
  the same as "no query at all".
- The e50 test number exceeds e35 but the **official checkpoint stays
  validation-best e35**; likewise a higher S3-final test number does not license
  switching to final.
- The round-2 conclusion depends on the candidate-pool range (see §4).
- Small fine-tuning budgets are non-monotone (10% drops); 20% is strongest.
- RouteCo's early "Oracle drop" mixed the core and official forward pipelines and
  was overstated; always compare same-pipeline.
- `q_multi`'s peer count changes with the pool size (7 vs 13), so its thresholds
  **cannot be silently inherited** across versions.

---

# V2 — SAM3-encoder dual-route

Corresponds to `work/rerun_c0_256_sam3knn_s256_base/`,
`work/rerun_c0_256_round2a_fixed_knn_e33/`, and the
`docs/knn_experiment_sam3enc_*.md` series.

## 1. Why this version exists

V1 built routes in DINOv3 feature space while propagation ran in SAM3. V2 removes
that mismatch: the KNN feature extractor becomes the **frozen base SAM3 image
trunk at 256**. Everything else is controlled — anchors, splits, beam width,
propagation model, propagation canvas, student architecture, seeds, selection
logic (`reports/C0_256_sam3knn_s256_b0_b6_run.md`).

Second change: the route pool is expanded to **`b0–b6` on every split** (V1 used
train `b3-b6` / val-test `b0-b6`), and no `b0–b2` result is discarded. The peer
count for `q_multi` therefore changes from 7 to 13, and the S3 consensus variance,
committee variance and B7 distribution all shift — which is why the students are
retrained: the DINO-trained X3+B7 scores only 0.838158 on the new routes
(0.853511 on DINO routes).

## 2. Pipeline, end to end

| component | V1 (C0-256-base) | V2 |
|---|---|---|
| KNN backbone | DINOv3 ViT-S/16 | **base SAM3 image trunk** |
| feature input | 224×224 | 256×256 |
| descriptor | L2-normalised patch mean | L2-normalised patch mean |
| descriptor width | 384 | **1024** |
| anchors | 8 frozen human masks | unchanged |
| route modes | anchor-conditioned target / patch | SAM3-encoder analogues |
| beam width | 32 | 32 |
| propagation model | base `sam3.pt` | unchanged |
| propagation canvas | 256 | 256 |
| saved bridge range | train b3–b6; val/test b0–b6 | **all splits b0–b6** |

Features: SAM3 trunk @256, patch 14 → grid 18×18 = **324 tokens**, 1024 dims each,
**no CLS token**; cache `sam3_base_s256_features.npz`
(SHA256 `ab8ca194…9907`). Route pool = 7 steps × 2 modes = **14 candidates per
target**; train 792×7×2 = 11 088 routes, val/test 1400 each. The @1008 native
feature resolution is kept as a separate control.

Pool gates, committee rules, X3 three-stream recipe and the B7 formula are
inherited from V1. Result: pool **410/792** (the `b3-b6` control gives 389),
committee **A 94 / B 128 / C 160**, X3 pool **632**.

## 3. Results

Selector: frozen X3 val-best, `0.826032` at iteration 28 800
(`reports/C0_256_sam3knn_s256_b0_b6_test_details.md`).

| result | Test Dice | IoU / Oracle | gap |
|---|---:|---:|---:|
| X3 val-best direct | 0.853920 | 0.771766 IoU | – |
| **X3 val-best + B7, b0–b6** | **0.895432** | 0.921471 Oracle | 0.026039 |
| X3 final direct | 0.858591 | 0.778503 IoU | – |
| X3 final + B7 (control, not mainline) | 0.897684 | 0.921471 Oracle | 0.023786 |

Versus the DINOv3 C0-256-base routes the SAM3-encoder routes improve B7 at both
checkpoints: X3 val-best **0.886130 → 0.895432 (+0.009302)**, X3 final
0.875302 → 0.897684 (+0.022382). The unified re-scored export entry gives X3
val-best 0.854101 / IoU 0.771470.

Per-step test Dice (100 targets each; `combined` is the 200-route mean, not B7):

| step | target pooling | patch correspondence | combined |
|---|---:|---:|---:|
| direct / b0 | 0.798144 | 0.706645 | 0.752395 |
| b1 | 0.846623 | 0.754897 | 0.800760 |
| b2 | 0.870032 | 0.788470 | 0.829251 |
| b3 | 0.862253 | 0.859413 | 0.860833 |
| b4 | 0.873930 | 0.857625 | **0.865778** |
| b5 | 0.869552 | 0.855339 | 0.862446 |
| b6 | **0.874004** | 0.853581 | 0.863793 |

Target pooling is consistently stronger; target pooling peaks at b6, patch
correspondence at b3. X3 test by lesion size (24 small / 21 medium / 55 large):
val-best 0.853920 / 0.853231 / 0.899043 / 0.836993.

B7 selection behaviour: 68 targets chose target pooling (mean selected Dice
0.905496) and 32 chose patch correspondence (mean 0.874046) — patch has a
**higher mean B7 but lower true Dice**, i.e. self-consistency is occasionally
overconfident on patch candidates. B7 values describe selector confidence only.

S2/S3 (base, b0–b6): S2 val-best 0.820152 @32400 → test 0.855042 / 0.771485;
S3 val-best 0.816272 @20200 → 0.845245 / 0.763529; S3 final test
0.862539 / 0.786166.

Same-topology 5-fold target-level router comparison (DINO vs SAM3 features):
`b0-b6` selected 0.847886 vs 0.841517, Oracle 0.889047 vs 0.889432;
`b3-b6` selected 0.831619 vs 0.836823, Oracle 0.868303 vs 0.878693.

## 4. V2's own experiments

**Round-1.** Rule frozen on validation: X3 val-best + both SAM3-KNN modes +
`b0-b6` + `B7 >= 0.94` → train pool **486/792** (+8 anchors = 494 training
images, 100 validation). Validation threshold sweep: 0.90 → 64/100 (0.946575),
**0.94 → 54/100 (0.950447)**, 0.96 → 43/100 (0.953728); X3-final at 0.94 →
52/100 (0.950145); the `b3-b6` control at 0.94 → 58/100 (0.939349).

**Second full-module LoRA (e33).** 486 pseudo + 8 GT = 494 train / 100 val; same
LoRA recipe; AdamW 5e-5, wd 0.01, batch 1, seed 2026, 50 epochs; training
resolution 1008, video canvas 256. **Loss-best is e20** (val loss 6.7681) but
**Direct-Dice-best is e33** (0.872377) — e33 becomes the teacher for Round-2A.
e33 direct test: 1008 + text **0.885042** (IoU 0.825374), effective-256 + text
0.882267, effective-256 + empty 0.866378; the same-protocol base SAM3 scores only
0.461427.

**Round-2A (fixed topology, teacher base → e33).** Reuses e33 val/test
propagation and only adds the 11 088 train propagations; `q_model` still uses the
frozen X3-best. B7 recalibration rule fixed in advance ("among B7 prefixes whose
selected validation Dice ≥ 0.95, take the largest coverage") → frozen threshold
**0.9460352822729875**, 56/100 pass, selected val Dice 0.950100. Pool
**524/792** (coverage 66.16%; original 394 / tier_a 72 / tier_b 58), 38 more than
the base teacher's 486; overlaps the old X3 manifest on 504, adds 20, drops 128.
Student-X4 (same architecture, seed, recipe, 40 000 iterations): best 26 000
(val 0.839122) vs X3 best 28 800 (0.826032). X4 best single image: val 0.839122,
test 0.860007 / IoU 0.781950.

Teacher-quality control on the same 1400 validation candidates (same KNN graph,
same X3 `q_model`): `q_return` 0.807626 → **0.915586** (+0.107959), `q_multi`
0.869469 → 0.904258, `q_model` 0.813009 → 0.833374, B7 0.756561 → 0.856623, GT
Dice (audit only) 0.796126 → 0.845750; per-target best-B7 candidate selected val
Dice 0.833866 → **0.883854** (+0.049988). With the frozen threshold, accepted
targets go 54 → 56 while the accepted audit GT Dice stays at ~0.95
(0.950447 → 0.950100).

B7 closeout: **e33 + X3-best + B7 test 0.906380** (val 0.883854, Oracle
0.920206); e33 + X4-best + B7 0.905375 (val 0.880070); X4-final + B7 0.905385
(diagnostic). e33's candidate Oracle (0.920206) is slightly **below** base
(0.921471) yet the selected Dice is higher — the selector-to-oracle gap narrows
from 0.026039 to 0.013826, so the gain comes from choosing better, not from a
higher ceiling.

e33 per-step test Dice (`b0–b6`): target 0.796551 / 0.809615 / 0.875972 /
0.891441 / **0.904009** / 0.897443 / 0.899980; patch 0.828143 / 0.840549 /
0.856388 / 0.878680 / 0.885240 / 0.893036 / **0.908151**; combined 0.812347 /
0.825082 / 0.866180 / 0.885061 / 0.894624 / 0.895240 / **0.904065**. Against
base, e33 improves `b3–b6` by +0.024…+0.082 on validation; `b0` is the only clear
regression.

**SAM3-encoder feature study** (`docs/knn_experiment_vitb256.md` §9,
`docs/knn_experiment_sam3enc_imr_s256.md`,
`docs/knn_experiment_sam3enc_anchor_ablation.md`,
`docs/knn_experiment_lesion_mechanism.md`,
`docs/knn_experiment_negcontrol_transport.md`):

- @1008 + lora: target pooling `cond` b1/b4 **0.9154**; patch correspondence
  `patch_mean` b6 0.9150. @256 + lora: patch correspondence `cond` b4 **0.9082**,
  b6 0.9012; patch_mean b5 0.9036. Base: @256 `cond` b3 0.8841, @1008
  `patch_mean` b5 0.8874.
- Best fixed route across feature extractors: DINOv3 ViT-B @256 0.8573 (base) /
  0.8978 (lora); SAM3-enc @256 0.8841 / 0.9082; SAM3-enc @1008 **0.8874** /
  **0.9154**.
- Ranking reversal: under base `t18_corrected` leads; under LoRA the
  anchor-conditioned long bridges overtake it, and `cond` (weakest under base)
  gains the most (target b6 +0.125). Direct (b0) also gains strongly (1008:
  0.82 base → 0.90 lora vs DINO 0.69–0.74 → 0.80).
- IMR @256 (pyramid + contrast + Qwen text): base λ = 0 gives b1 0.8572,
  b3 0.8674, b4 0.8751; λ > 0 is uniformly worse. `sam3enc_contrast` is a design
  failure — after L2 normalisation the token-channel variance differences are
  negligible, so variance weighting degenerates to a plain mean and the routes
  are identical to the plain patch average. With lora, `sam3enc_imr__tb050`
  reaches b6 **0.9110**, b4 0.9074, b5 0.9060.
- FPN foreground transport (fixed `ft_1pct`, canvas 256): val b4 0.8788, test b4
  0.8871 vs DINO patch b4 0.8700 / 0.8820; b4 is positive on both splits
  (val +0.00881, test +0.00507, 64% per-target win rate) but the paired bootstrap
  CI still spans 0, so it is **not** yet significant and does not justify deleting
  the DINO branch.
- Anchor ablation (SAM3-enc @1008, `cond`): mean(b1–b6) 8→0.8721, 6→0.8474,
  4→0.8217, 3→0.8240, 2→0.7855, 1→**0.8596** — non-monotone; multi-seed n = 1
  spans −0.013 to −0.087 (four unique anchors, 0.8390 ± 0.027). **Anchor strategy
  matters more than anchor count.**
- Lesion KNN: lesion-anchor/lesion-bridge (L/L) per-target best 0.9300 vs
  global/global 0.9022; L/G and G/L are worse than G/G at long bridges.
- Negative controls: naive unsupervised transport proxies all fail —
  `rho(q_model, GT Dice) = 0.029` (p = 0.446), `rho(cycle, GT Dice) = −0.090`
  (p = 0.372, not significant); proxy-oracle 0.8409–0.8495 ≈ the per-target mean
  0.8435, far below the GT oracle 0.8972.

**PC series (why V3 drops patch correspondence).** Joint-v1
(`0.75 z(TP) + 0.25 z(top-8 PC)`) test 0.850575 and Joint-v2 0.847341, both below
the TP baseline 0.885433, with paired intervals excluding 0. The PC candidate
rerank and the PC gain gate both failed their pre-set validation thresholds
(gain ≥ 0.003 with a positive CI lower bound) and **never ran test**. The nested
candidate-pool decomposition shows ~**93.28%** of the PC Oracle gain comes from
switching to anchors the TP pool did not select — an observational diagnostic,
not a causal ablation, and TP and PC share the same `s_j = pᵀx_j`.

## 5. Caveats recorded for V2

- The official mainline stays X3 val-best (chosen on validation); X3 final is a
  test-only control even though its test number is higher.
- This test execution was explicitly authorised after earlier diagnostic use, so
  it is **not a blind test**.
- The V1 → V2 comparison is not a single-factor ablation: it changes the feature
  backbone, the feature resolution, the bridge range and the students, and V2
  additionally fixes the 16-bit S3 soft-label reader.
- Dropping PC lowers the candidate Oracle (0.921471 → 0.907426), so the smaller
  Oracle gap does not by itself prove a better scorer.
- Joint-scoring failure cannot be attributed to the bridge path alone: b0 has no
  bridge, and the round changed the score combination, the cross-anchor scale and
  the bridge-transition scale at once.
- Both rerank/gate experiments tested only a pre-fixed 15-feature, strongly
  regularised linear model with three thresholds — this does not prove all gating
  methods are useless. Candidate rows are not independent images (only 100
  targets), so nested per-target OOF grouping is mandatory.
- Better teacher → better student holds, but **better student → better B7
  selector does not**: X4-best single image +0.006087 while B7 −0.001005
  (val −0.003784); on test 44 targets changed selection, X3 picks averaging
  0.927092 vs X4 0.924808. X4 also degrades medium lesions (−0.019098).
- The KNN graph is not rebuilt in Round-2A (routes are symlinked), so it tests a
  stronger teacher, not a better graph.
- The IMR λ was chosen on test (not validation), so those numbers are slightly
  optimistic.
- `q_multi` thresholds change meaning with the peer count; the validation
  54/100 is threshold coverage, not dropped data.

---

# V3 — single-TP baseline, 448-image pool

Corresponds to `work/kvasir_tp_b0_b6_router_baseline_20260907/` and
`work/kvasir_tp_student_mainline_20260907/`.

## 1. Why this version exists

V2 carried two route families. Diagnostics showed the union router was worse than
a TP-only router (test 0.874083 vs 0.885433, +1.135 pp for TP-only) even though
PC supplied some masks the TP pool lacked (candidate Oracle fell 0.921471 →
0.907426, −1.404 pp). Two failure modes were identified in the union router: cross
family mis-selection (the 22 targets that chose a PC candidate contributed
−0.007930 over all 100; a PC-only router scores 0.861688 and beats TP on 32 test
images, ties 1, loses 67) and degraded TP-internal ranking (the 78 targets still
on TP contributed −0.003419, 30 of them picking a different TP candidate than the
TP independent router). The new joint variants also broke anchor choice: at `b0`
they swapped the anchor on 62 of 100 test targets, costing −0.095413 / −0.097427.
On 2026-09-09 the decision was taken to **drop PC and freeze a single-TP
baseline**; PC, alternative TP anchors, TP/PC retrieval comparisons and the
auxiliary gain selectors were archived.

## 2. Pipeline, end to end

```text
frozen SAM3-base (sam3.pt SHA256 9999e234...8c9e), features 256, canvas 256
route_mode = sam3enc_anchor_conditioned_target_pooling, bridges b0-b6
  -> 7 candidates per target; 792x7 = 5544 train propagations, 700 validation
independent Router: standardised ridge (ridge=1, no mode one-hot) fitted on
validation candidate features and true Dice; inference never sees target GT
V3 order  = select, then check:
  Router picks 1 of 7 -> check image-level q_multi >= 0.90 and the SELECTED
  candidate's q_return >= 0.95 -> both pass or the whole image is excluded
  792 breakdown: both pass 448, fail only q_multi 76, fail only q_return 160,
  fail both 108  ->  acceptance 448/792 = 56.57%
  image weight = min-max normalised q_multi clipped to [0.2, 1]
  validation keeps 52/100 by the same gates (descriptive, not OOF)
S2: 8 GT + 448 hard masks, batch 12 = 6 GT + 6 pseudo, loss
    L_GT + 0.5*ramp*L_pseudo, ramp 0->1 over 2000 iterations,
    40000 iterations, val every 200, seed 2026
S3: consensus soft labels (mean of the 7 candidates), pixel weight
    max(0.1, exp(-4*variance)), weighted BCE + weighted soft Dice
committee over the remaining 344 targets, members S2-best, S2-final, S3-best,
S3-final; q_route = 0.20*q_multi + 0.80*q_model_mean
  Tier A: q_multi>=0.90, q_model_mean>=0.90, min>=0.80, var<=0.01,
          q_route>=0.88, non-empty, >=2 students non-empty,
          area within the 1%-99% quantiles of the original pool
          (about 1.62%-28.48% of image area)
  Tier B: q_multi>=0.75, mean>=0.75, min>=0.60, var<=0.03
new labels: soft = 0.75*M + 0.25*(four-student mean),
  pixel weight clip(exp(-5*V_route)*exp(-5*V_student)*(1-|M-S|), 0.05, 1)
X3 = 448 + A + B pseudo labels + 8 GT, fresh U-Net init (no S2/S3 weights),
  batch 12 = 3 GT / 3 original / 6 new, group weights 1.0 / 0.75 / 0.50,
  40000 iterations, 2000-step ramp
B7 = (q_return * q_multi^2 * q_model^2)^(1/5), copy the winning SAM3 mask,
  all 100 test targets, no confidence filtering
```

Pool: **448 first pass → committee A 94 / B 113 / C 137 → X3 pool 655**
(+8 GT = 663 distinct training images). 137 Tier-C images stay out. `q_return` is
the return consistency computed with only the anchor's known GT; `q_multi` is the
image-level mean of the 21 pairwise candidate Dice.

## 3. Results

| Method | final mask source | Test Dice | Test IoU |
|---|---|---:|---:|
| SAM3 TP + independent Router (no student) | router-picked TP candidate | **0.885433** | **0.828274** |
| X3 single image | X3 direct prediction | **0.867161** | **0.786951** |
| X3-best + TP-only B7 | B7-picked TP candidate | **0.892639** | **0.834468** |
| B7 candidate Oracle (post-hoc only) | GT-best candidate | **0.907426** | not reported |

Validation: B7 0.839690 / IoU 0.774689, Oracle 0.869544; X3 single (export entry)
0.824269 / 0.738784. In-training best X3 validation was 0.821969 at iteration
24 600 (the checkpoint-selection criterion). Pairwise X3+B7 vs the frozen router:
**45 improve / 17 tie / 38 degrade**, mean **+0.007207**, paired bootstrap 95% CI
**[−0.005002, +0.025372]**.

448-pool S2/S3 test (added 2026-09-12, 256, threshold 0.5, per-image macro mean):

| student | checkpoint | iter | train val Dice | Test Dice | Test IoU |
|---|---|---:|---:|---:|---:|
| S2 | best | 32800 | 0.816123 | 0.846007 | 0.760456 |
| S2 | final | 40000 | 0.810172 | 0.838730 | 0.751171 |
| S3 | best | 29800 | 0.814231 | **0.860283** | **0.779007** |
| S3 | final | 40000 | 0.806084 | 0.853202 | 0.771334 |

X3's final checkpoint was **not** tested in V3 (`test_evaluated: false`; B7 uses
X3-best only).

## 4. Caveats recorded for V3

- **uint16 soft-label reader bug.** The old T24 loader called `convert('L')` on
  16-bit PNGs, saturating S3's 8 probability levels to 0/255 and every pixel
  weight to 255. V3 normalises by bit depth (uint16 ÷ 65535, uint8 ÷ 255) before
  any student training (pre-fix scripts kept in `code_before_uint16_fix`). So V3
  contains both the new TP-only pool **and** a reader fix; the difference against
  the old dual-route students cannot be attributed to the route change alone.
- Two evaluation entries must not be mixed: in-training best X3 validation
  0.821969 (the checkpoint-selection criterion) vs the fixed-checkpoint export
  entry 0.824269.
- **Supervision misalignment** (discovered later): `LongTwoStreamBatchSampler`
  shuffles all 12 samples while the loss still assigns GT to the first 6 and
  pseudo to the last 6. Reproducing 1000 batches at 8/580 scale, **999 were
  misaligned** (on average 2.974 pseudo labels in the first 6 slots). The old 448
  and 580 students both inherit the bug; the single-TP + router baseline does
  not. Old test numbers are real measurements of those checkpoints and should be
  kept **with** the implementation note, not used for causal claims about ideal
  recipes.
- Choosing the highest predicted quality ≠ meeting the pseudo-label bar; higher
  acceptance is not higher quality. The out-of-fold router diagnostic shows the
  new filter admits 71/100 validation images at mean 0.913386 (3 below 0.5) while
  the old rule admits 51 at 0.942856 (none below 0.5); the 20 new ones average
  0.838237 against the old pick's 0.843000.
- Dropping PC lowered the candidate Oracle, so the smaller Oracle gap does not by
  itself prove a better scorer.
- The B7 gain is inside a CI that crosses 0; `q_model` is agreement between two
  predicted masks, not target GT Dice.
- The test set had been examined many times before; later comparisons are
  exploratory on an already-used set.
- The X3-pool-v2 pipeline **failed**: S2 retraining completed, S3 was killed
  (`RuntimeError: train_S3 failed (-9)`), so no official new pool / new X3 exists.
- Reproduction must fix the environment: a Python 3.9 run stopped at a strict
  parameter-equality assertion; the original Python 3.12 environment reproduces
  parameter differences of 0 and 100/100 identical route selections.

---

# V4 — filter-first 580, single student, 620 rescreen, Round-2

Corresponds to `work/kvasir_tp_pseudo_filter_20260909/`,
`work/kvasir_tp_filterfirst_students_20260909/`,
`work/kvasir_x3_pool_v2_20260909/`, and the `mainline/` experiment tree.

## 1. Why this version exists

V3 selected first and checked afterwards: when the router's chosen candidate
failed the return gate, the **whole image** was dropped even though another
candidate might pass. Of the 160 images that failed only the return gate, **132
had another candidate with `q_return >= 0.95`**. Reordering raises the first pass
from 448 to 580 while the originally selected masks of the 448 stay identical.
On top of that, the fixed 6+6 batch quota plus shuffling is exactly the
supervision-misalignment bug, and the two-student committee was replaced by one
equal-weight student. Coverage gains were shown not to be quality gains — hence
the full retraining comparison.

## 2. Pipeline, end to end

```text
same 7 TP b0-b6 candidates, frozen SAM3-base, frozen independent Router
V4 order = check, then select:
  image-level q_multi >= 0.90? no -> out
  keep candidates with q_return >= 0.95; empty -> out
  frozen Router picks the highest-scoring admissible candidate
```

**Single equal-weight student (588 images).** 8 GT + 580 pseudo = 588; one
shuffled pass per epoch, **no GT resampling**; batch 12 → 49 iterations/epoch;
**816 epochs = 39 984 iterations**; **all sample and pixel weights are 1** (the
0.5 pseudo coefficient, the q_multi image weights and the ramp are all removed);
loss = per-image `BCE + SoftDice` (smooth 1) averaged over the batch; SGD lr 0.01,
momentum 0.9, wd 1e-4, per-epoch linear decay `0.01*(1-(e-1)/816)` constant within
an epoch; validation on the full 100 every 4 epochs (196 iterations); GT weak
augmentation / pseudo strong augmentation; 256×256.

Soft target: `C = {candidates with q_return >= 0.95}`, `M` = the frozen router's
pick inside `C`, `P = mean(M_k, k in C)`, **`Y = 0.75*M + 0.25*P`**; hard control
`Y = M`. Thresholding at 0.5 gives back `M`, so this softens confidence without
changing the binary boundary. Eligible-candidate counts over the 580 pool: 1→42,
2→54, 3→64, 4→67, 5→41, 6→30, 7→282; 538 images have non-binary soft pixels,
about 0.716% of pixels per image are softened.

**580 committee.** 212 remaining images re-audited → **A 21 / B 75 / C 116**;
X3 = 580 + 21 + 75 = **676** after de-duplication. Area quantile thresholds shift
with the pool (q01 0.01381897, q99 0.32972504 vs 0.01619675 / 0.28475357).

**620 full rescreen.** Audit all 792 images and 5544 candidates, no legacy-pool
privilege; candidates must be non-empty. Tier A: `q_return >= 0.95` and mean Dice
against the other 6 TP candidates `>= 0.95` (**students are not a hard gate**).
Tier B: `q_return >= 0.95`, TP mean Dice `>= 0.80`, Dice with the student binary
prediction `>= 0.85` and student prediction non-empty. Prefer A, else B; within a
tier sort with the frozen TP router. Label always
`Y = 0.75*M + 0.25*mean(return-eligible TP candidates)`; the teacher only gates
admission and never enters the target. All samples and pixels equal-weight; A/B do
not form separate weighting or sampling streams. Auditing student = the previous
stage's **soft validation-best, epoch 556, val Dice 0.827330** (SHA256
`c03b6a63…fc58`) — deliberately not the higher-test final. Training: 620 + 8 =
**628 images**, batch 12 (last batch 4) → **53 iterations/epoch**, 816 epochs =
**43 248 iterations**, starting from the **same saved initial U-Net weights**, not
from the auditing student.

**620 B7.** Formula unchanged, uses all 7 original SAM3 TP candidates with **no
training-pool gate**, copies the winning SAM3 mask, no pixel fusion; both
checkpoints' choices frozen before reading GT.

**X3 pool v2 (fixed rules, not completed).** Audit all 792; per candidate compute
return consistency, mean Dice against the other 6 candidates, and mean/min/variance
of Dice against the four teacher predictions. Both tiers require
`q_return >= 0.95`, non-empty candidate and ≥2 non-empty teacher predictions:

| tier | q_multi | teacher Dice mean | min | var cap |
|---|---:|---:|---:|---:|
| A | ≥0.90 | ≥0.90 | ≥0.80 | ≤0.01 |
| B | ≥0.75 | ≥0.75 | ≥0.60 | ≤0.03 |

Same-tier candidates are ordered by the frozen independent TP router (the
`0.2*q_multi + 0.8*q_model_mean` rule no longer decides the mask); the area-quantile
Tier-A gate is removed; all admitted images share soft label
`0.75*TP mask + 0.25*four-teacher mean` and one continuous pixel weight
`clip(exp(-5*TP var)*exp(-5*teacher var)*(1-|TP mask - teacher mean|), 0.05, 1)`;
image weight `tier_factor*(q_multi + q_model_mean)/2` with A = 0.75, B = 0.50 as
absolute values; the new X3 uses one pool with **6 GT + 6 uniformly sampled
pseudo** per batch.

**Round-2 SAM3 LoRA.** Full-module LoRA rank 16 / alpha 32 / dropout 0.1; AdamW
lr 5e-5, wd 0.01; original RGB bilinearly to **1008** (labels nearest to 1008),
Dice still macro-averaged at **256**; batch 1, images equal weight, one pass per
epoch, no GT oversampling, no warmup, no gradient accumulation; fixed text
`colon polyp`; validation-best on the full 100 each epoch (ties → earlier);
inference is single-image + fixed text, union of query masks with class score
≥ 0.5, mask probability threshold 0.5, **no TP / Router / B7 and no GT
points/boxes**.

## 3. Pool evolution

| stage | first pass | committee | student/X3 pool | rule difference |
|---|---:|---|---:|---|
| V3-448 | 448 | A 94 / B 113 / C 137 | X3 655 | select then check |
| V4-580 (old S2/S3 control) | **580** | A 21 / B 75 / C 116 (212 audited) | X3 676 | check then select |
| V4-580 single student | 580 | – | 588 | one student, equal weight |
| V4-448 (new recipe, original pool) | 448 | – | 456 | pool-size control |
| V4-620 rescreen | **620** | A 538 / B 82 / C 172 | 628 | all 792 re-audited, no privilege |
| V4-X3-pool-v2 preview (**preview_only**) | **644** | A 564 / B 80 / C 148 | not trained | new unified rules; pipeline failed |
| V4-Round-2 pool A | **428** | – | 436 | B7 rescreen, val threshold 0.97 |
| V4-Round-2 A1 / A2 | A1 **596**, A2 **503** | – | 604 / 511 | A1 hard `R >= 0.95`; A2 also `C, S >= 0.95` |

Details:

- Of the 132 newly admitted images, **115 were already Tier A/B** (73 A + 42 B)
  and only 17 were previously C — so 132 ≠ 132 extra X3 images.
- 620 vs 580: **all 580 retained** (538 A + 42 B, re-audited individually, not
  automatically), **0 removed, 40 added** (all via Tier B), **22 retained images
  changed their main mask**, `unlabeled_gt_masks_read = 0`.
- 620 validation side: the frozen rule admits **75/100**, whose SAM3 main masks
  average Dice **0.912085** (describes the admitted subset only).
- X3-pool-v2 preview vs the old 676 pool: **640 overlap, 36 out, 4 in**; the 580
  first-pass images split 557 A / 20 B / 3 C.
- Round-2 pool A vs 620: 423 retained, 197 removed, 5 added. Pool A vs 580: mean
  Dice 0.910379 → **0.934062 (+0.023683)**; over the 423 common images
  0.932220 → 0.933674 (+0.001454), 193 wins / 130 losses / 100 ties; 325 mask
  files changed hash.

## 4. Results

### 4.1 Router / X3 / B7 / Oracle (test, 100 targets)

| version | Router Dice/IoU | X3 single Dice/IoU | X3+B7 Dice/IoU | Oracle |
|---|---|---|---:|---:|---:|
| V3-448 (control) | 0.885433 / 0.828274 | 0.867161 / 0.786951 | 0.892639 / 0.834468 | 0.907426 |
| V4-580 (old S2/S3) | 0.885433 / 0.828274 | 0.853667 / 0.772762 | 0.890846 / 0.831564 | 0.907426 |
| V4-620 single student, best (epoch 576) | 0.885433 / 0.828274 | 0.858330 / 0.781490 | **0.894641** / 0.837132 | 0.907426 |
| V4-620 single student, final (epoch 816) | 0.885433 / 0.828274 | **0.869478** / 0.796199 | 0.893723 / 0.835849 | 0.907426 |

- 580 validation: X3 0.819411 / 0.737066, B7 0.851547 / 0.786831, Oracle
  0.869544; B7 validation delta vs 448 = +0.011857 [−0.000739, +0.029837]
  (14/15/71).
- 580 paired test: B7 **−0.001793** [−0.005390, +0.001539] (14/72/14);
  X3 **−0.013495** [−0.027368, −0.001083] (38/0/62) — the X3 drop is clearly
  outside 0.
- 620 B7 vs the original 448 B7: best +0.002001 [−0.000839, +0.005076],
  final +0.001084 [−0.002159, +0.004466]; vs the 580 B7: best +0.003795
  [+0.000594, +0.007316], final +0.002877 [+0.000380, +0.005983]; vs the TP
  independent router: best +0.009208, final +0.008290. Same-pool Oracle
  0.907426, so the realised gap is 0.012785 / 0.013703.

### 4.2 580-pool S2/S3 test (old recipe, filter-first pool)

| model | checkpoint | steps | train val Dice | Test Dice | Test IoU |
|---|---|---:|---:|---:|---:|
| S2 | val-best | 26000 | 0.824722 | 0.851530 | 0.771841 |
| S2 | final | 40000 | 0.817908 | 0.851675 | 0.774503 |
| S3 | val-best | 22800 | 0.824242 | 0.841958 | 0.760371 |
| S3 | final | 40000 | 0.803916 | 0.843769 | 0.765014 |

### 4.3 Single student, hard vs soft (588 images, 816 epochs / 39 984 iterations)

| arm | best epoch | Val Dice | Test Dice | Test IoU |
|---|---:|---:|---:|---:|
| hard | 436 | 0.819562 | 0.848592 | 0.769222 |
| soft | 556 | 0.827330 | 0.851243 | 0.773038 |

| arm | best test | final test | final IoU | final val |
|---|---:|---:|---:|---:|
| hard | 0.848592 | **0.854011** | 0.775123 | 0.812377 |
| soft | 0.851243 | **0.859530** | 0.783362 | 0.814561 |

Paired soft − hard: +0.002651 [−0.013394, +0.017837], 47 wins / 0 ties /
53 losses. 448-pool new-recipe control: best epoch 476 val 0.828689 / test
0.855191 / IoU 0.770759; final epoch 816 val 0.806991 / test 0.851452 / IoU
0.767965. Fixed-epoch pool shrink also changes total steps (39 984 → 31 008), so
it is not a fixed-compute comparison.

Key stage-2/3 deltas (`mainline/stage23_old_new_comparison_20260912/report.md`):
580 old S2 best − 448 old S2 best = +0.5523 pp; 580 old S3 best − 448 old S3 best
= **−1.8325 pp**; 580 new soft best − 448 old S3 best = −0.9039 pp; 448 new soft
best − 448 old S3 best = −0.5092 pp; 448 new soft best − 580 new soft best =
+0.3947 pp; 448 new soft final − 580 new soft final = −0.8078 pp.

### 4.4 620 rescreen

| checkpoint | epoch | Val Dice | Test Dice | Test IoU | vs old 580 soft pool |
|---|---:|---:|---:|---:|---:|
| best | 576 | 0.826621 | **0.858330** | 0.781490 | +0.007086 [−0.009256, +0.022945] (48/52/0) |
| final | 816 | 0.810647 | **0.869478** | 0.796199 | +0.009948 [+0.000333, +0.018744] (61/38/1) |

### 4.5 Propagation-endpoint fine-tune pilot (`tp_tracker_endpoint_20260910`)

Isolates whether endpoint fine-tuning helps, keeping the frozen 620 soft pool.
Pretraining check: reusing the old `forward_tracking` training interface vs the
public TP inference interface agrees at Dice 0.941157 / 0.979372 / 0.944759 /
0.897360 / 0.820794 / 0.431339, so the old interface **cannot** be treated as the
current TP baseline; the fix reproduces all forward behaviour, recomputes the
endpoint step with gradients enabled, matches historical masks exactly (max logit
difference 0), and disables the predictor's long-lived BF16 autocast weight cache.
Training: from the original SAM3-base, 1 epoch = 628 updates, one pass per target,
equal weight, pseudo targets balanced over `b0–b6` (96/90/89/89/88/88/88), GT
targets use another anchor's `b0`; anchor GT box only, no text; endpoint BCE +
soft Dice; LoRA only on memory attention and mask-decoder cross attention —
**52 modules, 90 112 parameters, rank 4 / alpha 8 / dropout 0**; AdamW 1e-5,
wd 0.01, clip 1; gradients only through the recomputed endpoint step (a
**length-1 truncated BPTT**, not end-to-end). Result on validation with the same
B7 formula: TP+B7 0.851807 → 0.842634, TP Oracle 0.869544 → **0.873182**;
validation-based selection picks **epoch 0 (keep base)**. No test run.

### 4.6 Round-2 pools and fine-tunes

Pool A: re-run B7 on all 792 (formula and tie-break unchanged, mask must be
non-empty, no legacy 620 privilege); the validation threshold rule was fixed
**before** reading the table ("keep validation mean Dice ≥ 0.95 with at least 20
images, take the highest coverage; otherwise take the best mean among n ≥ 20 and
flag it"). Chosen threshold **0.97** → validation keeps **34/100** at mean Dice
**0.958583**. Pool A = **428/792 (54.04%)**, pseudo mean Dice 0.934062, IoU
0.887872, median 0.961650, p10 0.887890, min 0.154452, 24 images below 0.8 and 47
below 0.9; with 8 GT the training set is **436**. Pool A's mean Dice is a
post-hoc audit, not a test score.

Ablation with the same B7 formula and threshold grid:

| pool | rule | B7 thr | val kept | val mean Dice | pseudo count | pool mean Dice | Dice<0.8 |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | B7 top candidate, then non-empty + total-score threshold | 0.97 | 34 | 0.958583 | 428 | 0.934062 | 24 |
| A1 | require non-empty and `R >= 0.95` first, then B7 top | 0.92 | 57 | 0.952134 | 596 | 0.911098 | 58 |
| A2 | also require `R, C, S >= 0.95` | 0.00 | 37 | 0.957821 | 503 | 0.923180 | 37 |

At matched count K = 428 the three pools' mean Dice is essentially identical
(0.934062 / 0.934064 / 0.934064). The pre-set validation rule selected **A1 and
A0** for training; the old A0 was stopped by the user at epoch 8 (7 epochs done,
best epoch 2 at val 0.8795421), so the planned two-pool 10-epoch comparison was
truncated and a new A0 was started.

New training settings (all pools): original RGB straight to 1008, cosine
5e-5 → 5e-7, gradient clipping 1.0, full-module LoRA rank 16 / alpha 32 /
dropout 0.1, batch 1, equal weight, 10 epochs; best chosen by `colon polyp`
validation Dice; test on all 100 at 256.

| pool | pseudo + GT | steps | best epoch | Val Dice | Test Dice | Test IoU | Test Dice (empty) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 428 + 8 = 436 | 4360 | 6 | 0.880909 | 0.901755 | 0.842425 | 0.885264 |
| A1 | 596 + 8 = 604 | 6040 | 4 | 0.887257 | 0.903294 | 0.847450 | 0.870346 |
| A2 | 503 + 8 = 511 | 5110 | 1 | 0.888680 | **0.912932** | 0.857916 | 0.845776 |

A2 − A1 = +0.009639 [−0.001056, +0.025148], 38 wins / 62 losses / 0 ties (per
target). A1 final val 0.874312, A2 final val 0.877710. A2 selected epoch 1 and
never exceeded it in the next 9 epochs.

Two seeds (2026 / 2027):

| pool | seed2026 | seed2027 | mean |
|---|---:|---:|---:|
| A0 | 0.901755 | 0.892288 | 0.897022 |
| A1 | 0.903294 | 0.886602 | 0.894948 |
| A2 | 0.912932 | 0.884289 | 0.898611 |

seed2027 detail: A0 best epoch 7 val 0.882076 / 0.892288 / empty 0.885013;
A1 best epoch 2 val 0.879658 / 0.886602 / 0.867812; A2 best epoch 2 val 0.881723
/ 0.884289 / 0.849984.

50-epoch A0 control (no clipping): 428 + 8, cosine 5e-5 → 5e-7 over 50 epochs,
**21 800 updates**. Best epoch 19: val 0.893717, test **0.901950**, IoU 0.843133,
empty 0.890174. Final epoch 50: val 0.873586, test 0.894245, IoU 0.836810, empty
0.884782 — final is **−0.007705** versus its own validation best.

A2 learning-rate / clipping ablation:

| setting | best epoch | Val Dice | Test Dice | Test IoU | empty-test Dice |
|---|---:|---:|---:|---:|---:|
| A2 clip 1.0 (5e-5 → 5e-7) | 1 / 3 | 0.888680 / 0.880555 | 0.912932 / 0.901113 | 0.857916 / 0.842727 | 0.845776 / 0.816486 |
| A2 seed2027 (clip 1.0) | 8 | 0.880205 | 0.889544 | 0.827697 | 0.808463 |
| A2 low lr (1e-5 → 1e-7, clip 1.0) | 3 | 0.880555 | 0.901113 | 0.842727 | 0.816486 |
| A2 low lr + no clip | 6 | 0.873855 | 0.894284 | 0.831481 | 0.764290 |
| A2 restored lr + no clip, best | 1 | 0.882945 | 0.897351 | 0.835200 | 0.863107 |
| A2 restored lr + no clip, final | 10 | 0.868210 | 0.901877 | 0.845183 | 0.879847 |

Low learning rate narrows the two-seed spread (0.028643 → 0.011569) without
raising the mean (0.898611 → 0.895329); disabling clipping costs −0.006829 test
Dice versus clip 1.0. The training-monitor label-fit Dice rises 0.967371 →
0.982954, which is a fit to training labels and not a real unlabelled Dice.

## 5. Caveats recorded for V4

1. **Round 2 does not beat round 1.** The three 10-epoch pools reach
   0.9018 / 0.9033 / 0.9129 on seed 2026, but the ordering fully reverses on
   seed 2027 (A2>A1>A0 vs A0>A1>A2). The best single number is not a stable
   level.
2. The 50-epoch control peaks at epoch 19 and decays; it also disables clipping
   and changes the cosine horizon, so it is **not** a clean epoch-length ablation.
3. Round-2 A0 changed input resolution, learning-rate schedule and clipping at
   once, and the old A0 was stopped at epoch 8 — differences cannot be attributed
   to one change.
4. Every one of the 4360 updates triggered clipping with post-clip norm
   ≤ 1.0001; that reflects the total loss scale and is **not** evidence of
   exploding gradients.
5. A2 picked epoch 1 and never improved; the filtering conditions are stricter
   but a single seed cannot show long-run superiority, and equal epochs mean
   different numbers of steps.
6. `empty text` is a pre-declared diagnostic; `colon polyp` is the primary prompt.
   The same A0 weights lose 0.016491 without text, while A1's earlier no-text
   gain does not generalise to A0.
7. Pool A's training-GT means are descriptive only; validation doubles as threshold
   calibration and checkpoint selection, so the independent number is the frozen
   test report.
8. Pool A's image set and mask choices both differ from the 580 pool, so the mean
   difference **cannot** be attributed to better B7 ordering.
9. Low learning rate narrows the seed spread but does not raise the mean; two
   seeds cannot establish a variance reduction, and label-fit Dice is not real
   unlabelled Dice, so label error cannot be called the sole cause of the
   generalisation gap.
10. **Coverage is not quality**: out-of-fold router diagnostic — old rule 51
    images at 0.942856 (0 below 0.5), new rule 71 images at 0.913386 (3 below
    0.5), the 20 new ones at 0.838237 vs the old pick's 0.843000. The 52/100
    figure elsewhere comes from the full-validation-fitted router, a different
    estimator. Do not treat the 580 as equally reliable hard labels just to raise
    acceptance.
11. 132 new images ≠ 132 new X3 images (115 were already A/B); the committee
    re-audits the new 212 and the manifest is de-duplicated.
12. The new single-student recipe does **not** beat the old 448 S3 by
    validation-best (0.904 pp lower), while by final it is 0.633 pp higher —
    best and final must be compared separately, never selected on test. The
    change bundles pool membership, soft target, sampling and loss weights, so it
    is a recipe comparison, not a single-factor ablation.
13. The 448-pool new-recipe control is not fixed-compute (39 984 → 31 008 steps);
    the 620 rescreen is a fixed-epoch, not fixed-iteration, comparison; its
    thresholds were frozen before reading the validation diagnostic.
14. The test set has been used many times and can no longer be called blind; the
    single-student hard/soft contrast is a single seed and its sampling/loss
    differ from the historical S2/S3, so label softening is not the only change.
15. The X3-pool-v2 pipeline failed (`train_S3 failed (-9)`); its 644-image pool is
    `preview_only` and the training entry refuses it. That round bundles a
    necessary implementation fix **and** a new X3 recipe, so it is not a
    single-factor ablation.

---

# V5 — automatic anchors, single TP

Corresponds to `mainline/experiments/automatic_anchor_selection_pilot_20260911/`,
`auto8_tp_validation_20260911/`, `automatic_anchor_tp_test_20260913/kvasir/`,
`automatic_anchor_tp_router_20260913/kvasir/`, and the `original_per_bridge` arm
of `cross_dataset_calibration_factorial_20260914/`.

## 1. Why this version exists

All of V1–V4 start from **8 hand-fixed** anchors. The fixed set is highly
concentrated (the top two sources account for 88.19% of train chains, 81.00% of
val, 85.29% of test; 763/792 train, 98/100 val, 97/100 test targets have a single
common source across all seven chains), and its GT area coverage is narrow
(0.47%–10.20%, median 4.41%). An anchor-count ablation showed the count is not
the driver: mean(b1–b6) is non-monotone in n (8→6 −0.025, 4 −0.050, 3 −0.048,
2 −0.087, 1 −0.013) and a single anchor can span −0.013 to −0.087 across seeds —
**anchor strategy matters more than anchor count**. V5 therefore changes only
where the anchors come from, holding the rest of the protocol fixed.

## 2. Anchor selection, exactly

```text
only train RGB is read; GT, masks, pseudo labels, conditioned features and
val/test features/metrics are never used
1008 features : RGB -> bicubic + antialias resize to 1008x1008 -> [0,1]
                -> (x-0.5)/0.5 ; frozen SAM3-base
                -> detector.backbone.vision_backbone.trunk first output
                -> 72x72 patches x 1024 dims
global descriptor : per-patch L2, then mean of normalised patches, then L2
local descriptor  : 64 patches per image, rows/cols = np.linspace(4,67,8).round(),
                    index = row*72 + col
dictionary        : MiniBatchKMeans(n_clusters=64, random_state=2026, n_init=3,
                    batch_size=2048, max_iter=100) over all train patches
IDF               : df[c] = #train images with hist[c] > 0
                    idf[c] = log((N+1)/(df[c]+1)) + 1
                    local[i] = L2(sqrt(hist[i] * idf[i]))
similarity        : S = 0.5*clip(GG^T,0,1) + 0.5*clip(HH^T,0,1)
objective         : F(A) = (1/N) * sum_i max_{a in A} S(i,a)  (monotone submodular)
selection         : greedy; train sorted by ID; exact-SHA256 duplicates keep the
                    first occurrence; K = round(0.01 * 800) = 8;
                    numpy argmax breaks ties to the earliest index
```

Main scheme = `global_local_facility`. Coverage over the 792 non-selected images
(a proxy, **not** Dice)
(`automatic_anchor_selection_pilot_20260911/report.md`):

| scheme | global | local | combined | max-anchor share |
|---|---:|---:|---:|---:|
| existing 8 | 0.986559 | 0.688024 | 0.835373 | 31.37% |
| `global_facility` | 0.988960 | 0.694420 | 0.839891 | 20.12% |
| **`global_local_facility`** | 0.987532 | **0.721408** | **0.852560** | **15.50%** |
| random ×100 mean | 0.986493 | 0.679539 | 0.830501 | 24.20% |

GT area audit after freezing: existing 8 → 0.47%–10.20% (median 4.41%);
`global_facility` → 6.13%–47.38% (18.92%); `global_local_facility` →
1.71%–22.72% (10.35%).

Frozen Kvasir anchors (greedy order), from
`automatic_anchor_selection_pilot_20260911/SELECTIONS_FROZEN.json`
`selected.global_local_facility[:8]`:

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

Reproduction check `verify_kvasir/selection_replay.json`: `exact_match = true`,
same order, `mask_pixels_read = 0`, `eligible_count = 800`,
`dictionary_max_abs = 0.0`, `histogram_max_abs = 0.0`.

## 3. Routes, propagation, router

- KNN/TP input 256, patch grid `256//14 = 18`, 324 normalised 1024-d vectors per
  image; `patch_mean` = L2 of the mean of normalised patches; bridge frames come
  from train only.
- Anchor prototype `p_A` = L2 of the mean of the foreground-grid patches (GT
  nearest-resized to 18×18); empty foreground grid falls back to the whole-image
  patch mean.
- `s_j = dot(p_A, x_j)`, `w_j = softmax(10 s_j)` (implemented as
  `exp((sims - max) * 10)`), `z = L2(sum_j w_j x_j)`, `TP(A,T) = dot(p_A, z)`.
  Only the `cond_target` mode is used; the PC family is not.
- Path: `b0` = A → T, `bN` = A → N train bridges → T; beam width 32; KNN ranks
  bridges by whole-image `patch_mean` cosine; path score is the `(min, mean)`
  lexicographic pair of the node condition scores plus adjacent `patch_mean`
  cosine. For each `(target, b)` only the single best path across all anchors is
  kept — that is the **`original_per_bridge`** rule, so V5 has 7 candidates per
  target, not 8×7.
- Prompt: anchor GT **tight box**, no text; propagation canvas 256; forward mask
  is the candidate; return consistency `q_cycle` initialises the backward pass
  with the **predicted** target mask (never target GT) and does not replace the
  forward mask.
- Router: **legacy 28 features** = 7 base (`bridge_count`,
  `path_bottleneck_similarity`, `path_mean_similarity`, `final_sam_score`,
  `final_candidate_count`, `bridge×bottleneck`, `bridge×path_mean`) + 21
  propagation features (`q_cycle`, `cycle_success`, `cycle_sam_score`, 8 area
  terms, empty count, 2 component terms, centroid step, 2 bbox terms, 3
  adjacent-Dice terms, 3 SAM-score terms, 2 candidate-count terms); standardised
  Ridge(alpha=1) regressing candidate Dice; image-grouped 5-fold, seed 2026, all
  7 candidates of an image in the same fold; configuration chosen by OOF, plus a
  nested outer-5/inner-4 estimate of the whole selection procedure; models and
  test choices are frozen **before** any test metric is read. The router only
  chooses among existing candidates, so it cannot change the Oracle.

## 4. Results

Validation (100 targets) with the **old, unfitted** router (descriptive only)
(`auto8_tp_validation_20260911/report.md`):

| bridge | original 8 | automatic 8 |
|---|---:|---:|
| b0 | 0.773972 | 0.797777 |
| b1 | 0.808410 | 0.829016 |
| b2 | 0.817428 | 0.828908 |
| b3 | 0.808974 | 0.837240 |
| b4 | 0.811338 | 0.843508 |
| b5 | 0.820067 | 0.844415 |
| b6 | 0.842460 | 0.850480 |
| Oracle | 0.869544 | **0.889347** |
| unfitted Router | 0.851917 | 0.840154 |

Automatic anchors lift every bridge and the ceiling, but the unfitted router
realises less (gap 0.017627 → 0.049193). Automatic − original: Oracle +0.019803
[−0.003072, +0.047191], Router −0.011763 [−0.048631, +0.024355] — both span 0.

Test (100 targets) (`automatic_anchor_tp_test_20260913/kvasir/report.md`):

| bridge | automatic 8 Dice | IoU |
|---|---:|---:|
| b0 | 0.804710 | 0.740799 |
| b1 | 0.863804 | 0.802767 |
| b2 | 0.858386 | 0.793706 |
| b3 | 0.862556 | 0.798998 |
| b4 | 0.874749 | 0.812229 |
| b5 | 0.865332 | 0.801673 |
| b6 | 0.887069 | 0.825453 |
| **7-candidate Oracle** | **0.917615** | – |

(The original-anchor Oracle on the same protocol was 0.907426.)

Independent router (`automatic_anchor_tp_router_20260913/kvasir/results.json`):

| item | value |
|---|---:|
| selected configuration | `legacy_a1` |
| validation OOF Dice | **0.851317** |
| nested (whole selection procedure) OOF | 0.832701 |
| test Dice / IoU | **0.871374 / 0.809810** |
| test Oracle / distance to Oracle | 0.917615 / 0.046241 |
| fixed `b6` (validation-chosen) | 0.887069 / 0.825453 |

`rank_peer_a1/a10/a100` validation values are 0.829937 / 0.834230 / 0.835469 —
below `legacy_a1` — so Kvasir selects the legacy configuration. The router is
**1.5695 points below** the fixed b6 candidate; the paired interval
[−0.037478, +0.001524] cannot prove a router benefit, and the validation ranking
improvement did not carry over. ISIC2018, by contrast, gains +1.3499 pp over
fixed b6 ([0.004297, 0.024660]) but the new `rank_peer` ordering is not better
than the legacy router (−0.000210, [−0.007163, +0.006470]).

**A separate arm — `raw_top1`.** The four-arm factorial defines `raw_top1` as
"keep the single rank-1 anchor per target by raw `TP(A,T)` and run `b0–b6`",
which is **not** the same rule as `original_per_bridge`: test 0.870848 / IoU
0.808683, Oracle 0.912017, validation OOF 0.848638, validation Oracle 0.889982.
The factorial README keeps the historical per-bridge control separate for exactly
this reason.

## 5. Caveats recorded for V5

- Coverage is a proxy, not Dice; the selection pilot ran **no** TP propagation and
  produced no new segmentation score.
- The automatic-stop rule (stop after 3 consecutive gains below 1% of the
  remaining mean distance) fired at 14 (`global_facility`) and 11
  (`global_local_facility`); the threshold was never validated on segmentation and
  must not be read as a minimum sufficient label count.
- Descriptor nearest-neighbour assignment is not the actual TP chain source share.
- There is no patient/video ID for near-duplicate removal; only exact SHA256
  duplicates are dropped.
- Greedy selection has **no downstream-Dice evidence against random anchors** —
  the 100 historical random sets only measured coverage (`No propagation
  performed`).
- The unfitted-router validation row is a development-set descriptive control,
  not an independent generalisation estimate.
- Test has been examined across rounds and is not a blind set; validation GT is
  extra supervision beyond the 1% anchor budget; the Oracle is not a deployable
  score and candidate counts must be reported separately.
- A single anchor-set experiment cannot establish stable superiority, and the
  Oracle / Router intervals both span 0.

---

# V6 — automatic anchors, raw TP top-2

Corresponds to the `raw_top2` arm of
`mainline/experiments/cross_dataset_calibration_factorial_20260914/`.

## 1. Why this version exists

V5's fixed b6 candidate alone reached 0.887069 on test while the router realised
0.871374 and the 7-candidate Oracle was 0.917615. The pool is narrow. V6 keeps the
**top-2** anchors under the raw TP score and lets each generate `b0–b6`, doubling
the pool to 14 candidates.

## 2. Pipeline delta

```text
per target: rank the 8 automatic anchors by raw TP(A,T) descending
            (anchor ID ascending breaks ties); keep the top 2
            each kept anchor generates b0-b6  -> 14 candidates per target
inside a path the original TP and adjacent patch_mean cosine still drive the
(bottleneck, mean) lexicographic score -- unchanged
router: the same legacy-28 Ridge(alpha=1), fitted per group, same image-grouped
5-fold (identical fold file, md5 013d10c2bf289096df28907cfa1644ae), no extra
hyper-parameter search
freeze order: validation OOF -> fit on all validation -> models_frozen.json
(test_GT_read=False) -> test_choices_frozen.json (test_GT_read=False) -> read
test GT; completion_audit confirms test_choices_frozen_before_GT_read=true
```

## 3. Results

| group | candidates | val OOF | val Oracle | test Dice | test IoU | test Oracle | test gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw_top1` | 7 | 0.848638 | 0.889982 | 0.870848 | 0.808683 | 0.912017 | 0.041169 |
| **`raw_top2` (V6)** | 14 | 0.846724 | 0.926433 | **0.891226** | **0.827754** | **0.932897** | 0.041671 |
| `centered_top1` | 7 | 0.852949 | 0.906218 | 0.860331 | 0.799370 | 0.906667 | 0.046337 |
| `centered_top2` (V7) | 14 | 0.853773 | 0.923757 | 0.862761 | 0.798492 | 0.937542 | 0.074782 |
| `original_per_bridge` (V5) | 7 | 0.851317 | 0.889347 | 0.871374 | 0.809810 | 0.917615 | 0.046241 |

Per-bridge best-of-2 Dice (recomputed from `pool_membership_frozen.json` +
`test_candidate_metrics.json`; the Oracle reproduces):

| bridge | best-of-2 Dice |
|---|---:|
| b0 | 0.877797 |
| b1 | 0.912387 |
| b2 | 0.903830 |
| b3 | 0.898346 |
| b4 | 0.895539 |
| b5 | 0.899515 |
| b6 | 0.908555 |

This is **not** a per-bridge router result — the router emits one final choice.
The `fixed_rank1_b0_b6` field in `results.json` records only the rank-1 anchor's
`b0–b6` (0.800809 / 0.857833 / 0.858343 / 0.863679 / 0.875074 / 0.865583 /
0.887149), identical to `raw_top1`.

Paired comparisons:

| comparison | split | mean | 95% CI | improved / worsened |
|---|---|---:|---|---:|
| `raw_top2 − raw_top1` | test | **+0.020378** | [−0.001077, +0.047059] | 40 / 37 |
| `raw_top2 − raw_top1` | val OOF | −0.001914 | [−0.034593, +0.030049] | 37 / 28 |
| `centered_top2 − original_per_bridge` | test | −0.008614 | [−0.040066, +0.020608] | 44 / 35 |
| `centered_top2 − raw_top2` (net calibration effect) | test | −0.028465 | [−0.065831, +0.004531] | 41 / 30 |

Reference usage (test): selected anchors 25 / 35 / 11 / 11 / 11 / 6 / 1 for the
seven used anchors; candidate chains 350 / 343 / 238 / 231 / 119 / 91 / 21 / 7.
Historical candidate `seconds` per target: `raw_top2` 31.469 vs
`original_per_bridge` 15.624 — roughly double the propagation work, but this is a
workload reference, not a controlled end-to-end timing.

## 4. Caveats recorded for V6

- The four arms are pre-registered and nothing is tuned on test; test had already
  been used for diagnosis.
- **V6 has the highest test mean but validation does not support selecting it**:
  its validation OOF (0.846724) is below the historical router (0.851317) and
  below `centered_top2` (0.853773). Its test lead over the automatic-anchor fixed
  b6 (0.887069) is only about 0.4156 pp. It must not be promoted to a baseline on
  test evidence alone.
- More candidates do not guarantee a better realised score: on ISIC the Oracle
  rises 0.883608 → 0.903881 while the realised Dice moves only +0.001048.
- All paired test intervals include 0.
- Do not construct a selector from the test Oracle.
- V6 has no per-bridge router result; the Oracle is an upper-bound diagnostic and
  groups with different candidate counts must be reported separately.
- The router uses extra validation GT; "1%" is the anchor-labelling budget only.

---

# V7 — automatic anchors + calibrated scores + top-2

Corresponds to the `centered_top2` arm of
`mainline/experiments/cross_dataset_calibration_factorial_20260914/`, plus
`mainline/reproduction_guides/automatic_selection_20260914/{P1_pilot,P2_fullbank,E0_diagnostics}_20260914/`.

## 1. Calibration, exactly

```text
mu_A          = mean_{x in train RGB} TP(A, x)      # no GT is read
centered(A,T) = TP(A,T) - mu_A
per target    : rank anchors by centered(A,T) descending, keep the top 2
                each kept anchor generates b0-b6  -> 14 candidates
inside a path the original TP and adjacent cosine still drive the
(bottleneck, mean) score: the centred value only ranks anchors and is never
mixed with the ~0.8-scale edge scores (centred values are near zero and can be
negative)
reference GT is used only for the already-selected 8/21/5 anchors (prototype,
initial box, return consistency); other train GT is never read
```

`mu_A` is computed from all train RGB of each dataset — Kvasir 800, ISIC2018
2075, BUSI 517, TN3K 2303. (Note: BUSI de-duplicates to 516 eligible files while
the mean still uses the original 517 train RGBs; the source files do not explain
the discrepancy.)

Why subtracting the mean removes the bias (hypothesis P4 in `PAPER_OUTLINE.md` §3,
labelled "assumption + empirical test"): if `s(A,T) = q(A,T) + b_A + ξ` and
`E_train[q(A,·)]` is approximately independent of A, subtracting `mu_A` cancels
`b_A` and leaves the ordering unbiased, with the benefit increasing in
`Var_A(mu_A)`.

Unconditional mechanism evidence on the BUSI full bank (E0 §E0.3b, 64 targets ×
35 candidates = 2240 records): `rho(raw, Dice)` +0.139 pooled / +0.005 per-target
vs `rho(centered, Dice)` **+0.380** / **+0.183**. Anchor choice: raw-score argmax
0.5923 vs calibrated argmax **0.6921** vs per-target best 0.7691 — a gain of
**+0.0998** [0.0317, 0.1702], 56.2% per-target win rate, moving the achieved
fraction of the selection ceiling from 77.0% to 90.0%. The intuitive extreme is
BUSI's `malignant (187)`: highest train mean (0.8607) but the lowest b0 validation
Dice (0.4817), while the nearly discarded `benign (3)` has mean 0.5110 yet
validation Dice 0.6064.

## 2. Bias-ratio diagnostic

```text
bias_ratio = std_A(mu_A) / std_T( max_A s(A,T) )
```

| dataset | anchors | std(mu_A) | **bias ratio** | test re-selection | concentration raw → cal | raw top-1 ties |
|---|---:|---:|---:|---:|---|---:|
| Kvasir | 8 | 0.0216 | **0.68** | 43% | 0.360 → 0.220 | 12% |
| ISIC2018 | 21 | 0.0449 | **1.10** | 65% | 0.235 → 0.158 | 9% |
| BUSI | 5 | 0.1448 | **5.42** | 85% | **0.985 → 0.348** | 0% |
| TN3K | 23 | not found | not found | 100% (25/25) | not found | not found |

Validation re-selection rates are 56% (Kvasir), 66% (ISIC), 80% (BUSI). BUSI
cross-check: E0's recomputed `mu_A` matches `calibration_frozen.json` to
0.00e+00. Tie rate matters because with raw scores the argmax is arbitrary for
that fraction of targets.

Calibration benefit vs bias ratio:

| dataset | bias ratio | ΔDice (b0, k=1) | ΔDice (b0, k=2) | ΔDice (full depth, k=1) |
|---|---:|---:|---:|---:|
| BUSI | 5.42 | **+0.2063** [+0.116,+0.302] | **+0.1159** [+0.051,+0.191] | **+0.0788** [+0.022,+0.141] |
| ISIC2018 | 1.10 | +0.0103 [−0.009,+0.030] | +0.0115 [−0.001,+0.025] | not measured (P3) |
| Kvasir | 0.68 | −0.0239 [−0.078,+0.029] | −0.0045 [−0.028,+0.022] | val +0.0167 / test −0.0113 (CI spans 0) |

Strength of the "bigger bias ratio → more useful calibration" rule: monotone over
three datasets, but only **two full-depth points** (BUSI significant, Kvasir
null), the ISIC full-depth point is missing, and the magnitude is far below what
the bias ratio would suggest. The benefit shape differs too: BUSI is a broad
improvement (56% win rate, median +0.027, 38% baseline catastrophe rate,
62% rescued), while ISIC/Kvasir have median 0 and win rates below 50% — the mean
gain comes entirely from the right tail ("tail insurance", concentrated on
high-margin targets where the raw score was confidently wrong). TN3K has an even
larger train-mean span (0.5502 vs BUSI's 0.3498) and a positive per-bridge
calibration gain at every depth, yet its four-arm router result gets **worse**
under calibration — so a large span does not imply the router can cash it in.

## 3. V7 results on Kvasir

| group | candidates | val OOF | val Oracle | test Dice | test IoU | test Oracle | test gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw_top1` | 7 | 0.848638 | 0.889982 | 0.870848 | 0.808683 | 0.912017 | 0.041169 |
| `centered_top1` | 7 | 0.852949 | 0.906218 | 0.860331 | 0.799370 | 0.906667 | 0.046337 |
| `raw_top2` | 14 | 0.846724 | 0.926433 | 0.891226 | 0.827754 | 0.932897 | 0.041671 |
| **`centered_top2` (V7)** | 14 | **0.853773** | 0.923757 | 0.862761 | 0.798492 | **0.937542** | **0.074782** |
| `original_per_bridge` | 7 | 0.851317 | 0.889347 | 0.871374 | 0.809810 | 0.917615 | 0.046241 |

Historical legacy reproduction: validation OOF 0.8513170057291387, test
0.8713744761396023, bit-identical to the archive.

Paired test deltas (10 000-image bootstrap, seed 2026, exploratory, no
multiplicity correction):

| comparison | delta | 95% CI | improved / worsened |
|---|---:|---|---:|
| `centered_top1 − raw_top1` | −0.010517 | [−0.031494, +0.004702] | 39 / 27 |
| `centered_top2 − raw_top2` | **−0.028465** | [−0.065831, +0.004531] | 41 / 30 |
| `raw_top2 − raw_top1` | +0.020378 | [−0.001077, +0.047059] | 40 / 37 |
| `centered_top2 − centered_top1` | +0.002430 | [−0.024604, +0.029778] | 34 / 37 |
| `centered_top2 − original_per_bridge` | −0.008614 | [−0.040066, +0.020608] | 44 / 35 |
| interaction | −0.017948 | [−0.053177, +0.015024] | – |

All Kvasir test intervals cross 0; so do the validation paired OOF values
(`centered_top1 − raw_top1` +0.004311 [−0.030271, +0.038264];
`centered_top2 − raw_top2` +0.007049 [−0.028426, +0.040778];
`centered_top2 − original` +0.002456 [−0.034761, +0.038912]).

**Why Kvasir is the negative control.**

1. Calibration cannot be declared a universal gain: BUSI gains clearly, ISIC's
   calibrated top-2 has the highest validation mean but only +0.9291 pp over the
   historical router on test with a CI containing 0, and **Kvasir's realised Dice
   actually falls**.
2. On Kvasir the problem is **selection**, not candidate quality: the calibrated
   top-2 pool has the **highest Oracle of all four arms** (0.937542 > raw top-2's
   0.932897) yet the router realises only 0.862761, leaving the largest gap in
   the table (0.074782). `centered_top1`'s Oracle *also* drops (0.906667 <
   0.912017), so calibration changes retained-candidate quality as well as the
   router.
3. More candidates do not guarantee a better realised score (ISIC: Oracle
   0.883608 → 0.903881, realised +0.001048).
4. The raw top-2's 0.891226 must not be promoted on test alone — its validation
   OOF (0.846724) is below both the historical router (0.851317) and
   `centered_top2` (0.853773).
5. Cross-dataset mechanism confirmation: on Kvasir the calibrated rank-1
   single-anchor, no-router per-bridge Dice is **lower than the raw rank-1 at
   every bridge** (raw 0.800809 / 0.857833 / 0.858343 / 0.863679 / 0.875074 /
   0.865583 / 0.887149 vs cal 0.780805 / 0.831111 / 0.844340 / 0.843392 /
   0.861184 / 0.850350 / 0.875166).

Router-selected anchor distribution over the 100 test targets (8 anchors, short
names by list position): raw top-1 0/34/17/16/20/6/5/2; centered top-1
9/22/7/9/10/7/22/14; raw top-2 0/25/35/11/11/11/6/1; centered top-2
11/13/15/12/4/5/16/24. (BUSI is starker: raw top-1 sends 65/66 targets to
`malignant (187)`; under calibrated top-2 that anchor gets 0 and `benign (3)`
rises to 29.)

## 4. P2 — the Kvasir full bank

56 candidates per target (8 anchors × `b0–b6`), validation and test, 11 200
propagations, of which 9448 newly computed; ~2–3 h GPU; zero new labels, zero
model change. Completeness: val 5600/5600 and test 5600/5600 success; the
original 700 test routes reproduce bit-exactly (difference 0.00e+00), validation
route_ids 700/700 with only a BLAS last-bit `path_mean` difference of 9.9e-8;
1752 reused masks verified by SHA256.

| k | candidates | val raw | val cal | Δcal (val) | test raw | test cal | Δcal (test) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 0.8895 | 0.9062 | +0.0167 [−0.005,+0.044] | 0.9180 | 0.9067 | **−0.0113** [−0.032,+0.004] |
| 2 | 14 | 0.9265 | 0.9238 | −0.0027 | 0.9328 | 0.9388 | +0.0059 |
| 3 | 21 | 0.9328 | 0.9371 | +0.0043 | 0.9432 | 0.9455 | +0.0022 |
| 5 | 35 | 0.9400 | 0.9405 | +0.0004 | 0.9525 | 0.9510 | −0.0015 |
| 8 | 56 | 0.9458 | 0.9458 | 0.0000 | 0.9542 | 0.9542 | 0.0000 |

At k = 8 the two pools are identical (only the order differs), so Δ = 0 is the
built-in self-consistency check. All Δ intervals cross 0 and val/test signs are
opposite. The b0-only version behaves the same way (val k=1 +0.036077
[−0.008150, +0.084669]; test k=1 −0.023905 [−0.078129, +0.028699]; test k=2
−0.004508; k=3 −0.007339; k=5 +0.001556).

Adjacent-`k` ceiling gains (paired bootstrap):

| interval | validation | test |
|---|---|---|
| k=1→2 | **+0.0370** [+0.015,+0.066] | **+0.0148** [+0.007,+0.025] |
| k=2→3 | +0.0063 [+0.002,+0.015] | +0.0104 [+0.002,+0.022] |
| k=3→5 | +0.0072 [+0.002,+0.017] | +0.0093 [+0.005,+0.015] |
| k=5→8 | +0.0058 [+0.002,+0.010] | +0.0017 [+0.001,+0.003] |

All significantly positive, decaying fast — consistent with
`H_k ⊆ H_{k+1} ⇒ Oracle` monotone.

Budget curve (greedy anchor prefixes, full-depth Oracle):

| labelled anchors | val Oracle | test Oracle | test vs K=1 |
|---:|---:|---:|---:|
| 1 (0.125% of train) | 0.8822 | **0.9142** | — |
| 2 | 0.9027 | 0.9239 | +0.0097 |
| 4 | 0.9367 | 0.9440 | +0.0299 |
| 8 (1%) | 0.9458 | 0.9542 | +0.0400 |

One labelled image already reaches a 0.9142 test ceiling; the full 1% adds only
+0.040. Four caveats: this is a **ceiling**, not a deployable score; K = 1 is the
first greedy pick, not a random single image and not "label one image and ship";
only Kvasir has this curve; and K is the anchor-labelling budget while the router
still uses validation GT.

## 5. Three levers substitute for each other

| dataset / split | k=1→2, b0 only | k=1→2, full depth |
|---|---:|---:|
| Kvasir test | **+0.0689** | **+0.0148** [+0.007,+0.025] |
| Kvasir validation | **+0.0899** | **+0.0370** [+0.015,+0.066] |
| BUSI validation | **+0.1603** | **+0.1079** |
| ISIC2018 test | **+0.0389** | not measured (P3) |
| ISIC2018 validation | **+0.0366** | not measured (P3) |

The same "keep one more anchor" move pays far more on shallow paths than on deep
ones. Depth also compensates for a bad anchor: BUSI's calibration gain falls from
+0.2063 at `b0` to +0.0788 at full depth (about 40%), and the gain decays to zero
by k = 3 (BUSI has only 5 anchors): Δb0 = +0.2063 → +0.1159 → +0.0076 → 0.0000
for k = 1/2/3/5. So calibration, breadth `k` and depth `m` all enlarge
"effective evidence" and are **substitutes**, not independent contributions.

Depth selection itself: only `q_cycle` is a genuine per-target signal. On ISIC it
beats the best fixed depth (0.861732 vs 0.856643); on Kvasir (0.872898 vs
0.887069) and BUSI (0.530451 vs 0.539444) it loses. `path_bottleneck` barely
varies with depth and `path_mean` rises monotonically, so their argmax is
effectively "always take b6" and is not a depth selector.

## 6. Cross-dataset arm

BUSI (`busi_calibration_factorial_20260914`): train 517 / val 64 / test 66, 5
fixed automatic anchors.

| group | candidates | val OOF | val Oracle | test Dice | test IoU | test Oracle | test gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw top-1 | 7 | 0.616268 | 0.675247 | 0.565990 | 0.479339 | 0.617141 | 0.051151 |
| **centered top-1** | 7 | 0.728772 | 0.754000 | **0.719992** | 0.635915 | 0.775461 | 0.055469 |
| raw top-2 | 14 | 0.665683 | 0.783128 | 0.656084 | 0.564468 | 0.776347 | 0.120263 |
| **centered top-2** | 14 | **0.738958** | **0.823477** | **0.752198** | **0.668155** | **0.823649** | 0.071452 |
| historical per-bridge | 7 | 0.617634 | 0.675247 | 0.566808 | 0.480026 | 0.617141 | 0.050333 |

Paired test deltas: centered top-1 − raw top-1 **+0.154001** [+0.076563,
+0.236401]; centered top-2 − raw top-2 **+0.096114** [+0.021322, +0.173987];
centered top-2 − historical control **+0.185389** [+0.104277, +0.270905];
raw top-2 − raw top-1 +0.090094 [+0.017658, +0.165548]; centered top-2 −
centered top-1 +0.032206 [−0.014657, +0.081100]; interaction −0.057888
[−0.139379, +0.016294]. Validation paired intervals: centered top-1 − raw top-1
[0.044333, 0.185253], centered top-2 − raw top-2 [0.011933, 0.141118], top-2 −
top-1 after calibration [−0.023520, 0.046310]. Without calibration, widening to
top-2 raises the Oracle (0.776347 ≈ centered top-1's 0.775461) but not the
realised score (0.656084 < 0.719992) — the gap blows up to 0.120263.

The preceding BUSI experiment (`busi_calibrated_multi_anchor_20260913`) froze
**K = 2, legacy alpha = 1** on validation (OOF 0.738958; nested OOF of the whole
configuration choice 0.709939); the 41 extra `calibrated_peer` features did not
win. Calibrated rank-1 per-bridge test Dice: 0.705708 / 0.709401 / 0.679150 /
0.689764 / 0.662677 / 0.698703 / 0.718360, 7-candidate Oracle 0.775461 (raw
version: 0.458291 / 0.514197 / 0.514843 / 0.533733 / 0.517095 / 0.514702 /
0.539444, Oracle 0.617141). In the raw version 65/66 b0 targets and 455/462 paths
came from `malignant (187)`.

The earlier BUSI router study (`busi_auto5_tp_router_20260913`) also illustrates
the same pattern: the router scores 0.566808 vs the validation-selected fixed b5
at 0.514702 (+5.2106 pp, [0.012606, 0.101015]), but on validation the best fixed
depth is b5 at 0.644838 — the router did **not** beat the best fixed depth there.
Its failure diagnosis records raw b0 anchor choice 0.494374 → calibrated 0.700627
(+20.6253 pp, [0.116289, 0.301813]), the five-anchor b0 Oracle 0.792824 (above the
old 7-bridge pool's 0.675247, directly showing discarded cross-anchor candidates
had value), and text-only prompting producing 60/64 empty candidates (0.058197).
A separate mask-prompt ablation is **completely identical** to the box version
candidate-by-candidate (val 448/448, test 462/462): b6 +1.7096 pp but the router
−3.1259 pp and the Oracle −1.3621 pp, i.e. no overall improvement, and the change
cannot be attributed to box→mask alone because the box version takes the
highest-scoring object per frame while the mask version tracks a specified object.

BUSI vs a trained low-label baseline (same 5 labels, same 517/64/66 split,
`results/busi_1pct_protocol/result.md`; SynFoC = MedSAM ViT-B LoRA(rank 4) + UNet,
EMA teacher, 40 000 iterations / 80 epochs, seed 2026, model selection on the
64-image validation only):

| method | BUSI test Dice |
|---|---:|
| raw top-1 | 0.565990 |
| raw top-2 | 0.656084 |
| SynFoC MedSAM+LoRA (validation-selected branch) | 0.653120 |
| SynFoC UNet (other branch, not selected) | 0.683123 |
| centered top-1 (no router) | 0.719992 |
| **centered top-2 + router** | **0.752198** |
| centered top-2 Oracle (upper bound) | 0.823649 |

SynFoC validation: MedSAM+LoRA 0.654125 (iter 37500), UNet 0.647518 (iter 38000);
test 0.653120 / 0.683123. The UNet branch scores higher on test but validation did
not select it, so it is **not** switched on test. Single seed, single run — a few
Dice points should not be over-interpreted. SynFoC is an end-to-end fine-tuned
network producing one mask per image, so it belongs next to the single-prediction
/ router rows, not next to the Oracle.

ISIC2018 (`isic2018` arm): train 2075 / val 259 / test 260, 21 automatic anchors.

| group | candidates | val OOF | val Oracle | test Dice | test IoU | test Oracle | test gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw top-1 | 7 | 0.861163 | 0.886672 | 0.868228 | 0.789278 | 0.883608 | 0.015380 |
| centered top-1 | 7 | 0.876904 | 0.901635 | 0.871717 | 0.794094 | 0.890596 | 0.018879 |
| raw top-2 | 14 | 0.868854 | 0.903707 | 0.869276 | 0.793486 | 0.903881 | 0.034605 |
| **centered top-2** | 14 | **0.880950** | **0.917948** | **0.875864** | **0.801266** | **0.911909** | 0.036045 |
| historical per-bridge | 7 | 0.862396 | 0.887467 | 0.866573 | 0.787929 | 0.884316 | 0.017743 |

| comparison | validation (259) | test (260) |
|---|---|---|
| centered top-1 − raw top-1 | **+0.015742** [+0.000417, +0.031808] | +0.003489 [−0.007685, +0.014676] |
| centered top-2 − raw top-2 | **+0.012096** [+0.000883, +0.024890] | +0.006588 [−0.003581, +0.018563] |
| raw top-2 − raw top-1 | +0.007692 [−0.002029, +0.018074] | +0.001048 [−0.012279, +0.013093] |
| centered top-2 − centered top-1 | +0.004046 [−0.002022, +0.011855] | +0.004147 [−0.006658, +0.014842] |
| centered top-2 − historical control | **+0.018554** [+0.004823, +0.033475] | +0.009291 [−0.001039, +0.019631] |
| interaction | −0.003646 [−0.015699, +0.008491] | +0.003099 [−0.010321, +0.017579] |

So the **full validation set supports calibration**, but every test interval
crosses 0 — the calibrated top-2's lead over the historical router is only
+0.9291 pp on test. (The earlier 21-anchor ISIC validation run,
`isic2018_auto21_tp_validation_20260911`, gave original-21 OOF 0.754020 / Oracle
0.772915 vs automatic-21 OOF 0.862396 / Oracle 0.887467, with no test at that
time.)

## 7. TN3K — complete, with its own failure story

The TN3K arm finished on 2026-09-16 (train 2303 / validation 576 / test 614, 23
automatic anchors). Full numbers, the paired comparisons, the anchor/target
**scale-mismatch** root cause, the falsified prompt-box enlargement probe and the
"good candidates exist, ranking loses them" evidence are written up in
[`cross_dataset_1pct.md`](cross_dataset_1pct.md) §TN3K; the artifacts are in
`mainline/experiments/tn3k_busi_factorial_20260915/` plus
`tn3k_failure_diagnosis_20260916/`, `tn3k_boxscale_probe_20260916/` and
`tn3k_reference_ranking_pilot_20260916/`.

Headline: test raw top-1 0.519585, centered top-1 **0.556705**, raw top-2
**0.569973**, centered top-2 0.560661, historical per-bridge 0.515486. Calibrated
top-1 beats raw top-1 by **+0.0371** [0.0094, 0.0654] and the interaction is
**−0.0464** [−0.0737, −0.0192], so TN3K is a third regime — neither BUSI (both
arms help) nor Kvasir (neither does).

What it implies for the Kvasir ladder: V7's calibration is **dataset-gated**, and
the gating is not just the bias ratio. On TN3K the raw score does degenerate
(train-mean span 0.5502, above BUSI's 0.3498) but degenerates onto the anchor that
is *already* the per-bridge best, so there is nothing to repair; on BUSI the raw
score is systematically wrong. The real TN3K bottleneck is a **scale-blind
ranking**: the all-23-anchor `b0` Oracle is 0.828171 while the top-2 `b0` Oracle is
0.618, i.e. the pool has good answers and retention loses them.

## 8. Limitations and open items recorded for V7

1. Calibration is not a universal fix — neutral at bias ratio ≈ 1, useless below 1.
2. ISIC2018 **full-depth** calibration is unmeasured (P3, ~12 h GPU).
3. The router uses validation GT; "1%" is the *anchor-labelling* budget only.
4. Test has been examined across many rounds — not a blind test.
5. Greedy coverage still has **no downstream-Dice evidence against random
   anchors** (P4, ~4 h); the 100 historical random sets measured coverage only.
6. Score → quality rank correlation is only ~0.1–0.4 (Kvasir `rho(q_cycle)` 0.206,
   bottleneck 0.253, path_mean 0.243; ISIC 0.089 / 0.358 / 0.260; BUSI −0.018 /
   −0.164 / −0.066; calibrated BUSI 0.499 / −0.361 / −0.259) — the intrinsic
   ceiling of this family and the source of the Gap.
7. BUSI validation is only 64 images, so its chosen hyper-parameters and
   intervals are wide.
8. Low-quality or empty predictions may not be dropped (every image enters the
   macro mean); `q_cycle` is not target Dice; the Oracle is not deployable and
   candidate counts must be reported separately; the split does not exclude
   same-patient / near-duplicate images (no reliable video-chain source ID), and
   "pseudo-video" means an algorithmically constructed path; `b0` has no
   intermediate bridge but still needs the anchor prompt, so it is not prompt-free
   SAM3; 7 → 14 doubles the propagation task count and the recorded `seconds` are
   a workload reference, not controlled end-to-end timing; the paired intervals
   are exploratory (10 000 image-pair bootstrap, seed 2026, no multiplicity
   correction) and cross-fold OOF is not an independent training repeat.
9. The unified main table is **not yet valid** (`PAPER_OUTLINE.md` §5 table 1):
   Kvasir and ISIC are at k = 1 while BUSI is at k = 2, so the "one method"
   comparison is unfinished. Kvasir already has the full k = 1…8 pool, so
   **P5 (unified-router refit, 0 GPU)** is the zero-cost next step, and **P4 is
   the highest submission risk**.

Open items: **P3** ISIC2018 top-5 full depth (~12 h), **P4** random-anchor
control with real propagation (~4 h), **P5** Kvasir unified-router refit (0 GPU),
**P7** a never-diagnosed blind dataset.

---

# Cross-version summary

| Version | anchors | route families | teacher / propagation | candidates | student pipeline | headline test Dice | Oracle |
|---|---|---|---|---|---:|---|---:|---:|
| V1 | fixed 8 | TP + PC | DINOv3 kNN; `ft_1pct` (or base) | 8 (b3–b6) / 14 (b0–b6) | S2/S3 → committee → X3 → B7 | 0.897146 (256) / **0.899369** (512) | 0.918109 / 0.919805 |
| V2 | fixed 8 | TP + PC | SAM3-trunk kNN @256; base → LoRA e33 | 14 | S2/S3 → X3 → B7 (→ X4) | **0.895432** (base) / **0.906380** (e33) | 0.921471 / 0.920206 |
| V3 | fixed 8 | single TP | base `sam3.pt` | 7 | S2/S3 → committee → X3 → B7 | **0.892639** | 0.907426 |
| V4 | fixed 8 | single TP | base + Round-2 LoRA | 7 | filter-first 580 → single student → 620 rescreen; B7 | B7 0.890846 → **0.894641**; student 0.858330 / **0.869478**; Round-2 LoRA 0.901755 / 0.903294 / **0.912932** | 0.907426 |
| V5 | automatic 8 | single TP (per-bridge) | base `sam3.pt` | 7 | Router only | **0.871374** | 0.917615 |
| V6 | automatic 8 | single TP (raw top-2) | base `sam3.pt` | 14 | Router only | **0.891226** | 0.932897 |
| V7 | automatic 8 | single TP (calibrated top-2) | base `sam3.pt` | 14 | Router only | 0.862761 | **0.937542** |

Read the ladder as three pairwise stories, not one monotone curve:

- **V1 → V2.** Same pipeline, DINOv3 → SAM3 features. SAM3-encoder routes lift B7
  by about +0.009 at the same checkpoint (0.886130 → 0.895432); the second LoRA
  round then lifts it to 0.906380.
- **V2 → V3 → V4.** Dropping patch correspondence, then reordering the filter,
  then rebuilding the student. B7 moves 0.895432 → 0.892639 → 0.890846 → 0.894641
  while the single-image student and the router both get stronger and the
  Oracle-to-realised gap narrows; the Round-2 SAM3 LoRA is a separate,
  single-image line that reaches 0.912932 on one seed but reverses ordering on
  another.
- **V5 → V6 → V7.** Same pipeline, different anchor ranking. Breadth (V6) lifts
  both the ceiling (0.912017 → 0.932897) and the realised score (0.870848 →
  0.891226); calibration (V7) lifts the ceiling further (0.937542) but on Kvasir
  **reduces** the realised score (0.862761) — the bottleneck there is selection,
  not candidates.

## Protocol shared by V1–V4

```text
Kvasir-SEG, train=800 / validation=100 / test=100
8 fixed human anchors = round(0.01 x 800) inside the 792 unlabelled train images
teacher  : SAM3 (base, fine-tuned, or LoRA-adapted — per version)
routes   : TP and/or PC families, bridge depth b0-b6
router   : independent, fitted on validation only, image-grouped folds
students : SC-SAM SamUnet single-image U-Nets (no SAM3 encoder)
metric   : per-image Dice/IoU, macro mean over all 100 test targets
```

## Protocol shared by V5–V7

```text
Kvasir-SEG, train=800 / validation=100 / test=100
anchors  : 8 selected by coverage-greedy facility location on train RGB only
           (1008 SAM3 features; global descriptor + 64 IDF-weighted local words)
teacher  : frozen SAM3-base
routes   : 256 features, patch grid 18, patch_mean kNN, beam 32, Target Pooling,
           b0-b6, anchor GT tight box, no text, canvas 256
candidates: top-k anchors per target x b0-b6  -> 7 (V5) or 14 (V6/V7)
router   : legacy-28-feature Ridge(alpha=1), image-grouped 5-fold on validation
metric   : per-image Dice/IoU, macro mean over all 100 test targets
```

## Where each version lives

| Version | Primary reports | Primary artifacts |
|---|---|---|
| V1 | `reports/512_C0_reproduction.md`, `reports/C0_256_reproduction.md`, `reports/C0_256_base_reproduction.md`, `reports/C0_256_full_experiment_history.md`, `reports/Kvasir_TP_only_vs_dual_route_20260908.md`, `docs/knn_experiment_vitb256.md`, `docs/method_cn.md`, `docs/method_en.md`, `docs/phase1_kvasir1pct_student.md`, `docs/routeco_sam3_v1.md` | `results/rerun_c0/`, `results/rerun_c0_c0/`, `results/rerun_c0_256/`, `results/rerun_c0_256_base/`, `results/kvasir_tp_guided_pc_joint_20260907/`, `results/kvasir_tp_pc_*/`, `results/kvasir_1pct_anchors/*` |
| V2 | `reports/C0_256_sam3knn_s256_b0_b6_run.md`, `reports/C0_256_sam3knn_s256_b0_b6_test_details.md`, `reports/C0_256_Round1_Round2A_complete_reproduction.md`, `reports/C0_256_round2a_fixed_knn_e33_x4.md`, `reports/C0_256_round2b_*.md`, `reports/C0_256_round2c_*.md`, `reports/C0_256_round3_*.md`, `docs/knn_experiment_sam3enc_*.md` | `results/rerun_c0_256_sam3knn_s256_base/`, `results/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_*/` |
| V3 | `reports/Kvasir_TP_student_mainline_20260907.md`, `reports/Kvasir_single_TP_mainline_20260909.md`, `reports/Kvasir_TP_only_students_pipeline_CN_20260909.md`, `reports/Kvasir_joint_diagnosis_and_next_steps_20260907.md` | `results/kvasir_tp_student_mainline_20260907/`, `results/kvasir_tp_b0_b6_router_baseline_20260907/` |
| V4 | `reports/Kvasir_TP_router_pseudo_filter_diagnosis_20260909.md`, `reports/Kvasir_TP_filterfirst_students_20260909.md`, `reports/Kvasir_TP_filterfirst_X3_diagnosis_20260909.md`, `reports/Kvasir_TP580_S2_S3_test_checkpoints_20260909.md`, `reports/Kvasir_X3_pool_v2_plan_20260909.md`, `mainline/single_student_*.md`, `mainline/tp_student_rescreen_*.md`, `mainline/tp_tracker_*.md`, `mainline/round2_*.md`, `mainline/A0_*.md`, `mainline/A1_A2_*.md`, `mainline/new_b7_results.md` | `results/kvasir_tp_pseudo_filter_20260909/`, `results/kvasir_tp_filterfirst_students_20260909/`, `results/kvasir_x3_pool_v2_20260909/`, `mainline/experiments/{single_student_hard_soft_20260909,tp_student_rescreen_20260910,tp448_*,round2_*,stage23_old_new_comparison_20260912}/` |
| V5 | `mainline/experiments/automatic_anchor_selection_pilot_20260911/report.md`, `auto8_tp_validation_20260911/report.md`, `automatic_anchor_tp_router_20260913/report.md`, `automatic_anchor_tp_test_20260913/README.md`, `mainline/reproduction_guides/automatic_selection_20260914/三数据集自动选图与TP路线_中文复现指南.md` | `mainline/experiments/automatic_anchor_*/`, `mainline/experiments/auto8_tp_validation_20260911/`, `mainline/reproduction_guides/automatic_selection_20260914/verify_*/` |
| V6 | `mainline/experiments/cross_dataset_calibration_factorial_20260914/kvasir/report.md` (`raw_top2` arm) | same factorial directory |
| V7 | `mainline/experiments/cross_dataset_calibration_factorial_20260914/{README.md,kvasir/report.md,isic2018/report.md}`, `mainline/reproduction_guides/automatic_selection_20260914/{PAPER_OUTLINE.md,E0_diagnostics_20260914/E0_diagnostics.md,P1_pilot_20260914/P1_REPORT.md,P2_fullbank_20260914/P2_REPORT.md}` | same factorial directory + `mainline/experiments/{busi_*,tn3k_*}/`, `results/busi_1pct_protocol/` |

## Not found / do not conflate

- The literal labels "V1"…"V7" do not exist in the server files.
- V1's DINOv3 route family does not have a reported X3 single-image test IoU, and
  the C0-256-base / `ft_1pct`-256 routes have no X3 single-image test Dice.
- V3's X3 final checkpoint was never tested.
- No official results exist for the X3-pool-v2 pool (the pipeline failed at
  `train_S3`).
- Round-1 LoRA (fixed-route forward-only Dice) and Round-2 fine-tunes (direct
  single-image Dice) were never compared under an identical evaluation protocol.
- TN3K has no published bias ratio, concentration or tie statistics.
