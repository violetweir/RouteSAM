# Pseudo-Video SAM3 — X3 + B7

Category-free **pseudo-video** segmentation with frozen SAM3, single-image
student auditors, and geometric route selection.

[中文说明](README_CN.md) · [Latest cross-dataset study](docs/cross_dataset_1pct.md) · [Stage 0 + Stage 1 protocol & results](docs/stage0_stage1.md) · [V1–V7 per-version walkthrough](docs/kvasir_versions.md) · [Repository layout](docs/REPOSITORY_LAYOUT.md) · [Full reproduction guide](docs/reproduction_guide.md) · [Method (EN)](docs/method_en.md) · [方法（中文）](docs/method_cn.md)

---

## What this is

**The mainline is the cross-dataset 1% anchor study on Kvasir-SEG, ISIC2018 and
BUSI and TN3K**: keep SAM3 **frozen**, construct
**pseudo-video propagation routes** from a 1%-budget set of annotated anchors,
and put all the modelling effort into **selecting** among the routes.

| Line | Datasets | Budget | Role |
|---|---|---|---|
| **1% anchor cross-dataset study** | Kvasir-SEG, ISIC2018, BUSI, TN3K | 1% labelled anchors, frozen SAM3 | **mainline** |
| ↳ Kvasir **V1–V4** | Kvasir-SEG only | 8 fixed anchors, router + S2/S3 students | route-family and student-pipeline evolution |
| ↳ Kvasir **V5–V7** | Kvasir-SEG | automatic coverage anchors, single TP + Router | V7 (calibrated top-2) is the arm extended cross-dataset |
| Pseudo-video `S27 X3 + B7` | CVC-ClinicDB + Kvasir-SEG merged | 16 fixed anchors | **historical** — kept fully reproducible |

Everything is in this repository, together with the paper-facing result tables
and the curated experiment reports behind every number.

---

## Mainline: cross-dataset 1% anchor study

Frozen SAM3, one pipeline, same code for every dataset. Coverage anchor
selection → annotation-free score calibration → breadth × depth candidates →
ridge verification. Full write-up: **[docs/cross_dataset_1pct.md](docs/cross_dataset_1pct.md)**.

| Method | cand/target | Kvasir (100) | ISIC2018 (260) | BUSI (66) |
|---|---:|---:|---:|---:|
| raw score top-1 | 7 | 0.870848 | 0.868228 | 0.565990 |
| calibrated top-1 | 7 | 0.860331 | 0.871717 | **0.719992** |
| raw score top-2 | 14 | **0.891226** | 0.869276 | 0.656084 |
| calibrated top-2 | 14 | 0.862761 | **0.875864** | **0.752198** |
| historical per-bridge + legacy Router | 7 | 0.871374 | 0.866573 | 0.566808 |

**What it shows.** Anchor-conditioned TP scores are not comparable across
anchors. The spread of the per-anchor train mean is an **annotation-free**
diagnostic that predicts how much score calibration helps:

| Dataset | bias ratio | calibration gain (test) | verdict |
|---|---:|---:|---|
| BUSI | 5.42 | +0.154 (top-1), +0.185 (top-2) | large, significant |
| ISIC2018 | 1.10 | +0.004 / +0.007 (val: +0.016 / +0.012, significant) | small, not significant at test |
| Kvasir | 0.68 | −0.011 / −0.028 | no effect — **negative control** |

Breadth `k` and depth `m` are real levers and **substitute** for calibration: the
`k=1→2` ceiling gain is +0.069 at `b0` but only +0.015 at full depth on Kvasir.
A single labelled anchor already reaches a 0.9142 test ceiling on Kvasir; going
to the full 8 adds only +0.040. On BUSI, the calibrated top-2 router reaches
**0.752198** versus **0.653120** for a SynFoC student trained on the same 5 labels.

---

## Kvasir-SEG: the V1 → V7 ladder

> The current Stage 0 + Stage 1 definition (automatic anchors → TP routes →
> propagation → Router), with every hyper-parameter, result table and artifact
> path, is in **[docs/stage0_stage1.md](docs/stage0_stage1.md)**.
> A full per-version walkthrough of the older Kvasir lines is in
> **[docs/kvasir_versions.md](docs/kvasir_versions.md)**.

Kvasir-SEG is the largest arm of the mainline. It was rebuilt seven times:
**V1–V4** keep the original 8 fixed anchors and evolve the route families and the
student pipeline; **V5–V7** switch to automatic coverage anchors, and **V7**
(calibrated top-2) is the arm that was then extended to ISIC2018 / BUSI / TN3K.

The shared shape of V1–V4 is the Kvasir-only analogue of `S27 X3 + B7`: 8 fixed
anchors, frozen SAM3 routes `b0–b6` (7 candidates per image in the single-TP
versions), an independent router plus quality gates, then S2/S3 students, a
committee audit, X3, and B7 route selection.

```text
8 fixed GT anchors + 792 unlabelled train images
  -> SAM3 Target Pooling b0-b6, 7 candidates/image (5544 total)
     teacher = frozen base | full fine-tune | LoRA (see below)
  -> independent Router + quality gates (q_multi >= 0.90, q_return >= 0.95)
  -> 448 first-pass pseudo labels
  -> S2 (router-picked hard labels) / S3 (7-candidate consensus soft labels)
  -> committee audit of the remaining 344 -> Tier A 94 / B 113 / C 137
  -> X3 = 448 + A + B (655 pseudo labels) + 8 GT, retrained
  -> B7 selects one of the 7 SAM3 masks
```

| Method | Test Dice | Test IoU |
|---|---:|---:|
| Frozen SAM3 TP `b0–b6` + independent Router | 0.885433 | 0.828274 |
| X3 single-image student | 0.867161 | 0.786951 |
| **X3-best + TP-only B7** | **0.892639** | 0.834468 |
| B7 candidate Oracle (diagnostic only) | 0.907426 | – |

Follow-on refinements of the same line:

| Refinement | Result |
|---|---|
| Filter-first pool (screen admissible candidates before the router) | 448 → **580** pseudo labels |
| Single equal-weight student, 816 epochs (588 images, bs 12) | hard best/final test 0.848592 / 0.854011; soft **0.851243 / 0.859530** |
| Full 792-image rescreen with the frozen soft-label student | pool → **620**, 628 training images; best test **0.858330**, final test **0.869478** |
| S2/S3 frozen-checkpoint test (580 pool) | S2 valbest 0.851530 / final 0.851675; S3 valbest 0.841958 / final 0.843769 |
| Tracker endpoint, round-2 pool-A selection ablation | see [`mainline/EXPERIMENTS.md`](mainline/EXPERIMENTS.md) |

#### The Kvasir version ladder (V1 → V7)

Line 1 was rebuilt seven times. V1–V4 keep the original 8 fixed anchors and
evolve the route families and the student pipeline; V5–V7 replace the anchors with
automatic coverage selection.

| Version | Mainline | Main change |
|---|---|---|
| **V1** | original 8 anchors → DINOv3 kNN → SAM3 **TP + PC** → S2/S3 → committee → X3 → B7 | the earliest complete pipeline; SAM3 LoRA was added later |
| **V2** | original 8 anchors → **SAM3-base feature** kNN → TP + PC → S2/S3 → X3 → B7 | SAM3 features replace DINOv3; carries the Round-1 / Round-2A follow-ups |
| **V3** | original 8 anchors → **single TP** `b0–b6` + independent Router → 448 pseudo labels → S2/S3 → expansion, X3, B7 | drops PC, settles the single-TP baseline |
| **V4** | original 8 anchors → TP → **filter admissible candidates before the Router** → 580 → single student soft labels → full rescreen → new students, B7 | 580-pool S2/S3 control, single student hard/soft, 620 rescreen, and the later round-2 SAM3 fine-tune |
| **V5** | **automatic 8 anchors** → SAM3-base kNN → single TP `b0–b6` → Router | changes where the anchors come from |
| **V6** | automatic 8 anchors → **raw TP score top-2** anchors → `b0–b6` each → Router | 7 → 14 candidates per target |
| **V7** | automatic 8 anchors → **calibrated TP score top-2** anchors → `b0–b6` each → Router | annotation-free score calibration |

Test Dice (Kvasir, 100 targets) per version:

| Version | headline | Oracle | supporting numbers |
|---|---|---:|---|
| V1 | X3-best + B7 **0.897146** (canvas 256) / **0.899369** (original 512) | 0.918109 / 0.919805 | Router 0.891288 / 0.894648; X3 single 0.8633 (512); pools 470 / 491; committee 105-102 / 93-95-113; X3 pools 677 / 679 |
| V2 | X3-best + B7 **0.895432** (base) / **0.906380** (with the e33 LoRA teacher) | 0.921471 / 0.920206 | Router 0.874083 (TP+PC union) or 0.885433 (TP-only); X3 single 0.853920; 410 first-pass labels, committee A 94 / B 128 / C 160, X3 pool 632 |
| V3 | X3-best + B7 **0.892639** | 0.907426 | Router 0.885433; X3 single 0.867161; 448 labels, committee A 94 / B 113, X3 pool 655 |
| V4 | 620 rescreen final **0.869478** (best 0.858330); round-2 SAM3 LoRA 50 ep best **0.901950** | – | 580-pool S2/S3 S2 0.851530/0.851675, S3 0.841958/0.843769; single student hard 0.848592/0.854011, soft 0.851243/0.859530 |
| V5 | raw top-1 **0.870848** | 0.912017 | independent Router 0.871374; val OOF 0.848638 |
| V6 | raw top-2 **0.891226** | 0.932897 | val OOF 0.846724 |
| V7 | calibrated top-2 **0.862761** | **0.937542** | calibrated top-1 0.860331 (val 0.852949); val OOF 0.853773 |

**How to read the ladder**

- V1 → V2 is a route-feature change: swapping DINOv3 for SAM3's own encoder moved
  the best fixed route from 0.854627 to 0.8874 (`target pooling patch_mean` b5 at
  @1008) and the same-checkpoint B7 from 0.886130 to 0.895432 (+0.009302); a
  second LoRA round then took it to 0.906380.
- V2 → V3 is a candidate-set change: dropping patch correspondence and keeping
  one TP family per target gave a stronger router (0.874083 → 0.885433) and a
  stronger X3 (0.853920 → 0.867161) but a marginally lower B7
  (0.895432 → 0.892639), while the candidate Oracle fell from 0.921471 to
  0.907426 — the union pool did contain masks the TP pool lacked.
- V3 → V4 is a pseudo-label-pool change: screening admissible candidates before
  the router grew the first pass 448 → 580, and the follow-on work (single
  equal-weight student, full 792-image rescreen to 620) traded the multi-student
  committee for one student. Round 2 then re-fine-tuned SAM3 with LoRA on the
  student-audited pool.
- V5 → V6 → V7 changes the anchor source and the anchor ranking. Automatic
  anchors with a single raw top-1 reach 0.870848; keeping the raw top-2 raises
  both the ceiling (0.912017 → 0.932897) and the realized score (0.891226);
  calibration lowers the realized score to 0.862761 while raising the ceiling
  again (0.937542) — on Kvasir the bottleneck is **selection**, not candidates.

#### Round-2 SAM3 LoRA fine-tunes (inside V4)

Two rounds of full-module LoRA (rank 16, alpha 32, dropout 0.1) were run on the
B7-rescreened student-audited pool. `20 epochs` was the original LoRA teacher
adaptation; the round-2 fine-tunes use 10 epochs, plus one 50-epoch control.

| pool (pseudo + GT) | epochs | best epoch | Val Dice | Test Dice (`colon polyp`) | Test Dice (empty text) |
|---|---:|---:|---:|---:|---:|
| A0 · 428 + 8 | **50** (no clip) | 19 | 0.893717 | 0.901950 | 0.890174 |
| A0 · 428 + 8 | 50 (final) | 50 | 0.873586 | 0.894245 | 0.884782 |
| A0 · 428 + 8 | 10 | 6 | 0.880909 | 0.901755 | 0.885264 |
| A1 · 596 + 8 | 10 | 4 | 0.887257 | 0.903294 | 0.870346 |
| A2 · 503 + 8 | 10 | 1 | 0.888680 | **0.912932** | 0.845776 |

Two seeds at 10 epochs (2026 / 2027): A0 0.901755 / 0.892288, A1 0.903294 /
0.886602, A2 0.912932 / 0.884289 — the ordering **reverses** between seeds, so no
pool is stably best and the round-2 fine-tune does not beat the V1–V3 B7 results.
The 50-epoch control peaks at epoch 19 and falls back by epoch 50; it also
disables gradient clipping and changes the cosine horizon, so it is not a clean
epoch-length ablation.

#### LoRA teacher checkpoints used across V2–V4 (20 epochs)

The route teacher itself was adapted with the same full-module LoRA recipe for
20 epochs (`configs/kvasir_1pct_lora.yaml`,
`configs/kvasir_1pct_plus_pseudo491_lora.yaml`):

| checkpoint | LoRA training data |
|---|---|
| `lora_1pct_e20` | the 8 GT anchors (1%) |
| `lora_p491_e20` | 8 GT + **491 pseudo labels** (first router expansion pool) |

Forward-only test Dice per bridge (100 targets, canvas 256):

| checkpoint | route mode | direct | b1 | b2 | b3 | b4 | b5 | b6 | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora_1pct_e20` | target pooling | 0.771193 | 0.797666 | 0.861002 | 0.859552 | **0.886556** | 0.874425 | 0.880978 | 0.847339 |
| `lora_1pct_e20` | patch correspondence | 0.765991 | 0.800123 | 0.844173 | 0.852210 | **0.880581** | 0.869514 | 0.864183 | 0.839539 |
| `lora_p491_e20` | target pooling | 0.790054 | 0.824923 | 0.870163 | 0.877370 | **0.887164** | 0.890885 | 0.878756 | 0.859902 |
| `lora_p491_e20` | patch correspondence | 0.799188 | 0.830049 | 0.865747 | 0.884548 | 0.883273 | 0.889516 | **0.911171** | 0.866213 |

- Adding the 491 pseudo labels improves **every bridge** in both route modes
  (+0.009 to +0.047); `patch correspondence b6` = **0.911171** is the best single
  fixed route in the Kvasir line.
- Both LoRA checkpoints beat the frozen base and the full fine-tune (`ft_1pct`) at
  every bridge. With SAM3-encoder kNN features at @1008 the `lora_p491_e20` route
  reaches **0.9154** (`target pooling cond`, b1/b4); @256 it reaches **0.9082**
  (`patch correspondence cond`, b4).
- The downstream line was re-run on the `lora_p491_e20` routes: fixed `b6`
  **0.8978**, B7 geometric (3-student mean) **0.9014**, calibrated linear
  **0.9035**, candidate Oracle 0.9357; pairwise-ranker validation OOF
  **0.880443**; frozen transfer to ClinicDB test **0.857032**.
- Caveat: the S2/S3/X3 auditors used in those B7 numbers were trained from the
  **base-SAM3** pseudo labels, not re-trained on the LoRA teacher.

#### What changed from V4 to V7 — automatic anchors and calibratable scores

| V1–V4 | V5–V7 |
|---|---|
| 8 anchors fixed by hand | **coverage-greedy automatic selection** on train RGB only (`K = round(0.01·N_train)`) |
| methods differ by route family and student recipe | same single-TP `b0–b6` router pipeline, methods differ by **anchor source and anchor ranking** |
| anchor scores compared directly (V5, V6) | V7 adds **annotation-free calibration** `centered(A,T) = TP(A,T) − μ_A` |
| one anchor per target | V6/V7 keep **top-`k` anchors × `b0–b6`** → 7 → 14 candidates per target |
| manual `q_multi`/`q_return` thresholds | explicit `Realized = Oracle − Gap` decomposition; the same arms then run on **ISIC2018, BUSI and TN3K** |

On Kvasir this is the **negative control** for calibration (bias ratio 0.68): V7's
calibrated top-2 pool has the highest Oracle (0.937542) but the router only
realises 0.862761, so the bottleneck moved from candidates to selection. What it
does establish on Kvasir is the breadth lever (full-depth ceiling vs `k`) and the
budget curve — one labelled image already reaches a 0.9142 test ceiling.

Full inventory of everything run on Kvasir-SEG: **[docs/kvasir_program.md](docs/kvasir_program.md)**.

---

## Historical: S27 X3 + B7

> This is the project's original line, **not** the current mainline. It uses the
> merged CVC-ClinicDB + Kvasir-SEG protocol (16 anchors) and is kept because it
> is fully reproducible end to end. Full write-up:
> [`docs/s27_x3_b7_line.md`](docs/s27_x3_b7_line.md).

Starting from 16 fixed CVC + Kvasir train anchors, it builds frozen SAM3
pseudo-video routes, distills the resulting pseudo labels into single-image
**student auditors**, expands the pseudo-label pool with those auditors, and uses
the student to **select among frozen SAM3 routes** (B7). No validation or test
masks are used to build routes, select pseudo labels, or train the student.

```text
16 fixed train GT anchors
  -> T21 frozen SAM3 pseudo-video routes (direct / one_bridge / two_bridges)
  -> 568 high-confidence pseudo labels
  -> T24 committee students (S2 val-best, S2 final, S3 final)
  -> audit remaining train images into Tier A / B / C
  -> S27 X3 student = original 568 + Tier A + Tier B (876 pseudo labels)
  -> B7 student-assisted route selection on frozen SAM3 routes
```

Protocol: CVC-ClinicDB + Kvasir-SEG merged splits, **16 train anchors only**.
Test masks are used for final reporting only.

| Method | Test Dice | CVC Dice | Kvasir Dice |
|---|---:|---:|---:|
| Frozen SAM3 multi-route baseline | 0.8715 | – | – |
| S27 X3 single-image student | 0.866738 | 0.886304 | 0.854802 |
| S27 X3 + validation-selected linear selector | 0.885637 | 0.905077 | 0.873778 |
| **S27 X3 + B7 geometric selector** | **0.895835** | 0.904713 | 0.890420 |
| Oracle over three frozen SAM3 routes | 0.907835 | – | – |

B7 route score:

```text
score = (max(q_return, 1e-6) * max(q_multi, 1e-6)^2 * max(q_model, 1e-6)^2) ^ 0.2
```

- `q_return` — anchor round-trip consistency of the propagated mask
- `q_multi` — agreement between the three route families
- `q_model` — student confidence on the candidate mask

## Kvasir route engine: propagation-quality router

The routing engine behind Kvasir line 1 (frozen SAM3, `train=800 / validation=100 /
test=100`, **8 fixed anchors**) replaces candidate-relative weighting with an
absolute **propagation-quality router**:

| selector | candidates | Test Dice | Oracle | Oracle gap |
|---|---|---:|---:|---:|
| fixed best route (target pooling `b6`) | – | 0.854627 | – | – |
| v2 absolute ridge router | target pooling `b0–b7` | 0.854437 | 0.898969 | 0.044531 |
| propagation-quality router | target pooling `b0–b7` | 0.872938 | 0.898969 | 0.026030 |
| propagation-quality router | target + patch `b0–b6` | 0.873626 | 0.903846 | 0.030220 |
| **propagation-quality router** | **target + patch `b3–b6`** | **0.877299** | 0.896918 | 0.019619 |

Fine-tuning SAM3 on the same 8 anchors pushes the same router to **0.894648**
(`ft_1pct`, `b3–b6` target + patch), oracle 0.919805. The full Kvasir-SEG
inventory is in [`docs/kvasir_program.md`](docs/kvasir_program.md).

## Repository layout

```text
configs/        machine-editable run configuration (example + study configs)
docs/           method, protocol and experiment documents (docs/v2/ = older series)
docs/cross_dataset_1pct.md   THE MAINLINE: Kvasir / ISIC2018 / BUSI / TN3K 1% study
docs/kvasir_program.md       full Kvasir-SEG experiment inventory
docs/s27_x3_b7_line.md       historical S27 X3 + B7 line
envs/           example conda environment manifests (sam3, student)
protocols/      fixed path-free split + 16-support-anchor protocol
scripts/        reproduction stages and experiment drivers (mainline: run_pipeline.py)
src/pvseg/      small shared utilities (io, metrics, protocol)
tests/          protocol / leakage / checkpoint invariants
third_party/    vendored SC-SAM and SynFoC-T20 student baselines
medsam3/        vendored SAM3 + LoRA training toolkit
paper/          paper-facing experiment packages (tables, figures, source maps)
reports/        per-experiment reproduction reports (Markdown)
mainline/       cross-dataset 1% studies: coverage anchors, calibration, TP routes
results/        curated per-run summary JSON/CSV/Markdown extracted from `work/`
```

See [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md) for the mapping
between this clean layout and the original server working tree.

## Quick start

```bash
cp configs/reproduction.example.toml configs/reproduction.toml
# edit paths in configs/reproduction.toml

python scripts/run_method_ladder.py --config configs/reproduction.toml --dry-run
python scripts/run_pipeline.py     --config configs/reproduction.toml --dry-run
python scripts/run_pipeline.py     --config configs/reproduction.toml
```

Resume from any stage:

```bash
python scripts/run_pipeline.py \
  --config configs/reproduction.toml \
  --from-stage s27_x3_train \
  --to-stage t25_b7_test
```

`scripts/run_method_ladder.py` walks the whole experimental ladder
(B00 single-image → T19/T20 low-label students → T18/E1 pseudo-video →
T21 frozen routes → T24 committee → S27 X3 → B7). `scripts/run_pipeline.py` is
the clean final mainline.

### Data layout

```text
data/
  train/metadata.jsonl
  validation/metadata.jsonl
  test/metadata.jsonl
```

Each row:

```json
{
  "file_name": "/abs/path/to/image.png",
  "mask_file_name": "/abs/path/to/mask.png",
  "merged_id": "CVC-ClinicDB::156",
  "source_dataset": "CVC-ClinicDB"
}
```

Expected counts: `train=1290`, `validation=161`, `test=161`. The fixed support
IDs are in `protocols/reproduction_v1/support_ids.txt`; the full path-free split
is `protocols/reproduction_v1/splits.jsonl`.

## External dependencies

SAM3, the datasets, and the model checkpoints are **not** vendored here.
SC-SAM and SynFoC student baselines are vendored under `third_party/` and are
loaded through `SC_SAM_ROOT` / `synfoc_root` (`third_party/SC-SAM`,
`third_party/SynFoC-T20` by default).

SAM3 commands and student commands run in different environments; the original
server used:

```text
SAM3:    /home/violet/anaconda3/envs/sam3/bin/python
Student: /home/violet/anaconda3/envs/mkunet_mamba/bin/python
```

## What is / is not in this repository

**Included:** source code, stage scripts, configuration, protocols, method and
protocol documents, experiment reports, paper-facing tables, the seven
cross-dataset diagnostic figures, the eight Kvasir anchor images/masks, and
curated per-run summary metrics under `results/`.

**Excluded** (large or machine-specific): datasets, SAM3 / student checkpoints
(`*.pt`, `*.pth`), per-run mask dumps and feature caches (`*.png` outside the
few figure directories, `*.npz`, `*.npy`), training logs of several GB, and the
original 7 GB `.git` history. `results/` keeps only the small summary artifacts,
so the numbers in the reports stay auditable without shipping the raw data.

## Important protocol notes

- The **cross-dataset 1% study** (Kvasir-SEG / ISIC2018 / BUSI) is the mainline.
  `S27 X3 + B7` is the project's original line, kept fully reproducible but no
  longer the headline.
- In the cross-dataset study, "1%" is the **anchor labelling budget only** — the
  router is fitted on validation and early test versions were used for
  diagnosis, so those test numbers are not a blind test.
- Calibration is **not** a universal gain: it is significant on BUSI
  (bias ratio 5.42), not significant at test on ISIC2018 (1.10), and ineffective
  on Kvasir (0.68). Do not quote BUSI's gain as a general result.
- **TN3K is complete** (576 validation / 614 test, 23 anchors) and behaves as a
  third regime: calibrated top-1 helps, calibrated top-2 does not, the interaction
  is negative. Its low absolute scores come mostly from anchor/target **scale
  mismatch**, not from a weaker segmenter — see
  [`docs/cross_dataset_1pct.md`](docs/cross_dataset_1pct.md) §TN3K.
- Use `docs/s27_x3_b7_line.md` before quoting any S27/X3/B7 number; the
  `S27 X0/X1/X3` trainers and the S27-vs-T24 Dice conventions differ.
- `S27 X0/X1/X3` use the later unified S27 student trainer; it is not a
  bit-level reproduction of the older T24 supervised-loss implementation.
- The S27 trainer uses a per-sample foreground Dice convention for GT; T24 used
  a two-class batch Dice convention. The reported X3 + B7 result comes from the
  preserved S27 implementation.
- B7 is reported as a fixed sensitivity/mainline selector; the
  validation-selected linear selector is reported separately for clarity.
- **Never** use validation/test masks to change support anchors, pseudo labels,
  route topology, training data, or checkpoints.

## Citation

See [CITATION.cff](CITATION.cff). If you use this package, please cite the
associated paper or preprint.

## License

BSD-3-Clause — see [LICENSE](LICENSE). Vendored third-party components keep
their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
