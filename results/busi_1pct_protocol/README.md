# SynFoC on the fixed BUSI split (1% labels)

Goal: run the vendored SynFoC-T20 method on BUSI **without touching the existing
BUSI split**, so the result is directly comparable with the `new_project`
BUSI 1pct mainline.

## Dataset split (frozen, reused as-is)

Source: `/Data_8TB/lht/MK-UNet/BUSI/BUSI_split`
(`BUSI_split.py`, `random.seed(42)`, per-class 8:1:1 stratified copy).

| Split (SynFoC name) | Source dir | Images |
|---|---|---:|
| `train` | `Busi_split/train` | 517 |
| `validation` | `Busi_split/val` | 64 |
| `test` | `Busi_split/test` | 66 |

Class composition is unchanged: train 349 benign / 168 malignant,
val 43 / 21, test 45 / 21. Masks are binary 0/255.

No image is re-shuffled, copied or re-split. The adapter
`scripts/prepare_busi_synfoc_protocol.py` re-expresses the *same* membership in
the `train/validation/test` layout SynFoC expects, using symlinks, and records
`protocol.json` with per-file paths and sha256.

## Labeled budget

SynFoC is semi-supervised, so it needs a labeled subset. This run reuses the
**frozen 5-image BUSI 1pct support** from
`new_project/experiments/busi_auto5_tp_1pct_20260913/SELECTED_SUPPORT_FROZEN.json`
(seed 2026, train-only, GT-free selection):

```
benign (3), benign (125), benign (305), malignant (195), malignant (187)
```

These are the *same* 5 anchors used by the current calibrated version
(`busi_calibration_factorial_20260914/calibration_frozen.json:anchor_ids` is
identical), so the labeled budget matches both versions.

5 / 517 = 0.9671% labeled; the remaining 512 train images are unlabeled
(masks are never opened for them:
`unlabeled_masks_loaded=false`).

## Adapter layout

```
work/busi_1pct_protocol/
  data/{train,validation,test}/metadata.jsonl   # absolute image/mask paths
  data/{train,validation,test}/images|masks/    # symlinks to BUSI_split
  busi_train1pct_labeled_images.txt             # the 5 frozen labels
  protocol.json                                 # counts, hashes, mapping
  synfoc/                                       # full run (output)
  synfoc_smoke/                                 # smoke run (output)
```

## Commands

```bash
# 1) build the adapter (idempotent, symlink-only)
python3 scripts/prepare_busi_synfoc_protocol.py

# 2) smoke test (~2 epochs, exercises data + validation + final test path)
bash scripts/run_busi_synfoc_1pct.sh smoke

# 3) full run: 40000 iters, num_eval_iter 500 (80 epochs), seed 2026
bash scripts/run_busi_synfoc_1pct.sh full \
  > work/busi_1pct_protocol/full.log 2>&1 &
```

SynFoC rewrites `CUDA_VISIBLE_DEVICES` from `--gpu`, so the physical device is
passed in `--gpu` (default `0` in the runner).

## Training configuration

Mirrors the ISIC18/Kvasir SynFoC baselines in this repo:

| Item | Value |
|---|---|
| `--dataset` | `ClinicDB` (generic metadata-backed adapter) |
| `--dataset_label` | `BUSI` |
| model | MedSAM ViT-B LoRA (`rank 4`) + UNet, EMA teacher |
| checkpoint | `/Data_8TB/lht/models/medsam_vit_b.pth` |
| `label_bs` / `unlabel_bs` / `test_bs` | 4 / 4 / 8 |
| `img_size` | 256 |
| `base_lr` | 0.03, AdamW + warmup |
| `max_iterations` / `num_eval_iter` | 40000 / 500 |
| `seed` | 2026 |
| consistency threshold | 0.95 (default) |

Model selection uses the 64-image `validation` split only; the 66-image `test`
split is evaluated once at the end from the best-validation checkpoints and
written to `synfoc/summary.json`.

## Reference numbers from the BUSI mainline (same split, same 5 labels)

**Current (score-calibrated) version**, from the frozen
`new_project/experiments/busi_calibration_factorial_20260914/results.json`.
Calibration subtracts the per-reference train mean from the target-pooling
score, `centered(A,T) = TP(A,T) - mean_train(TP(A,x))`, before ranking the
reference images.

Calibrated rank-1 reference, one bridge fixed, **no Router** (test, 66 images):

| bridge | Dice | | bridge | Dice |
|---|---:|---|---:|---:|
| b0 | 0.705708 | | b4 | 0.662677 |
| b1 | 0.709401 | | b5 | 0.698703 |
| b2 | 0.679150 | | b6 | 0.718360 |
| b3 | 0.689764 | | oracle (7 cand) | **0.775461** |

Group comparison:

| Config | cand/target | val OOF Dice | test Dice | test IoU | test Oracle |
|---|---:|---:|---:|---:|---:|
| raw_top1 (pre-calibration) | 7 | 0.616268 | 0.565990 | 0.479339 | 0.617141 |
| centered_top1 (calibrated, no Router) | 7 | 0.728772 | 0.719992 | 0.635915 | 0.775461 |
| raw_top2 | 14 | 0.665683 | 0.656084 | 0.564468 | 0.776347 |
| **centered_top2 (calibrated, Router) — current best** | 14 | 0.738958 | **0.752198** | 0.668155 | 0.823649 |
| original_per_bridge (historical control) | 7 | 0.617634 | 0.566808 | 0.480026 | 0.617141 |

The pre-calibration single-anchor numbers (b6 0.539444, Router 0.566808,
oracle 0.617141, from `busi_auto5_tp_1pct_20260913`) are **superseded**: they
are the raw-TP ranking and are kept only as history. SynFoC is a fine-tuned
segmentation network that emits one mask per image, so the honest comparison is
against `centered_top1` (0.719992) and `centered_top2` (0.752198), not against
the oracle.

`scripts/collect_synfoc_busi_result.py` reads these reference numbers straight
from the frozen `results.json`, so they cannot drift.

## Monitoring

```bash
# progress (per-500-iter validation Dice)
grep -aE "val_lesion_best_dice|stu_val_lesion_best_dice" \
  work/busi_1pct_protocol/synfoc/log.txt | tail

# refresh the markdown report (works before, during and after training)
python3 scripts/collect_synfoc_busi_result.py
```

The run is detached (`setsid nohup`) and survives the agent session; logs go to
`work/busi_1pct_protocol/full.log` and `work/busi_1pct_protocol/synfoc/log.txt`.
Runtime is roughly 5–7 h for 40000 iterations on one RTX 3090 (measured ~2.1
it/s plus 80 validation passes).

## Status

- [x] Adapter built: 517 / 64 / 66 verified byte-equal membership to `Busi_split`.
- [x] Confirmed the calibrated protocol uses the *same* split and the *same* 5
      anchors (`calibration_frozen.json:anchor_ids` == frozen support), so this
      SynFoC protocol needs no change.
- [x] Smoke run (400 iters, `synfoc_smoke/`): pipeline OK, test on 66 images
      gave UNet Dice 0.3875 / SAM Dice 0.5307 at epoch 2.
- [x] Full run finished 2026-09-15 17:40 (40000 iters, 80 epochs, GPU 0).
      Test (66 images): **UNet 0.683123**, **MedSAM+LoRA 0.653120**.
      Best validation: UNet 0.647518 (iter 38000), SAM 0.654125 (iter 37500).
- [x] `result.md` written, including a head-to-head against the calibrated
      mainline.
