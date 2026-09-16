#!/usr/bin/env python3
"""Collect the SynFoC-on-BUSI result into a markdown report.

Reads ``work/busi_1pct_protocol/synfoc/{summary.json,log.txt}`` and writes
``work/busi_1pct_protocol/result.md``.

The reference block is the **current (score-calibrated) BUSI 1pct mainline**,
read from the frozen
``mainline/experiments/busi_calibration_factorial_20260914/results.json``
so the comparison numbers cannot drift.  The pre-calibration (raw) numbers are
kept only as a clearly-labelled historical column.

Safe to run while training is still in progress (then it reports the latest
validation curve instead of the final test row).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
DEFAULT_RUN = ROOT / "work/busi_1pct_protocol/synfoc"
DEFAULT_OUT = ROOT / "work/busi_1pct_protocol/result.md"
CALIB = (
    ROOT
    / "mainline/experiments/busi_calibration_factorial_20260914/results.json"
)

GROUP_LABELS = [
    ("raw_top1", "raw_top1 (pre-calibration)"),
    ("centered_top1", "centered_top1 (calibrated, no Router)"),
    ("raw_top2", "raw_top2"),
    ("centered_top2", "centered_top2 (calibrated, Router) — current best"),
    ("original_per_bridge", "original_per_bridge (historical control)"),
]

# Fallback if the frozen results.json is unavailable.
FALLBACK = {
    "validation": {
        "raw_top1": {"oof_dice": 0.6162684680007459, "oracle_dice": 0.6752466169520803},
        "centered_top1": {"oof_dice": 0.728772240305812, "oracle_dice": 0.7539997979581942},
        "raw_top2": {"oof_dice": 0.6656829014481289, "oracle_dice": 0.7831275433251789},
        "centered_top2": {"oof_dice": 0.7389582880918044, "oracle_dice": 0.8234767849940892},
        "original_per_bridge": {"oof_dice": 0.6176338160, "oracle_dice": 0.6752466169520803},
    },
    "test": {
        "raw_top1": {"dice": 0.5659903244694564, "iou": 0.4793393898541613, "oracle_dice": 0.617141189978921},
        "centered_top1": {"dice": 0.7199918056526315, "iou": 0.6359152029116032, "oracle_dice": 0.7754607907129016},
        "raw_top2": {"dice": 0.6560838707747543, "iou": 0.5644683762690047, "oracle_dice": 0.7763469123368754},
        "centered_top2": {"dice": 0.7521975757993846, "iou": 0.6681545141238049, "oracle_dice": 0.8236492008403925},
        "original_per_bridge": {"dice": 0.5668083894947364, "iou": 0.4800257212966827, "oracle_dice": 0.617141189978921},
    },
}

EPOCH_RE = re.compile(r"\]\s*(?:domain\d+ )?epoch (\d+) : loss")
DICE_RE = re.compile(r"val_lesion_dice: ([0-9.]+),")


def parse_curve(text: str) -> dict[str, dict[int, float]]:
    """Return {model: {epoch: val_dice}} from a SynFoC log.

    Each evaluation logs the same per-epoch value twice (once with and once
    without the ``domainN`` prefix), so values are de-duplicated by
    (epoch, model).
    """
    curve: dict[str, dict[int, float]] = {"UNet": {}, "SAM": {}}
    model = "UNet"
    epoch = -1
    pending: float | None = None
    for line in text.splitlines():
        if "final one-shot test" in line:
            # Everything after this point is the test split, not validation.
            break
        if "test unet model" in line:
            model, pending = "UNet", None
            continue
        if "test sam model" in line:
            model, pending = "SAM", None
            continue
        match = EPOCH_RE.search(line)
        if match:
            epoch, pending = int(match.group(1)), None
            continue
        match = DICE_RE.search(line)
        if match and epoch >= 0:
            value = float(match.group(1))
            if pending is None:
                pending = value
                curve[model][epoch] = value
            continue
    return curve


def load_calibration() -> tuple[dict, str, list[float]]:
    try:
        data = json.loads(CALIB.read_text())
        val_groups = data["validation"]["groups"]
        names = [label[0] for label in GROUP_LABELS]
        groups = {
            "validation": {
                name: {
                    "oof_dice": val_groups[name]["oof_dice"],
                    "oracle_dice": val_groups[name]["oracle_dice"],
                }
                for name in names
            },
            "test": {
                name: {
                    "dice": data["test"][name]["dice"],
                    "iou": data["test"][name]["iou"],
                    "oracle_dice": data["test"][name]["oracle_dice"],
                }
                for name in names
            },
        }
        per_bridge = data["test"]["centered_top1"]["fixed_rank1_b0_b6"]
        return groups, str(CALIB), per_bridge
    except Exception as exc:  # pragma: no cover - fallback path
        return FALLBACK, f"embedded fallback ({exc})", []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    groups, source, per_bridge = load_calibration()
    lines: list[str] = ["# SynFoC on BUSI (fixed split, 1% labels) — result", ""]

    log_path = args.run / "log.txt"
    summary_path = args.run / "summary.json"

    if log_path.is_file():
        curve = parse_curve(log_path.read_text(errors="replace"))
        epochs = sorted(set(curve["UNet"]) | set(curve["SAM"]))
        if epochs:
            lines += ["## SynFoC validation curve (per epoch, 64 images)", ""]
            lines += ["| epoch | iter | UNet Dice | SAM+LoRA Dice |", "|---:|---:|---:|---:|"]
            for epoch in epochs:
                unet = curve["UNet"].get(epoch)
                sam = curve["SAM"].get(epoch)
                lines.append(
                    f"| {epoch} | {epoch * 500} | "
                    f"{'-' if unet is None else f'{unet:.6f}'} | "
                    f"{'-' if sam is None else f'{sam:.6f}'} |"
                )
            best_sam = max(curve["SAM"].items(), key=lambda kv: kv[1]) if curve["SAM"] else None
            if best_sam:
                lines += [
                    "",
                    f"Best SAM+LoRA so far: **{best_sam[1]:.6f}** (epoch {best_sam[0]}).",
                    "",
                ]

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        test = summary.get("test", {})
        best = summary.get("best_validation", {})
        lines += [
            "## SynFoC final (best-validation checkpoints, one-shot test)",
            "",
            "| Split | Model | Dice | Iteration |",
            "|---|---|---:|---:|",
            f"| validation (64) | MedSAM+LoRA | {best.get('sam_dice'):.6f} | {best.get('sam_iteration')} |",
            f"| validation (64) | UNet | {best.get('unet_dice'):.6f} | {best.get('unet_iteration')} |",
            f"| test (66) | MedSAM+LoRA | {test.get('sam_dice'):.6f} | – |",
            f"| test (66) | UNet | {test.get('unet_dice'):.6f} | – |",
            "",
        ]
    else:
        lines += ["SynFoC training still running: no `summary.json` yet.", ""]

    if summary_path.is_file():
        test = summary.get("test", {})
        sam = test.get("sam_dice", 0.0)
        unet = test.get("unet_dice", 0.0)
        ref = sam  # the validation-selected branch is the method headline
        lines += [
            "### Head-to-head on the test split (66 images, one mask per image)",
            "",
            f"Deltas are quoted against the **MedSAM+LoRA branch ({sam:.6f})**, which",
            "is the branch SynFoC's own validation selects. The UNet branch is listed",
            "for completeness; picking it because its test number is higher would be",
            "test-driven selection and is not done here.",
            "",
            "| Method | test Dice | Δ vs SynFoC (SAM branch) |",
            "|---|---:|---:|",
            f"| raw_top1 (pre-calibration) | {groups['test']['raw_top1']['dice']:.6f} | "
            f"{groups['test']['raw_top1']['dice'] - ref:+.6f} |",
            f"| original_per_bridge (pre-calibration) | {groups['test']['original_per_bridge']['dice']:.6f} | "
            f"{groups['test']['original_per_bridge']['dice'] - ref:+.6f} |",
            f"| raw_top2 | {groups['test']['raw_top2']['dice']:.6f} | "
            f"{groups['test']['raw_top2']['dice'] - ref:+.6f} |",
            f"| **SynFoC MedSAM+LoRA (validation-selected branch)** | **{sam:.6f}** | – |",
            f"| SynFoC UNet (other branch, not selected) | {unet:.6f} | {unet - ref:+.6f} |",
            f"| centered_top1 (calibrated, no Router) | {groups['test']['centered_top1']['dice']:.6f} | "
            f"{groups['test']['centered_top1']['dice'] - ref:+.6f} |",
            f"| centered_top2 (calibrated, Router) | {groups['test']['centered_top2']['dice']:.6f} | "
            f"{groups['test']['centered_top2']['dice'] - ref:+.6f} |",
            f"| centered_top1 oracle | {groups['test']['centered_top1']['oracle_dice']:.6f} | "
            f"{groups['test']['centered_top1']['oracle_dice'] - ref:+.6f} |",
            f"| centered_top2 oracle | {groups['test']['centered_top2']['oracle_dice']:.6f} | "
            f"{groups['test']['centered_top2']['oracle_dice'] - ref:+.6f} |",
            "",
            "SynFoC uses the same 5 labeled images and the same 517/64/66 split, but",
            "trains a segmentation network instead of retrieving pseudo-video paths,",
            "so the comparison is single-mask against single-mask. Router variants",
            "also emit one mask per image; oracle rows are upper bounds, not methods.",
            "Single seed, single run: no seed repeats, so differences below a few",
            "Dice points should not be over-interpreted.",
            "",
        ]

    lines += [
        "## BUSI 1pct mainline reference — current calibrated version",
        "",
        f"Source: `{Path(source).relative_to(ROOT)}` (frozen).",
        "",
        "### Calibrated rank-1 anchor, no Router (test, 66 images)",
        "",
        "Each column fixes one bridge length and uses the score-calibrated rank-1",
        "reference image; no Router is involved.",
        "",
    ]
    if per_bridge:
        lines += ["| bridge | calibrated Dice |", "|---|---:|"]
        for idx, value in enumerate(per_bridge):
            lines.append(f"| b{idx} | {value:.6f} |")
        lines.append("")
        lines.append(
            f"Oracle over these 7 candidates: "
            f"**{groups['test']['centered_top1']['oracle_dice']:.6f}**."
        )
        lines.append("")
    lines += [
        "### Group comparison",
        "",
        "| Config | cand/target | val OOF Dice | test Dice | test IoU | test Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    candidates = {
        "raw_top1": 7,
        "centered_top1": 7,
        "raw_top2": 14,
        "centered_top2": 14,
        "original_per_bridge": 7,
    }
    for name, label in GROUP_LABELS:
        val = groups["validation"][name]
        test = groups["test"][name]
        lines.append(
            f"| {label} | {candidates[name]} | {val['oof_dice']:.6f} | "
            f"{test['dice']:.6f} | {test['iou']:.6f} | {test['oracle_dice']:.6f} |"
        )
    lines += [
        "",
        "Notes: the Router variants use the same legacy28 Ridge(alpha=1) and the",
        "frozen image-level 5-fold validation split; `centered_*` subtracts the",
        "per-reference train mean from the target-pooling score. The",
        "pre-calibration numbers are the raw-TP ranking and are kept only for",
        "history. SynFoC is a fine-tuned segmentation network producing one mask",
        "per image, so it is compared against these single-prediction / Router",
        "pipelines, not against the oracle.",
        "",
    ]

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
