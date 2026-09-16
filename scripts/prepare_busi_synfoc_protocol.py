#!/usr/bin/env python3
"""Build a SynFoC/SC-SAM compatible adapter for the frozen BUSI split.

The BUSI protocol used by the `followup` experiments is the fixed
`/Data_8TB/lht/MK-UNet/BUSI/BUSI_split` directory split:

    train = 517, val = 64, test = 66   (Busi_split.py, seed 42, stratified 8:1:1)

SynFoC (``third_party/SynFoC-T20/train.py --dataset ClinicDB``) expects a dataset
root with ``train`` / ``validation`` / ``test`` phase directories.  This script
only *re-expresses* the existing split in that layout, using symlinks so the
original images and masks are never copied or modified, and writes one
``metadata.jsonl`` per phase with absolute image/mask paths.

Split membership is therefore byte-for-byte the frozen BUSI split; nothing is
re-shuffled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
SOURCE_SPLIT = Path("/Data_8TB/lht/MK-UNet/BUSI/BUSI_split")
OUT = ROOT / "work/busi_1pct_protocol"
DATA = OUT / "data"
LABELED_LIST = OUT / "busi_train1pct_labeled_images.txt"

# name in the source split dir -> phase name expected by SynFoC
PHASES = (("train", "train"), ("validation", "val"), ("test", "test"))

# Frozen 1% support selected for the BUSI 1pct mainline (seed 2026, train-only
# selection, GT-free).  Reused verbatim so the SynFoC baseline shares the exact
# labeled budget of the existing BUSI experiments.
SELECTED_IDS = [
    "benign (3)",
    "benign (125)",
    "benign (305)",
    "malignant (195)",
    "malignant (187)",
]
SELECTED_SOURCE = (
    ROOT
    / "mainline/experiments/busi_auto5_tp_1pct_20260913/SELECTED_SUPPORT_FROZEN.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    records_by_phase: dict[str, list[dict]] = {}

    for phase, source_name in PHASES:
        source_dir = SOURCE_SPLIT / source_name
        image_paths = sorted((source_dir / "images").glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(source_dir / "images")

        phase_dir = DATA / phase
        (phase_dir / "images").mkdir(parents=True, exist_ok=True)
        (phase_dir / "masks").mkdir(parents=True, exist_ok=True)

        records = []
        for image_path in image_paths:
            mask_path = source_dir / "masks" / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            linked_image = phase_dir / "images" / image_path.name
            linked_mask = phase_dir / "masks" / mask_path.name
            link(image_path, linked_image)
            link(mask_path, linked_mask)
            records.append(
                {
                    "file_name": str(linked_image.resolve()),
                    "mask_file_name": str(linked_mask.resolve()),
                    "merged_id": f"BUSI::{image_path.stem}",
                    "source_dataset": "BUSI",
                    "split": phase,
                    "source_file_name": str(image_path),
                    "source_mask_file_name": str(mask_path),
                }
            )
        records.sort(key=lambda item: item["file_name"])
        (phase_dir / "metadata.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
        )
        records_by_phase[phase] = records
        counts[phase] = len(records)

    # Frozen 1% labeled list: absolute paths, matching the SynFoC labeled_list
    # contract for the current ClinicDB adapter.
    train_index = {
        Path(record["file_name"]).stem: Path(record["file_name"])
        for record in records_by_phase["train"]
    }
    missing = [name for name in SELECTED_IDS if name not in train_index]
    if missing:
        raise ValueError(f"Frozen support images missing from train split: {missing}")
    labeled_paths = [train_index[name] for name in SELECTED_IDS]
    LABELED_LIST.write_text(
        "".join(f"{path}\n" for path in labeled_paths), encoding="utf-8"
    )

    train_total = counts["train"]
    labeled_count = len(labeled_paths)
    protocol = {
        "dataset": "BUSI",
        "source_split_root": str(SOURCE_SPLIT),
        "source_split_rule": "Busi_split.py, random.seed(42), per-class 8:1:1, files copied",
        "prepared_data_root": str(DATA),
        "phase_dir_map": {phase: source for phase, source in PHASES},
        "split": {
            "train": counts["train"],
            "validation": counts["validation"],
            "test": counts["test"],
        },
        "labeled_count": labeled_count,
        "labeled_fraction": labeled_count / train_total,
        "labeled_list": str(LABELED_LIST),
        "labeled_list_sha256": sha256(LABELED_LIST),
        "labeled_source": str(SELECTED_SOURCE),
        "labeled_ids": SELECTED_IDS,
        "labeled_images_are_symlinks": True,
        "split_membership_changed": False,
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "output": str(OUT),
        "counts": counts,
        "labeled_count": labeled_count,
        "labeled_fraction": labeled_count / train_total,
        "labeled_list": str(LABELED_LIST),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
