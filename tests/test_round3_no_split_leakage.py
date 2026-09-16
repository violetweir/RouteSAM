import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "work/rerun_c0_256_round3_tracker_stage4/sequence_manifests/train_sequences.jsonl"
PROTOCOL = ROOT / "work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl"

pytestmark = pytest.mark.skipif(
    not (MANIFEST.exists() and PROTOCOL.exists()),
    reason="run outputs live in the original work/ tree and are not shipped in this repository",
)


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_round3_no_split_leakage():
    manifest = rows(MANIFEST)
    protocol = rows(PROTOCOL)
    by_split = {split: {x["merged_id"] for x in protocol if x["split"] == split} for split in ("train", "validation", "test")}
    train_sequence_ids = {frame_id for row in manifest for frame_id in row["frame_ids"]}
    assert train_sequence_ids <= by_split["train"]
    assert not train_sequence_ids & by_split["validation"]
    assert not train_sequence_ids & by_split["test"]
