import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "round3_train", ROOT / "scripts/train_round3_tracker.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:  # sam3 is an external dependency
        pytest.skip(f"external dependency missing: {exc}", allow_module_level=True)
    return module


module = _load_module()

def test_memory_policy_is_exact():
    assert module.is_memory_parameter("maskmem_backbone.out_proj.weight")
    assert module.is_memory_parameter("transformer.encoder.layers.0.linear1.weight")
    assert module.is_memory_parameter("maskmem_tpos_enc")
    assert module.is_memory_parameter("no_mem_embed")
    assert not module.is_memory_parameter("sam_mask_decoder.iou_prediction_head.layers.0.weight")
    assert not module.is_memory_parameter("sam_prompt_encoder.no_mask_embed.weight")
    assert not module.is_memory_parameter("obj_ptr_proj.layers.0.weight")
