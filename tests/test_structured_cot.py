# SPDX-License-Identifier: Apache-2.0
"""Tests for Structured-CoT constrained decoding."""

import pytest

from vllm_mlx.api.structured_cot import (
    StructuredCoTLogitsProcessor,
    is_structured_cot_prefix,
    normalize_structured_cot_mode,
    structured_cot_complete,
)


class TinyTokenizer:
    def __init__(self, pieces):
        self._pieces = list(pieces)

    def __len__(self):
        return len(self._pieces)

    def decode(self, token_ids):
        return "".join(self._pieces[i] for i in token_ids)


def test_normalize_structured_cot_mode():
    assert normalize_structured_cot_mode(True) == "plan"
    assert normalize_structured_cot_mode(False) is None
    assert normalize_structured_cot_mode("basic") == "basic"
    assert normalize_structured_cot_mode("lcb_plan") == "plan"

    with pytest.raises(ValueError):
        normalize_structured_cot_mode("verbose")


def test_basic_prefix_accepts_structured_shape():
    assert is_structured_cot_prefix("", "basic")
    assert is_structured_cot_prefix("<think>\nGOAL: solve\nAPP", "basic")
    assert is_structured_cot_prefix(
        "<think>\nGOAL: solve\nAPPROACH: dp\nEDGE: empty\n</think>\n\ncode",
        "basic",
    )


def test_basic_prefix_rejects_empty_line():
    assert not is_structured_cot_prefix("<think>\nGOAL: \n", "basic")


def test_plan_completion_after_close():
    text = (
        "<think>\n"
        "GOAL: solve\n"
        "STATE: index\n"
        "ALGO: dp\n"
        "EDGE: empty\n"
        "VERIFY: tests\n"
        "</think>\n\n"
    )
    assert structured_cot_complete(text, "plan")


def test_logits_processor_masks_invalid_start_tokens():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx not available")

    tokenizer = TinyTokenizer(["<think>\n", "hello", "x"])
    processor = StructuredCoTLogitsProcessor(tokenizer, "basic")
    logits = mx.zeros((1, 3))

    masked = processor(mx.array([], dtype=mx.int32), logits)
    mx.eval(masked)

    assert masked[0, 0].item() == 0.0
    assert masked[0, 1].item() == -float("inf")
    assert masked[0, 2].item() == -float("inf")


def test_logits_processor_ignores_prompt_prefix():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx not available")

    tokenizer = TinyTokenizer(["prompt", "<think>\n", "hello", "x"])
    processor = StructuredCoTLogitsProcessor(
        tokenizer,
        "basic",
        prompt_token_count=1,
    )
    logits = mx.zeros((1, 4))

    masked = processor(mx.array([0], dtype=mx.int32), logits)
    mx.eval(masked)

    assert masked[0, 1].item() == 0.0
    assert masked[0, 2].item() == -float("inf")
    assert masked[0, 3].item() == -float("inf")
