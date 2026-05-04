# SPDX-License-Identifier: Apache-2.0
"""Tests for live TUI metric formatting helpers."""

from vllm_mlx.tui import _entry_tokens_per_second


def test_entry_tokens_per_second_prefers_decode_tps_over_generation_spike():
    assert (
        _entry_tokens_per_second(
            {
                "generated_tokens": 100,
                "elapsed": 10.0,
                "ttft": 2.0,
                "decode_tps": 12.5,
                "generation_tps": 60000.0,
            }
        )
        == 12.5
    )


def test_entry_tokens_per_second_uses_active_request_tokens_per_second():
    assert (
        _entry_tokens_per_second(
            {
                "completion_tokens": 25,
                "elapsed_s": 5.0,
                "tokens_per_second": 6.25,
            }
        )
        == 6.25
    )
