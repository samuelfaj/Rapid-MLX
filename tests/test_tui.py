# SPDX-License-Identifier: Apache-2.0
"""Tests for live TUI metric formatting helpers."""

from vllm_mlx.tui import _entry_tokens_per_second, _spec_path


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


def test_spec_path_shows_dash_without_actual_speculative_work():
    assert _spec_path({"spec_mode": "target-prefix-cache"}) == "-"
    assert _spec_path({"spec_mode": "target-fallback"}) == "-"
    assert _spec_path({"spec_mode": "ddtree", "speculative_proposed_tokens": 0}) == "-"
    assert _spec_path({"spec_mode": "ddtree-ngram", "ngram_cycles": 0}) == "-"


def test_spec_path_shows_technique_when_used():
    assert (
        _spec_path(
            {
                "spec_mode": "ddtree",
                "speculative_proposed_tokens": 12,
                "speculative_steps": 3,
            }
        )
        == "ddtree"
    )
    assert (
        _spec_path(
            {
                "spec_mode": "ddtree-ngram",
                "ngram_cycles": 4,
                "ngram_fallback_cycles": 2,
            }
        )
        == "ng+tree 4/2"
    )
