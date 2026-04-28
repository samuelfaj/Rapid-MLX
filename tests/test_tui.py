# SPDX-License-Identifier: Apache-2.0
"""Focused tests for TUI metric formatting."""

from vllm_mlx.tui import _build_screen, _weighted_tps


def test_weighted_tps_uses_token_weighted_rate():
    entries = [
        {"generated_tokens": 100, "generation_tps": 100.0},
        {"generated_tokens": 100, "generation_tps": 10.0},
    ]

    assert round(_weighted_tps(entries, "generated_tokens", "generation_tps"), 1) == 18.2


def test_tui_renders_ngram_acceptance_in_request_sections():
    requests_data = {
        "entries": [
            {
                "surface": "/v1/chat/completions",
                "finished_at": 1,
                "elapsed": 2.0,
                "ttft": 0.1,
                "prompt_tokens": 10,
                "generated_tokens": 20,
                "generation_tps": 40.0,
                "prompt_tps": 100.0,
                "finish_reason": "stop",
                "acceptance_ratio": 0.5,
            }
        ],
        "active": None,
    }
    status = {
        "status": "idle",
        "model": "test",
        "engine_type": "batched",
        "ngram_mod": {
            "enabled": True,
            "proposed_tokens": 10,
            "accepted_tokens": 5,
            "rejected_tokens": 5,
            "disabled_steps": 0,
            "acceptance_rate": 0.5,
        },
    }

    screen = _build_screen(
        "http://127.0.0.1:8010",
        "?",
        1.0,
        {"model_loaded": True},
        status,
        requests_data,
        [],
        False,
    )

    assert "ngram accept" in screen
    assert "spec accept" in screen
    assert "accept" in screen
    assert "50%" in screen
