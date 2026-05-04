# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlashEngine DDTree routing."""

import concurrent.futures
import sys
import threading
import types

import numpy as np
import pytest

from vllm_mlx.engine.dflash import DFlashEngine, _is_greedy_sampling
from vllm_mlx.speculative.ddtree.engine import _looks_like_tool_call_draft
from vllm_mlx.speculative.ddtree.tree import build_ddtree_tree_from_topk
from vllm_mlx.speculative.prefill import SpeculativePrefillConfig


class FakeTokenArray(list):
    def tolist(self):
        return list(self)


class FakeTokenizer:
    def encode(self, prompt):
        return prompt.split()


def test_none_sampling_is_greedy_for_default_temperature_zero():
    assert _is_greedy_sampling(None, None) is True
    assert _is_greedy_sampling(0, 0.9) is True
    assert _is_greedy_sampling(0, 1) is True
    assert _is_greedy_sampling(0.7, 1) is False


@pytest.mark.asyncio
async def test_dflash_engine_routes_to_ddtree(monkeypatch):
    fake_engine = types.ModuleType("vllm_mlx.speculative.ddtree.engine")

    def fake_generate_ddtree(**kwargs):
        assert kwargs["tree_budget"] == 4
        assert kwargs["block_size"] is None
        assert kwargs["stop_strings"] == ["STOP"]
        assert "</tool_call>" in kwargs["stop_after_strings"]
        return {
            "text": "ok",
            "generated_token_ids": [1, 2],
            "prompt_tokens": 3,
            "generated_tokens": 2,
            "finish_reason": "stop",
            "proposed_tokens": 4,
            "accepted_tokens": 3,
            "speculative_steps": 1,
            "avg_acceptance_ratio": 0.75,
            "block_size_history": [2],
            "avg_tree_node_count": 5.0,
            "ddtree_fast_path_ratio": 1.0,
            "tree_budget": 4,
            "generation_tps": 123.0,
            "ddtree_phase_timings_us": {"draft": 10.0, "tree_verify": 20.0},
        }

    fake_generate_ddtree.__name__ = "generate_ddtree"
    fake_engine.generate_ddtree = fake_generate_ddtree
    fake_engine.tokenize_prompt = lambda tokenizer, prompt: FakeTokenArray([1, 2, 3])
    monkeypatch.setitem(
        sys.modules,
        "vllm_mlx.speculative.ddtree.engine",
        fake_engine,
    )

    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        block_size=2,
        ddtree_budget=4,
    )
    engine._loaded = True
    engine._model = object()
    engine._drafter = object()
    engine._tokenizer = object()
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        outputs = [
            output
            async for output in engine.stream_generate(
                "prompt",
                max_tokens=8,
                temperature=0,
                top_p=1,
                stop=["STOP"],
                tools_requested=True,
            )
        ]
    finally:
        await engine.stop()

    assert outputs[-1].text == "ok"
    assert outputs[-1].completion_tokens == 2

    stats = engine.get_stats()
    assert stats["dflash"]["mode"] == "ddtree"
    assert stats["dflash"]["ddtree_budget"] == 4
    assert stats["dflash"]["ddtree_requests"] == 1
    assert stats["dflash"]["ddtree_last_generation_tps"] == 123.0
    assert stats["dflash"]["ddtree_last_phase_timings_us"]["tree_verify"] == 20.0


@pytest.mark.asyncio
async def test_dflash_engine_passes_ngram_first_config(monkeypatch):
    fake_engine = types.ModuleType("vllm_mlx.speculative.ddtree.engine")

    def fake_generate_ddtree(**kwargs):
        assert kwargs["tree_budget"] == 4
        assert kwargs["ngram_num_draft_tokens"] == 6
        assert kwargs["ngram_size"] == 3
        assert kwargs["ngram_min_matches"] == 1
        assert kwargs["ngram_disable_threshold"] == 0.5
        assert kwargs["ngram_disable_window"] == 2
        assert kwargs["ngram_disable_cooldown"] == 7
        return {
            "text": "ok",
            "generated_token_ids": [1],
            "prompt_tokens": 3,
            "prefill_seconds": 0.25,
            "generated_tokens": 1,
            "finish_reason": "stop",
            "proposed_tokens": 2,
            "accepted_tokens": 1,
            "speculative_steps": 1,
            "avg_acceptance_ratio": 0.5,
            "block_size_history": [2],
            "avg_tree_node_count": 4.0,
            "ddtree_fast_path_ratio": 1.0,
            "tree_budget": 4,
            "generation_tps": 99.0,
            "ngram_acceptance_ratio": 0.75,
            "ngram_cycles_completed": 3,
            "ngram_fallback_cycles": 2,
            "ngram_tool_guard_cycles": 1,
            "ddtree_phase_timings_us": {"draft": 1.0, "tree_verify_linear": 2.0},
        }

    fake_engine.generate_ddtree = fake_generate_ddtree
    fake_engine.tokenize_prompt = lambda tokenizer, prompt: FakeTokenArray([1, 2, 3])
    monkeypatch.setitem(
        sys.modules,
        "vllm_mlx.speculative.ddtree.engine",
        fake_engine,
    )

    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        fallback_mode="ngram",
        ngram_num_draft_tokens=6,
        ngram_size=3,
        ngram_min_matches=1,
        ngram_disable_threshold=0.5,
        ngram_disable_window=2,
        ngram_disable_cooldown=7,
    )
    engine._loaded = True
    engine._model = object()
    engine._drafter = object()
    engine._tokenizer = object()
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    active_stats = None
    try:
        outputs = []
        async for output in engine.stream_generate(
            "Create CRUD JSON tests for a schema.",
            max_tokens=8,
            temperature=0,
            top_p=1,
        ):
            outputs.append(output)
            active_stats = engine.get_stats()
    finally:
        await engine.stop()

    assert outputs[-1].text == "ok"

    stats = engine.get_stats()
    assert active_stats is not None
    assert active_stats["requests"][0]["tokens_per_second"] == 99.0
    assert active_stats["requests"][0]["ttft_s"] == 0.25
    assert active_stats["requests"][0]["completion_tokens"] == 1
    assert stats["dflash"]["mode"] == "ddtree-ngram"
    assert stats["dflash"]["ngram_first_enabled"] is True
    assert stats["dflash"]["ngram_num_draft_tokens"] == 6
    assert stats["dflash"]["ngram_last_acceptance_ratio"] == 0.75
    assert stats["dflash"]["ngram_last_cycles"] == 3
    assert stats["dflash"]["ngram_last_fallback_cycles"] == 2
    assert stats["dflash"]["ngram_last_tool_guard_cycles"] == 1


def test_ddtree_builder_preseeds_top1_chain(monkeypatch):
    monkeypatch.delenv("DDTREE_CHAIN_PRESEED", raising=False)
    top_token_ids = np.array(
        [
            [10, 11],
            [20, 21],
            [30, 31],
            [40, 41],
        ],
        dtype=np.int64,
    )
    top_log_probs = np.array(
        [
            [-0.01, -0.02],
            [-1.20, -1.21],
            [-1.20, -1.21],
            [-1.20, -1.21],
        ],
        dtype=np.float32,
    )

    tree = build_ddtree_tree_from_topk(top_token_ids, top_log_probs, budget=4)

    assert tree.node_token_ids.tolist() == [10, 20, 30, 40]
    assert tree.parents == [-1, 0, 1, 2, 3]


def test_ddtree_ngram_tool_call_guard_detects_xml_markers():
    class FakeTokenizer:
        def decode(self, tokens, *args, **kwargs):
            if 4 in tokens:
                return "<tool_call><function=bash>"
            return "plain text"

    assert _looks_like_tool_call_draft(FakeTokenizer(), [1, 2], 3, [4])
    assert not _looks_like_tool_call_draft(FakeTokenizer(), [1, 2], 3, [5])


def test_dflash_agentic_target_fallback_only_for_long_tool_prompts(monkeypatch):
    monkeypatch.setenv("DFLASH_AGENTIC_TARGET_FALLBACK", "1")
    monkeypatch.setenv("DFLASH_AGENTIC_TARGET_FALLBACK_MIN_PROMPT_TOKENS", "4")

    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        fallback_mode="ngram",
    )
    engine._tokenizer = FakeTokenizer()

    assert engine._should_use_agentic_target_fallback(
        "one two three four", tools_requested=True
    )
    assert not engine._should_use_agentic_target_fallback(
        "one two three", tools_requested=True
    )
    assert not engine._should_use_agentic_target_fallback(
        "one two three four", tools_requested=False
    )


def test_ddtree_prefix_cache_enabled_by_default_and_env_can_disable(monkeypatch):
    monkeypatch.delenv("DFLASH_DDTREE_CAPTURE_CACHE", raising=False)
    enabled = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
    )
    assert enabled._ddtree_capture_cache is True

    monkeypatch.setenv("DFLASH_DDTREE_CAPTURE_CACHE", "0")
    disabled = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
    )
    assert disabled._ddtree_capture_cache is False


def test_agentic_auto_allows_suffix_prefill_for_large_uncached_scaffold(monkeypatch):
    monkeypatch.setenv("DFLASH_AGENTIC_POLICY_MAX_PREFILL", "4")
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", "0")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
        speculative_prefill_config=SpeculativePrefillConfig(enabled=True),
    )
    engine._tokenizer = FakeTokenizer()

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four five six",
        tools_requested=True,
        max_tokens=1024,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=2,
    )

    assert mode == "ddtree"
    assert reason == "long_generation_ddtree"
    assert metadata["cached_tokens"] == 2
    assert metadata["remaining_prefill_tokens"] == 4

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four five six seven",
        tools_requested=True,
        max_tokens=1024,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=2,
    )

    assert mode == "ddtree"
    assert reason == "long_prefill_suffix_speculative_prefill"
    assert metadata["remaining_prefill_tokens"] == 5


def test_agentic_auto_targets_only_for_large_uncached_scaffold_without_prefill(monkeypatch):
    monkeypatch.setenv("DFLASH_AGENTIC_POLICY_MAX_PREFILL", "4")
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", "0")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
        speculative_prefill_config=SpeculativePrefillConfig(enabled=False),
    )
    engine._tokenizer = FakeTokenizer()

    mode, reason, _ = engine._agentic_policy_decision(
        "one two three four five six seven",
        tools_requested=True,
        max_tokens=1024,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=2,
    )

    assert mode == "target-fallback"
    assert reason == "prefill_too_large"


def test_agentic_auto_ngram_only_for_long_text_phase(monkeypatch):
    monkeypatch.delenv("DFLASH_AGENTIC_NGRAM_LONG_TEXT", raising=False)
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", "0")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._tokenizer = FakeTokenizer()

    mode, reason, _ = engine._agentic_policy_decision(
        "one two three four five six",
        tools_requested=True,
        max_tokens=1024,
        greedy_request=True,
        phase="long_text_or_code",
        cached_tokens=0,
    )

    assert mode == "ddtree-ngram"
    assert reason == "long_generation_ngram"
    assert engine._should_enable_ngram_first("one two three", "tool_json") is False
    assert engine._should_enable_ngram_first("one two three", "repair") is False
    assert engine._should_enable_ngram_first("one two three", "validation") is False

    monkeypatch.setenv("DFLASH_AGENTIC_NGRAM_LONG_TEXT", "0")
    assert engine._should_enable_ngram_first(
        "one two three",
        "long_text_or_code",
    ) is False


def test_agentic_auto_targets_outside_ddtree_sweet_spot(monkeypatch):
    monkeypatch.setenv("DFLASH_AGENTIC_DDTREE_MAX_PROMPT_TOKENS", "4")
    monkeypatch.setenv("DFLASH_AGENTIC_DDTREE_MAX_TOKENS", "1024")
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", "0")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._tokenizer = FakeTokenizer()

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four five",
        tools_requested=True,
        max_tokens=1024,
        greedy_request=True,
        phase="long_text_or_code",
        cached_tokens=4,
    )

    assert mode == "target-fallback"
    assert reason == "prompt_outside_ddtree_sweet_spot"
    assert metadata["ddtree_prompt_limit"] == 4

    monkeypatch.setenv("DFLASH_AGENTIC_DDTREE_MAX_PROMPT_TOKENS", "16")
    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four five",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="long_text_or_code",
        cached_tokens=0,
    )

    assert mode == "ddtree-ngram"
    assert reason == "long_generation_ngram"
    assert metadata["max_tokens_over_ddtree_limit"] is True

    monkeypatch.setenv("DFLASH_AGENTIC_DDTREE_STRICT_MAX_TOKENS", "1")
    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four five",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="long_text_or_code",
        cached_tokens=0,
    )

    assert mode == "target-fallback"
    assert reason == "max_tokens_outside_ddtree_sweet_spot"
    assert metadata["ddtree_max_tokens_limit"] == 1024


def test_agentic_auto_adaptive_ddtree_waits_for_slow_target(monkeypatch):
    monkeypatch.delenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", raising=False)
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_MIN_TARGET_SAMPLES", "2")
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_TARGET_TPS_TRIGGER", "18")
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_MIN_PROMPT_TOKENS", "4")
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_EXPLORE_EVERY", "0")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._tokenizer = FakeTokenizer()

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four five",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=0,
    )

    assert mode == "target-fallback"
    assert reason == "adaptive_target_warmup"
    assert metadata["adaptive_ddtree_ready"] is False

    engine._agentic_policy_history.extend(
        [
            {
                "mode": "target-prefix-cache",
                "phase": "initial_scaffold",
                "generation_tps": 16.0,
                "generated_tokens": 128,
            },
            {
                "mode": "target-prefix-cache",
                "phase": "long_text_or_code",
                "generation_tps": 15.0,
                "generated_tokens": 96,
            },
        ]
    )

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four five",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=0,
    )

    assert mode == "ddtree"
    assert reason == "long_generation_ddtree"
    assert metadata["adaptive_ddtree_ready"] is True
    assert metadata["recent_target_tps"] == 15.5


def test_agentic_history_records_acceptance_length_and_effective_tps():
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._track_request_start("ddtree")
    assert engine._active is not None
    engine._active.started_at -= 2.0
    engine._active.agentic_phase = "initial_scaffold"
    engine._active.agentic_policy_decision = "ddtree"
    engine._active.agentic_policy_reason = "long_generation_ddtree"
    engine._active.agentic_policy_bucket = "initial_scaffold|4-8k|513-1024|greedy"
    engine._update_active(
        types.SimpleNamespace(
            tokens=[1],
            prompt_tokens=5000,
            prefill_seconds=0.5,
            generation_tokens=100,
            generation_tps=1000.0,
            proposed_tokens=60,
            accepted_tokens=24,
            speculative_steps=6,
            avg_acceptance_ratio=0.4,
            avg_acceptance_length=0.0,
            block_size_history=[2],
            avg_tree_node_count=4.0,
            ddtree_fast_path_ratio=1.0,
            tree_budget=4,
            ngram_acceptance_ratio=0.0,
            ngram_cycles=0,
            ngram_proposed_tokens=0,
            ngram_accepted_tokens=0,
            ngram_fallback_cycles=0,
            ngram_tool_guard_cycles=0,
            cache_hit_type=None,
            cached_tokens=1000,
            phase_timings_us={},
        ),
        new_text="ok",
    )
    engine._track_request_end()

    row = engine.get_stats()["dflash"]["agentic_policy_history"][-1]
    assert row["acceptance_length"] == 4.0
    assert row["generation_tps"] == 1000.0
    assert 40.0 <= row["effective_tps"] <= 60.0
    assert row["uncached_tokens"] == 4000
    assert row["tree_budget"] == 4


def test_agentic_auto_uses_acceptance_length_bucket_cooldown(monkeypatch):
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", "0")
    monkeypatch.setenv("DFLASH_AGENTIC_DDTREE_AL_DISABLE", "2.5")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._tokenizer = FakeTokenizer()
    bucket = "initial_scaffold|0-4k|1025+|greedy"
    engine._agentic_policy_history.append(
        {
            "mode": "ddtree",
            "bucket": bucket,
            "phase": "initial_scaffold",
            "generated_tokens": 128,
            "effective_tps": 12.0,
            "acceptance_length": 1.5,
            "tree_budget": 4,
        }
    )

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=0,
    )

    assert mode == "target-fallback"
    assert reason == "low_acceptance_length"
    assert metadata["ddtree_recent_al"] == 1.5
    assert metadata["bucket_cooldown"] > 0


def test_agentic_auto_selects_best_ddtree_budget_from_bucket_history(monkeypatch):
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", "0")
    monkeypatch.setenv("DFLASH_AGENTIC_DDTREE_BUDGET_CANDIDATES", "2,4,8")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._tokenizer = FakeTokenizer()
    bucket = "initial_scaffold|0-4k|1025+|greedy"
    engine._agentic_policy_history.extend(
        [
            {
                "mode": "ddtree",
                "bucket": bucket,
                "phase": "initial_scaffold",
                "generated_tokens": 128,
                "effective_tps": 10.0,
                "acceptance_length": 4.5,
                "tree_budget": 4,
            },
            {
                "mode": "ddtree",
                "bucket": bucket,
                "phase": "initial_scaffold",
                "generated_tokens": 128,
                "effective_tps": 14.0,
                "acceptance_length": 6.0,
                "tree_budget": 8,
            },
        ]
    )

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=0,
    )

    assert mode == "ddtree"
    assert reason == "long_generation_ddtree"
    assert metadata["ddtree_budget"] == 8


def test_agentic_auto_ngram_low_acceptance_disables_ngram_only(monkeypatch):
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", "0")
    monkeypatch.setenv("DFLASH_AGENTIC_NGRAM_MIN_ACCEPTANCE", "0.30")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._tokenizer = FakeTokenizer()
    bucket = "long_text_or_code|0-4k|1025+|greedy"
    engine._agentic_policy_history.append(
        {
            "mode": "ddtree-ngram",
            "bucket": bucket,
            "phase": "long_text_or_code",
            "generated_tokens": 128,
            "effective_tps": 12.0,
            "acceptance_length": 4.5,
            "tree_budget": 4,
            "ngram_cycles": 3,
            "ngram_acceptance_ratio": 0.10,
        }
    )

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="long_text_or_code",
        cached_tokens=0,
    )

    assert mode == "ddtree"
    assert reason == "long_generation_ddtree"
    assert metadata["ngram_allowed"] is False


def test_agentic_auto_bucket_target_winner_blocks_ddtree(monkeypatch):
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", "0")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._tokenizer = FakeTokenizer()
    bucket = "initial_scaffold|0-4k|1025+|greedy"
    engine._agentic_policy_history.extend(
        [
            {
                "mode": "target-prefix-cache",
                "bucket": bucket,
                "phase": "initial_scaffold",
                "generated_tokens": 128,
                "effective_tps": 30.0,
            },
            {
                "mode": "target-prefix-cache",
                "bucket": bucket,
                "phase": "initial_scaffold",
                "generated_tokens": 128,
                "effective_tps": 32.0,
            },
            {
                "mode": "ddtree",
                "bucket": bucket,
                "phase": "initial_scaffold",
                "generated_tokens": 128,
                "effective_tps": 20.0,
                "acceptance_length": 3.0,
                "tree_budget": 4,
            },
            {
                "mode": "ddtree",
                "bucket": bucket,
                "phase": "initial_scaffold",
                "generated_tokens": 128,
                "effective_tps": 19.0,
                "acceptance_length": 3.5,
                "tree_budget": 4,
            },
        ]
    )

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=0,
    )

    assert mode == "target-fallback"
    assert reason == "bucket_target_winner"
    assert metadata["target_known_winner"] is True


def test_agentic_auto_deterministic_exploration_allows_ddtree(monkeypatch):
    monkeypatch.delenv("DFLASH_AGENTIC_ADAPTIVE_DDTREE", raising=False)
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_MIN_TARGET_SAMPLES", "4")
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_TARGET_TPS_TRIGGER", "1")
    monkeypatch.setenv("DFLASH_AGENTIC_ADAPTIVE_MIN_PROMPT_TOKENS", "4")
    monkeypatch.setenv("DFLASH_AGENTIC_POLICY_EXPLORE_EVERY", "8")
    monkeypatch.setenv("DFLASH_AGENTIC_POLICY_EXPLORE_MIN_OUTPUT_TOKENS", "64")
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        agentic_speculative_policy="auto",
    )
    engine._tokenizer = FakeTokenizer()
    engine._lifetime_responses = 8

    mode, reason, metadata = engine._agentic_policy_decision(
        "one two three four",
        tools_requested=True,
        max_tokens=2048,
        greedy_request=True,
        phase="initial_scaffold",
        cached_tokens=0,
    )

    assert mode == "ddtree"
    assert reason == "long_generation_ddtree"
    assert metadata["adaptive_ddtree_ready"] is True


@pytest.mark.asyncio
async def test_dflash_target_fallback_closes_generator_on_worker(monkeypatch):
    import mlx_lm

    threads: dict[str, int] = {}

    class FakeResponse:
        text = "</tool_call>"
        token = 1
        prompt_tokens = 4
        prompt_tps = 2.0
        generation_tokens = 1
        generation_tps = 10.0
        finish_reason = None

    class FakeGenerator:
        def __init__(self):
            self._done = False

        def __iter__(self):
            return self

        def __next__(self):
            threads["next"] = threading.get_ident()
            if self._done:
                raise StopIteration
            self._done = True
            return FakeResponse()

        def close(self):
            threads["close"] = threading.get_ident()

    monkeypatch.setattr(
        mlx_lm,
        "stream_generate",
        lambda *args, **kwargs: FakeGenerator(),
    )

    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
    )
    engine._model = object()
    engine._tokenizer = object()
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        outputs = [
            output
            async for output in engine._stream_target(
                "prompt",
                max_tokens=8,
                temperature=0,
                top_p=1,
                stop=None,
                tools_requested=True,
            )
        ]
    finally:
        await engine.stop()

    assert outputs[-1].finish_reason == "stop"
    assert threads["close"] == threads["next"]


@pytest.mark.asyncio
async def test_dflash_stop_closes_active_target_fallback_generator(monkeypatch):
    import mlx_lm

    threads: dict[str, int] = {}

    class FakeResponse:
        text = "partial"
        token = 1
        prompt_tokens = 4
        prompt_tps = 2.0
        generation_tokens = 1
        generation_tps = 10.0
        finish_reason = None

    class FakeGenerator:
        def __iter__(self):
            return self

        def __next__(self):
            threads["next"] = threading.get_ident()
            return FakeResponse()

        def close(self):
            threads["close"] = threading.get_ident()

    monkeypatch.setattr(
        mlx_lm,
        "stream_generate",
        lambda *args, **kwargs: FakeGenerator(),
    )

    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
    )
    engine._model = object()
    engine._tokenizer = object()
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    stream = engine._stream_target(
        "prompt",
        max_tokens=8,
        temperature=0,
        top_p=1,
        stop=None,
        tools_requested=False,
    )
    output = await stream.__anext__()
    assert output.text == "partial"

    await engine.stop()
    await stream.aclose()

    assert threads["close"] == threads["next"]
