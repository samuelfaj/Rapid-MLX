# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlashEngine DDTree routing."""

import concurrent.futures
import sys
import threading
import types

import numpy as np
import pytest

from vllm_mlx.engine.dflash import DFlashEngine
from vllm_mlx.speculative.ddtree.engine import _looks_like_tool_call_draft
from vllm_mlx.speculative.ddtree.tree import build_ddtree_tree_from_topk


class FakeTokenArray(list):
    def tolist(self):
        return list(self)


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

    class FakeTokenizer:
        def encode(self, prompt):
            return prompt.split()

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
