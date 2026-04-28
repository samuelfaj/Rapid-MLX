# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlash fallback policy."""

import concurrent.futures
import sys
import types
from dataclasses import dataclass

import pytest

from vllm_mlx.engine.dflash import DFlashEngine
from vllm_mlx.request import RequestOutput


def test_dflash_auto_disable_after_low_acceptance_window():
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        fallback_mode="ngram",
        disable_threshold=0.55,
        disable_window=2,
        disable_cooldown=3,
    )

    for ratio in (0.40, 0.45):
        engine._track_request_start("dflash")
        engine._active.proposed_tokens = 100
        engine._active.accepted_tokens = int(100 * ratio)
        engine._active.acceptance_ratio = ratio
        engine._track_request_end()

    assert engine._dflash_disabled_remaining == 3
    assert engine._choose_generation_mode() == "ngram"
    assert engine._dflash_disabled_remaining == 2


def test_dflash_does_not_disable_when_acceptance_is_good():
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        fallback_mode="ngram",
        disable_threshold=0.55,
        disable_window=2,
        disable_cooldown=3,
    )

    for ratio in (0.60, 0.75):
        engine._track_request_start("dflash")
        engine._active.proposed_tokens = 100
        engine._active.accepted_tokens = int(100 * ratio)
        engine._active.acceptance_ratio = ratio
        engine._track_request_end()

    assert engine._dflash_disabled_remaining == 0
    assert engine._choose_generation_mode() == "dflash"


def test_dflash_stats_include_fallback_policy():
    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        fallback_mode="ngram",
        disable_threshold=0.55,
        disable_window=4,
        disable_cooldown=8,
    )
    engine._fallback_responses = 2
    engine._fallback_proposed = 10
    engine._fallback_accepted = 7
    engine._dflash_disabled_remaining = 5

    stats = engine.get_stats()

    assert stats["dflash"]["fallback_mode"] == "ngram"
    assert stats["dflash"]["fallback_requests"] == 2
    assert stats["dflash"]["fallback_acceptance_ratio"] == 0.7
    assert stats["dflash"]["disabled_remaining"] == 5


@dataclass
class _FakeDFlashResponse:
    text: str
    tokens: list[int]
    prompt_tokens: int = 3
    generation_tokens: int = 1
    finish_reason: str | None = None
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    speculative_steps: int = 0
    avg_acceptance_ratio: float = 0.0
    block_size_history: tuple[int, ...] = ()


@pytest.mark.asyncio
async def test_dflash_terminal_none_len_error_becomes_clean_stop(monkeypatch):
    model_mlx = types.ModuleType("dflash.model_mlx")

    def fake_stream_generate(*args, **kwargs):
        yield _FakeDFlashResponse(text="ok", tokens=[1])
        raise TypeError("object of type 'NoneType' has no len()")

    model_mlx.stream_generate = fake_stream_generate
    monkeypatch.setitem(sys.modules, "dflash", types.ModuleType("dflash"))
    monkeypatch.setitem(sys.modules, "dflash.model_mlx", model_mlx)

    engine = DFlashEngine(model_name="dummy", drafter_path="dummy-drafter")
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
                max_tokens=4,
                temperature=0,
                top_p=1,
            )
        ]
    finally:
        await engine.stop()

    assert outputs[-1].finished is True
    assert outputs[-1].finish_reason == "stop"
    assert outputs[-1].text == "ok"


@pytest.mark.asyncio
async def test_dflash_cumulative_text_is_converted_to_delta(monkeypatch):
    model_mlx = types.ModuleType("dflash.model_mlx")

    def fake_stream_generate(*args, **kwargs):
        yield _FakeDFlashResponse(text="Hello", tokens=[1], generation_tokens=1)
        yield _FakeDFlashResponse(
            text="Hello world",
            tokens=[2],
            generation_tokens=2,
            finish_reason="stop",
        )

    model_mlx.stream_generate = fake_stream_generate
    monkeypatch.setitem(sys.modules, "dflash", types.ModuleType("dflash"))
    monkeypatch.setitem(sys.modules, "dflash.model_mlx", model_mlx)

    engine = DFlashEngine(model_name="dummy", drafter_path="dummy-drafter")
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
                max_tokens=4,
                temperature=0,
                top_p=1,
            )
        ]
    finally:
        await engine.stop()

    assert [output.new_text for output in outputs] == ["Hello", " world"]
    assert outputs[-1].text == "Hello world"
    assert outputs[-1].completion_tokens == 2


@pytest.mark.asyncio
async def test_dflash_ngram_fallback_terminal_none_len_error_becomes_clean_stop(
    monkeypatch,
):
    engine = DFlashEngine(model_name="dummy", drafter_path="dummy-drafter")
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def fake_fallback_iter(*args, **kwargs):
        yield (
            "output",
            RequestOutput(
                request_id="dflash-fallback",
                new_token_ids=[1],
                new_text="partial",
                output_token_ids=[1],
                output_text="partial",
                prompt_tokens=5,
                completion_tokens=1,
            ),
            {},
        )
        raise TypeError("object of type 'NoneType' has no len()")

    monkeypatch.setattr(engine, "_scheduler_fallback_iter", fake_fallback_iter)

    try:
        outputs = [
            output
            async for output in engine._stream_scheduler_fallback(
                "prompt",
                max_tokens=4,
                temperature=0,
                top_p=1,
                mode="ngram",
            )
        ]
    finally:
        await engine.stop()

    assert [output.new_text for output in outputs] == ["partial", ""]
    assert outputs[-1].finished is True
    assert outputs[-1].finish_reason == "stop"
    assert outputs[-1].text == "partial"
