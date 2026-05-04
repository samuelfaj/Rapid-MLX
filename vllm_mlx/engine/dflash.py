# SPDX-License-Identifier: Apache-2.0
"""
DFlash speculative-decoding engine.

Wraps the dflash MLX runtime (https://github.com/dflash/dflash) so that
`rapid-mlx serve <target> --drafter <drafter>` runs block-based draft+verify
with a separate drafter model that conditions on hidden states from selected
target layers.

This engine is single-request: only one prompt is generated at a time; further
requests wait on an asyncio.Lock. Continuous batching is not supported in this
mode.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import logging
import os
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from ..api.utils import clean_output_text
from ..speculative.prefill import SpeculativePrefillConfig
from .base import GenerationOutput
from .batched import BatchedEngine

logger = logging.getLogger(__name__)
_DDTREE_STREAM_IDLE_TIMEOUT_SECONDS = float(
    os.environ.get("DFLASH_DDTREE_STREAM_IDLE_TIMEOUT_SECONDS", "300")
)
_DDTREE_STOP_WAIT_SECONDS = float(
    os.environ.get("DFLASH_DDTREE_STOP_WAIT_SECONDS", "2")
)


_TOOL_STOP_AFTER_STRINGS = (
    "</tool_call>",
    "</minimax:tool_call>",
    "</invoke>",
    "<tool_call|>",
    "<|tool_call_end|>",
    "<｜tool▁call▁end｜>",
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_greedy_sampling(temperature: float | None, top_p: float | None = None) -> bool:
    temp = 0.0 if temperature is None else float(temperature)
    return temp == 0.0


def _init_dflash_step_thread() -> None:
    """Mirror engine_core._init_mlx_step_thread for our private executor."""
    import mlx.core as mx

    stream = mx.new_stream(mx.default_device())
    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream
    logger.info("DFlash step thread initialized: stream=%s", stream)


@dataclass
class _ActiveRequest:
    started_at: float
    mode: str = "dflash"
    first_token_at: float | None = None
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    generated_tokens: int = 0
    generation_tps: float = 0.0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    speculative_steps: int = 0
    acceptance_ratio: float = 0.0
    block_size: int = 0
    block_history: list[int] = field(default_factory=list)
    tree_budget: int = 0
    avg_tree_node_count: float = 0.0
    ddtree_fast_path_ratio: float = 0.0
    ngram_acceptance_ratio: float = 0.0
    ngram_cycles: int = 0
    ngram_fallback_cycles: int = 0
    ngram_tool_guard_cycles: int = 0
    cache_hit_type: str | None = None
    cached_tokens: int = 0
    agentic_phase: str | None = None
    agentic_policy_decision: str | None = None
    agentic_policy_reason: str | None = None
    phase_timings_us: dict[str, float] = field(default_factory=dict)


@dataclass
class _DDTreeCacheStats:
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


@dataclass
class _DDTreePrefixCacheFetch:
    state: Any | None
    hit_type: str = "miss"
    cached_tokens: int = 0


@dataclass
class _TargetPrefixState:
    prompt_tokens: tuple[int, ...]
    prompt_cache: list[Any]
    logprobs: Any


class _DDTreePrefixStateCache:
    """Small LRU cache for DFlash PromptPrefillState objects."""

    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max(0, int(max_entries))
        self._entries: dict[tuple[int, ...], Any] = {}
        self._lru: list[tuple[int, ...]] = []
        self.stats = _DDTreeCacheStats()

    def fetch(self, prompt_tokens: list[int]) -> _DDTreePrefixCacheFetch:
        self.stats.total_queries += 1
        if self.max_entries <= 0 or not prompt_tokens:
            self.stats.misses += 1
            return _DDTreePrefixCacheFetch(None)

        prompt_tuple = tuple(int(token) for token in prompt_tokens)
        best_key: tuple[int, ...] | None = None
        for key in self._entries:
            if len(key) > len(prompt_tuple):
                continue
            if prompt_tuple[: len(key)] == key and (
                best_key is None or len(key) > len(best_key)
            ):
                best_key = key

        if best_key is None:
            self.stats.misses += 1
            return _DDTreePrefixCacheFetch(None)

        self.stats.hits += 1
        self.stats.tokens_saved += len(best_key)
        self._touch(best_key)
        hit_type = "exact" if len(best_key) == len(prompt_tuple) else "prefix"
        return _DDTreePrefixCacheFetch(
            copy.deepcopy(self._entries[best_key]),
            hit_type=hit_type,
            cached_tokens=len(best_key),
        )

    def peek_cached_tokens(self, prompt_tokens: list[int]) -> int:
        if self.max_entries <= 0 or not prompt_tokens:
            return 0
        prompt_tuple = tuple(int(token) for token in prompt_tokens)
        best_len = 0
        for key in self._entries:
            if len(key) <= len(prompt_tuple) and prompt_tuple[: len(key)] == key:
                best_len = max(best_len, len(key))
        return best_len

    def store(self, prompt_tokens: list[int], state: Any | None) -> None:
        if self.max_entries <= 0 or state is None or not prompt_tokens:
            return
        key = tuple(int(token) for token in prompt_tokens)
        self._entries[key] = copy.deepcopy(state)
        self._touch(key)
        while len(self._entries) > self.max_entries:
            evict_key = self._lru.pop(0)
            if evict_key in self._entries:
                del self._entries[evict_key]
                self.stats.evictions += 1

    def _touch(self, key: tuple[int, ...]) -> None:
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def get_stats(self) -> dict[str, Any]:
        data = self.stats.to_dict()
        data.update(
            {
                "entry_count": len(self._entries),
                "max_entries": self.max_entries,
                "longest_prefix_tokens": max(
                    (len(key) for key in self._entries),
                    default=0,
                ),
            }
        )
        return data


class DFlashEngine(BatchedEngine):
    """Speculative-decoding engine using a separate drafter model."""

    def __init__(
        self,
        model_name: str,
        drafter_path: str,
        block_size: int | None = None,
        adaptive: bool = True,
        adaptive_min: int = 8,
        adaptive_max: int = 22,
        turboquant_bits: float | None = None,
        ddtree_budget: int = 0,
        ddtree_block_size: int | None = None,
        fallback_mode: str | None = None,
        ngram_num_draft_tokens: int | None = None,
        ngram_size: int | None = None,
        ngram_min_matches: int | None = None,
        ngram_disable_threshold: float | None = None,
        ngram_disable_window: int | None = None,
        ngram_disable_cooldown: int | None = None,
        thinking_ngram_num_draft_tokens: int | None = None,
        thinking_ngram_size: int | None = None,
        thinking_ngram_min_matches: int | None = None,
        agentic_speculative_policy: str = "off",
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.90,
        speculative_prefill_config: SpeculativePrefillConfig | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=False,
            gpu_memory_utilization=gpu_memory_utilization,
            speculative_prefill_config=speculative_prefill_config,
        )
        if self._is_mllm:
            raise ValueError(
                "DFlash mode is text-only; do not pass --mllm or use a multimodal model."
            )

        self._drafter_path = drafter_path
        self._drafter: Any = None
        self._block_size_override = block_size
        self._adaptive_enabled = bool(adaptive)
        self._adaptive_min = int(adaptive_min)
        self._adaptive_max = int(adaptive_max)
        self._turboquant_bits = turboquant_bits
        self._ddtree_budget = max(0, int(ddtree_budget or 0))
        self._ddtree_block_size = ddtree_block_size
        self._ddtree_last: dict[str, Any] = {}
        self._dflash_fallback_mode = fallback_mode
        self._ngram_first_enabled = (
            self._ddtree_budget > 0 and fallback_mode == "ngram"
        )
        self._ngram_num_draft_tokens = max(1, int(ngram_num_draft_tokens or 4))
        self._ngram_size = max(1, int(ngram_size or 3))
        self._ngram_min_matches = max(1, int(ngram_min_matches or 1))
        self._ngram_disable_threshold = float(
            0.55 if ngram_disable_threshold is None else ngram_disable_threshold
        )
        self._ngram_disable_window = max(1, int(ngram_disable_window or 4))
        self._ngram_disable_cooldown = max(0, int(ngram_disable_cooldown or 8))
        self._thinking_ngram_num_draft_tokens = (
            max(1, int(thinking_ngram_num_draft_tokens))
            if thinking_ngram_num_draft_tokens is not None
            else None
        )
        self._thinking_ngram_size = (
            max(1, int(thinking_ngram_size))
            if thinking_ngram_size is not None
            else None
        )
        self._thinking_ngram_min_matches = (
            max(1, int(thinking_ngram_min_matches))
            if thinking_ngram_min_matches is not None
            else None
        )
        self._thinking_ngram_enabled = (
            self._thinking_ngram_num_draft_tokens is not None
        )
        self._agentic_speculative_policy = str(
            agentic_speculative_policy or "off"
        )
        self._agentic_policy_history: deque[dict[str, Any]] = deque(maxlen=8)
        self._agentic_policy_cooldown = 0
        self._last_agentic_policy: dict[str, Any] = {}

        self._lock = asyncio.Lock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._target_stream_generators: set[Any] = set()
        self._active: _ActiveRequest | None = None
        self._inflight = 0

        self._adaptive_cfg: Any = None
        self._current_block_size = 0
        self._observed_block_min = 0
        self._observed_block_max = 0

        self._lifetime_proposed = 0
        self._lifetime_accepted = 0
        self._lifetime_responses = 0
        self._ddtree_responses = 0
        cache_entries = getattr(scheduler_config, "prefix_cache_size", 2) or 2
        cache_entries = int(os.environ.get("DFLASH_DDTREE_PREFIX_CACHE_ENTRIES", cache_entries))
        self._ddtree_prefix_cache = _DDTreePrefixStateCache(max_entries=cache_entries)
        target_cache_entries = int(
            os.environ.get(
                "DFLASH_TARGET_PREFIX_CACHE_ENTRIES",
                max(16, cache_entries),
            )
        )
        self._target_prefix_cache = _DDTreePrefixStateCache(
            max_entries=target_cache_entries
        )
        self._ddtree_capture_cache = _env_bool(
            "DFLASH_DDTREE_CAPTURE_CACHE",
            self._ddtree_budget > 0,
        )
        self._target_prefix_cache_enabled = _env_bool(
            "DFLASH_TARGET_PREFIX_CACHE",
            self._ddtree_budget > 0,
        )
        self._target_prefix_cache_stride = max(
            0,
            int(os.environ.get("DFLASH_TARGET_PREFIX_CACHE_STRIDE", "1024")),
        )
        self._last_memory_stats = (0.0, 0.0, 0.0)
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _start_llm(self) -> None:  # noqa: D401 — overrides BatchedEngine
        # Create the dedicated MLX worker thread BEFORE loading the model.
        # All model + drafter operations (including the initial weight load)
        # must run on this thread so that mlx-lm's per-thread generation
        # stream owns the arrays used during inference (PR 161 invariant).
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dflash-step",
            initializer=_init_dflash_step_thread,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_models_in_thread)
        # mark as not-engine-started (we don't run AsyncEngineCore)
        self._engine_started = False

    def _load_models_in_thread(self) -> None:
        """Runs on the dflash-step worker thread."""
        try:
            from dflash.model_mlx import (
                AdaptiveBlockSizeConfig,
                load_draft,
            )
        except ImportError as exc:
            raise RuntimeError(
                "DFlash mode requires the `dflash[mlx]` package. Install with:\n"
                "  pip install -e /Users/samuelfajreldines/dev/dflash[mlx]\n"
                f"Original error: {exc}"
            ) from exc

        from ..utils.tokenizer import load_model_with_fallback

        logger.info("[DFlash] Loading target model: %s", self._model_name)
        tokenizer_config = {"trust_remote_code": self._trust_remote_code}
        if "qwen3" in self._model_name.lower():
            tokenizer_config["eos_token"] = "<|im_end|>"
        self._model, self._tokenizer = load_model_with_fallback(
            self._model_name,
            tokenizer_config=tokenizer_config,
        )

        logger.info("[DFlash] Loading drafter: %s", self._drafter_path)
        self._drafter = load_draft(
            self._drafter_path,
            turboquant_bits=self._turboquant_bits,
        )
        self._drafter.bind(self._model)

        self._current_block_size = (
            int(self._block_size_override)
            if self._block_size_override is not None
            else int(self._drafter.config.block_size)
        )

        if self._adaptive_enabled:
            self._adaptive_cfg = AdaptiveBlockSizeConfig(
                enabled=True,
                min_block_size=self._adaptive_min,
                max_block_size=self._adaptive_max,
                grow_threshold=0.88,
                shrink_threshold=0.55,
                grow_streak=2,
                shrink_streak=2,
            )
        else:
            self._adaptive_cfg = None

        try:
            import mlx.core as mx

            if mx.metal.is_available():
                info = mx.device_info()
                max_rec = info.get(
                    "max_recommended_working_set_size",
                    info.get("memory_size", 0),
                )
                if max_rec > 0:
                    soft = int(max_rec * self._gpu_memory_utilization)
                    mx.set_memory_limit(soft)
                    mx.set_cache_limit(32 * 1024 * 1024 * 1024)
        except Exception as exc:
            logger.warning("[DFlash] Failed to set Metal memory limits: %s", exc)

    async def stop(self) -> None:
        if self._executor is not None:
            await self._close_target_stream_generators()
            self._executor.shutdown(wait=False)
            self._executor = None
        self._drafter = None

    async def _close_target_stream_generators(self) -> None:
        executor = self._executor
        if executor is None or not self._target_stream_generators:
            self._target_stream_generators.clear()
            return

        generators = tuple(self._target_stream_generators)
        loop = asyncio.get_running_loop()

        def _close_all() -> None:
            for gen in generators:
                close = getattr(gen, "close", None)
                if close is not None:
                    close()

        await loop.run_in_executor(executor, _close_all)
        for gen in generators:
            self._target_stream_generators.discard(gen)

    # ------------------------------------------------------------------
    # Generation core
    # ------------------------------------------------------------------

    async def _stream_dflash(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Run dflash.stream_generate on the dflash-step thread, yielding GenerationResponse objects."""
        from dflash.model_mlx import stream_generate as _ds

        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "DFlashEngine not started"

        def _make_gen():
            return _ds(
                self._model,
                self._drafter,
                self._tokenizer,
                prompt,
                block_size=self._block_size_override,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                adaptive_block_size=self._adaptive_cfg,
            )

        gen = await loop.run_in_executor(executor, _make_gen)

        sentinel = object()

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        while True:
            resp = await loop.run_in_executor(executor, _next)
            if resp is sentinel:
                return
            yield resp

    def _effective_ddtree_block_size(self) -> int | None:
        ddtree_block_size = (
            self._ddtree_block_size
            if self._ddtree_block_size is not None
            else self._block_size_override
        )
        if (
            ddtree_block_size is not None
            and self._ddtree_budget > 0
            and int(ddtree_block_size) <= self._ddtree_budget
        ):
            logger.warning(
                "[DDTree] Ignoring block_size=%s because it is <= tree_budget=%s; using drafter default.",
                ddtree_block_size,
                self._ddtree_budget,
            )
            return None
        return ddtree_block_size

    def _prompt_token_count(self, prompt: str) -> int:
        try:
            encoded = self._tokenizer.encode(prompt)
            return int(len(encoded))
        except Exception:
            return 0

    def _target_cached_token_count(self, prompt: str) -> int:
        if not self._target_prefix_cache_enabled:
            return 0
        try:
            encoded = self._tokenizer.encode(prompt)
            return self._target_prefix_cache.peek_cached_tokens(list(encoded))
        except Exception:
            return 0

    def _should_use_target_prefix_cache(
        self,
        temperature: float | None,
        top_p: float | None,
    ) -> bool:
        return self._target_prefix_cache_enabled and _is_greedy_sampling(
            temperature,
            top_p,
        )

    def _should_use_agentic_target_fallback(
        self,
        prompt: str,
        tools_requested: bool,
    ) -> bool:
        if not tools_requested:
            return False
        if not _env_bool("DFLASH_AGENTIC_TARGET_FALLBACK", True):
            return False
        threshold = max(
            1,
            int(os.environ.get("DFLASH_AGENTIC_TARGET_FALLBACK_MIN_PROMPT_TOKENS", "4096")),
        )
        return self._prompt_token_count(prompt) >= threshold

    def _recent_agentic_acceptance(self) -> float | None:
        ratios = [
            float(item["acceptance_ratio"])
            for item in self._agentic_policy_history
            if item.get("proposed_tokens", 0) > 0
            and item.get("acceptance_ratio") is not None
        ]
        if not ratios:
            return None
        return sum(ratios[-3:]) / len(ratios[-3:])

    def _agentic_policy_decision(
        self,
        prompt: str,
        *,
        tools_requested: bool,
        max_tokens: int,
        greedy_request: bool,
        phase: str | None,
        cached_tokens: int | None = None,
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Return forced mode for auto policy, or None for existing DFlash logic."""

        policy = self._agentic_speculative_policy
        if policy != "auto" or not tools_requested:
            return None, "policy_off_or_no_tools", {}

        prompt_tokens = self._prompt_token_count(prompt)
        cached_tokens = max(0, min(int(cached_tokens or 0), prompt_tokens))
        remaining_prefill_tokens = prompt_tokens - cached_tokens
        recent_acceptance = self._recent_agentic_acceptance()
        phase = phase or "tool_json"
        metadata: dict[str, Any] = {
            "policy": policy,
            "phase": phase,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_ratio": (
                cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
            ),
            "remaining_prefill_tokens": remaining_prefill_tokens,
            "recent_acceptance": recent_acceptance,
            "cooldown": self._agentic_policy_cooldown,
        }

        if phase in {"repair", "validation", "finalization", "tool_json"}:
            return "target-fallback", f"phase_{phase}", metadata
        prefill_limit = int(os.environ.get("DFLASH_AGENTIC_POLICY_MAX_PREFILL", "8000"))
        ddtree_prompt_limit = int(
            os.environ.get("DFLASH_AGENTIC_DDTREE_MAX_PROMPT_TOKENS", "16384")
        )
        ddtree_max_tokens_limit = int(
            os.environ.get("DFLASH_AGENTIC_DDTREE_MAX_TOKENS", "1024")
        )
        prefill_too_large = remaining_prefill_tokens > prefill_limit
        if self._agentic_policy_cooldown > 0:
            self._agentic_policy_cooldown -= 1
            metadata["cooldown"] = self._agentic_policy_cooldown
            return "target-fallback", "cooldown", metadata
        if recent_acceptance is not None and recent_acceptance < float(
            os.environ.get("DFLASH_AGENTIC_POLICY_MIN_ACCEPTANCE", "0.35")
        ):
            self._agentic_policy_cooldown = int(
                os.environ.get("DFLASH_AGENTIC_POLICY_COOLDOWN", "3")
            )
            metadata["cooldown"] = self._agentic_policy_cooldown
            return "target-fallback", "low_acceptance", metadata
        if prompt_tokens > ddtree_prompt_limit:
            metadata["ddtree_prompt_limit"] = ddtree_prompt_limit
            return "target-fallback", "prompt_outside_ddtree_sweet_spot", metadata
        if max_tokens > ddtree_max_tokens_limit:
            metadata["ddtree_max_tokens_limit"] = ddtree_max_tokens_limit
            return "target-fallback", "max_tokens_outside_ddtree_sweet_spot", metadata
        if (
            max_tokens > 512
            and greedy_request
            and self._ddtree_budget > 0
            and phase in {"initial_scaffold", "long_text_or_code"}
        ):
            if (
                self._should_enable_ngram_first(prompt, phase)
                or self._thinking_ngram_enabled
            ):
                reason = (
                    "long_prefill_suffix_speculative_prefill_ngram"
                    if prefill_too_large
                    and self._speculative_prefill.config.enabled
                    else "long_generation_ngram"
                )
                if prefill_too_large and not self._speculative_prefill.config.enabled:
                    return "target-fallback", "prefill_too_large", metadata
                return "ddtree-ngram", reason, metadata
            if prefill_too_large and not self._speculative_prefill.config.enabled:
                return "target-fallback", "prefill_too_large", metadata
            return (
                "ddtree",
                "long_prefill_suffix_speculative_prefill"
                if prefill_too_large
                else "long_generation_ddtree",
                metadata,
            )
        return "target-fallback", "short_or_non_greedy", metadata

    def _ddtree_cached_token_count(self, prompt: str) -> int:
        if not self._ddtree_capture_cache:
            return 0
        try:
            from ..speculative.ddtree.engine import tokenize_prompt

            prompt_array = tokenize_prompt(self._tokenizer, prompt)
            prompt_token_ids = (
                prompt_array.tolist()
                if hasattr(prompt_array, "tolist")
                else list(prompt_array)
            )
        except Exception:
            return 0
        return self._ddtree_prefix_cache.peek_cached_tokens(prompt_token_ids)

    async def _stream_target(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        tools_requested: bool,
        logits_processor_factories: list[Any] | None = None,
    ):
        """Run target-only mlx-lm decode on the DFlash worker thread."""
        from mlx_lm import stream_generate as _target_stream_generate
        from mlx_lm.sample_utils import make_sampler

        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "DFlashEngine not started"

        def _make_gen():
            processors = []
            for factory in logits_processor_factories or ():
                processor = factory()
                if processor is not None:
                    processors.append(processor)
            sampler = make_sampler(temp=float(temperature or 0.0), top_p=float(top_p or 0.0))
            return _target_stream_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=processors or None,
            )

        gen = await loop.run_in_executor(executor, _make_gen)
        self._target_stream_generators.add(gen)
        sentinel = object()
        stop_after_strings = list(_TOOL_STOP_AFTER_STRINGS) if tools_requested else []
        stop_strings = [*(stop or ()), *stop_after_strings]
        cumulative = ""
        stopped = False

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        exhausted = False

        def _close_gen():
            close = getattr(gen, "close", None)
            if close is not None:
                close()

        try:
            while True:
                resp = await loop.run_in_executor(executor, _next)
                if resp is sentinel:
                    exhausted = True
                    return
                text = getattr(resp, "text", "") or ""
                cumulative += text
                finish_reason = getattr(resp, "finish_reason", None)
                if stop_strings and any(marker in cumulative for marker in stop_strings):
                    stopped = True
                    finish_reason = "stop"
                yield SimpleNamespace(
                    text=text,
                    tokens=[int(getattr(resp, "token", 0))]
                    if getattr(resp, "token", None) is not None
                    else [],
                    prompt_tokens=int(getattr(resp, "prompt_tokens", 0) or 0),
                    prefill_seconds=(
                        float(getattr(resp, "prompt_tokens", 0) or 0)
                        / float(getattr(resp, "prompt_tps", 0.0) or 1.0)
                        if getattr(resp, "prompt_tps", 0.0)
                        else 0.0
                    ),
                    generation_tokens=int(getattr(resp, "generation_tokens", 0) or 0),
                    generation_tps=float(getattr(resp, "generation_tps", 0.0) or 0.0),
                    finish_reason=finish_reason,
                    proposed_tokens=0,
                    accepted_tokens=0,
                    speculative_steps=0,
                    avg_acceptance_ratio=0.0,
                    block_size_history=(),
                    avg_tree_node_count=0.0,
                    ddtree_fast_path_ratio=0.0,
                    tree_budget=0,
                    ngram_acceptance_ratio=0.0,
                    ngram_cycles=0,
                    ngram_fallback_cycles=0,
                    ngram_tool_guard_cycles=0,
                    cache_hit_type="agentic-target-fallback",
                    cached_tokens=0,
                    phase_timings_us={},
                )
                if stopped:
                    return
        finally:
            if not exhausted:
                if self._executor is not None:
                    try:
                        await loop.run_in_executor(executor, _close_gen)
                    except RuntimeError as exc:
                        logger.warning(
                            "[DFlash] target stream close skipped after executor "
                            "shutdown: %s",
                            exc,
                        )
            self._target_stream_generators.discard(gen)

    def _stream_target_cached_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools_requested: bool,
        logits_processors: list[Any] | None = None,
        prefix_boundaries: list[int] | None = None,
    ):
        import importlib

        import mlx.core as mx

        generate_mod = importlib.import_module("mlx_lm.generate")
        generate_mod.generation_stream = mx.new_stream(mx.default_device())
        from mlx_lm.models import cache as mlx_cache
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        tokenizer = self._tokenizer
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)
        prompt_tokens = list(self._tokenizer.encode(prompt))
        if not prompt_tokens:
            return

        fetch = self._target_prefix_cache.fetch(prompt_tokens)
        cached_tokens = int(fetch.cached_tokens or 0)
        state = fetch.state if isinstance(fetch.state, _TargetPrefixState) else None
        suffix_tokens = prompt_tokens[cached_tokens:]
        if state is None:
            cached_tokens = 0
            suffix_tokens = prompt_tokens
        elif not suffix_tokens and (state.logprobs is None or tools_requested):
            state = None
            cached_tokens = 0
            suffix_tokens = prompt_tokens
        prompt_cache = (
            copy.deepcopy(state.prompt_cache)
            if state is not None
            else mlx_cache.make_prompt_cache(self._model)
        )
        sampler = make_sampler(temp=float(temperature or 0.0), top_p=float(top_p or 0.0))
        detokenizer = tokenizer.detokenizer
        token_context = mx.array(prompt_tokens[:cached_tokens], dtype=mx.uint32)

        def _apply_processors(input_tokens: mx.array, logits: mx.array) -> mx.array:
            nonlocal token_context
            if logits_processors and len(input_tokens) > 0:
                token_context = (
                    mx.concat([token_context, input_tokens])
                    if token_context.size
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(token_context, logits)
            return logits

        def _model_call(input_tokens: mx.array):
            return self._model(input_tokens[None], cache=prompt_cache)

        def _step(input_tokens: mx.array):
            with mx.stream(generate_mod.generation_stream):
                logits = _model_call(input_tokens)[:, -1, :]
                logits = _apply_processors(input_tokens, logits)
                logprobs_2d = logits - mx.logsumexp(logits, keepdims=True)
                sampled = sampler(logprobs_2d)
                return sampled, logprobs_2d

        prompt_started = time.perf_counter()
        with generate_mod.wired_limit(self._model, [generate_mod.generation_stream]):
            with mx.stream(generate_mod.generation_stream):
                if state is not None and not suffix_tokens:
                    logprobs_2d = copy.deepcopy(state.logprobs)
                    y = sampler(logprobs_2d)
                else:
                    suffix = mx.array(suffix_tokens or prompt_tokens, dtype=mx.uint32)
                    boundary_set = {
                        int(boundary) - cached_tokens
                        for boundary in (prefix_boundaries or [])
                        if cached_tokens < int(boundary) < len(prompt_tokens)
                    }
                    stored_boundaries = 0
                    processed_suffix_tokens = 0
                    next_checkpoint = (
                        self._target_prefix_cache_stride
                        if self._target_prefix_cache_stride > 0
                        else 0
                    )
                    while len(suffix) > 1:
                        remaining_prefill = len(suffix) - 1
                        next_boundary = min(
                            (
                                boundary
                                for boundary in boundary_set
                                if boundary > processed_suffix_tokens
                            ),
                            default=None,
                        )
                        n_to_process = min(2048, remaining_prefill)
                        if next_boundary is not None:
                            n_to_process = min(
                                n_to_process,
                                max(1, next_boundary - processed_suffix_tokens),
                            )
                        _model_call(suffix[:n_to_process])
                        mx.eval([c.state for c in prompt_cache])
                        token_context = (
                            mx.concat([token_context, suffix[:n_to_process]])
                            if token_context.size
                            else suffix[:n_to_process]
                        )
                        processed_suffix_tokens += n_to_process
                        absolute_processed = cached_tokens + processed_suffix_tokens
                        if absolute_processed in (prefix_boundaries or ()):
                            self._target_prefix_cache.store(
                                prompt_tokens[:absolute_processed],
                                _TargetPrefixState(
                                    prompt_tokens=tuple(
                                        prompt_tokens[:absolute_processed]
                                    ),
                                    prompt_cache=copy.deepcopy(prompt_cache),
                                    logprobs=None,
                                ),
                            )
                            stored_boundaries += 1
                        if (
                            next_checkpoint
                            and absolute_processed >= next_checkpoint
                            and absolute_processed < len(prompt_tokens)
                        ):
                            self._target_prefix_cache.store(
                                prompt_tokens[:absolute_processed],
                                _TargetPrefixState(
                                    prompt_tokens=tuple(
                                        prompt_tokens[:absolute_processed]
                                    ),
                                    prompt_cache=copy.deepcopy(prompt_cache),
                                    logprobs=None,
                                ),
                            )
                            stored_boundaries += 1
                            next_checkpoint = (
                                absolute_processed
                                + self._target_prefix_cache_stride
                            )
                        suffix = suffix[n_to_process:]
                        mx.clear_cache()
                    y, logprobs_2d = _step(suffix)
                    can_trim_cache = mlx_cache.can_trim_prompt_cache(prompt_cache)
                    full_state: _TargetPrefixState | None = None
                    if tools_requested or logits_processors:
                        if can_trim_cache and len(prompt_tokens) > 1:
                            first_token_cache = copy.deepcopy(prompt_cache)
                            mlx_cache.trim_prompt_cache(first_token_cache, 1)
                            self._target_prefix_cache.store(
                                prompt_tokens[:-1],
                                _TargetPrefixState(
                                    prompt_tokens=tuple(prompt_tokens[:-1]),
                                    prompt_cache=first_token_cache,
                                    logprobs=None,
                                ),
                            )
                    else:
                        full_state = _TargetPrefixState(
                            prompt_tokens=tuple(prompt_tokens),
                            prompt_cache=copy.deepcopy(prompt_cache),
                            logprobs=copy.deepcopy(logprobs_2d),
                        )
                        self._target_prefix_cache.store(prompt_tokens, full_state)
                    if prefix_boundaries and can_trim_cache:
                        boundary_source_cache = (
                            full_state.prompt_cache
                            if full_state is not None
                            else prompt_cache
                        )
                        for boundary in prefix_boundaries:
                            boundary = int(boundary)
                            if boundary <= 0 or boundary >= len(prompt_tokens):
                                continue
                            boundary_cache = copy.deepcopy(boundary_source_cache)
                            mlx_cache.trim_prompt_cache(
                                boundary_cache,
                                len(prompt_tokens) - boundary,
                            )
                            self._target_prefix_cache.store(
                                prompt_tokens[:boundary],
                                _TargetPrefixState(
                                    prompt_tokens=tuple(prompt_tokens[:boundary]),
                                    prompt_cache=boundary_cache,
                                    logprobs=None,
                                ),
                            )
                            stored_boundaries += 1
                    logger.info(
                        "[target-prefix-cache] store prompt_tokens=%d boundaries=%d",
                        len(prompt_tokens),
                        stored_boundaries,
                    )
                mx.eval(y, logprobs_2d)

            prefill_seconds = max(time.perf_counter() - prompt_started, 1e-9)
            decode_started = time.perf_counter()
            generated = 0
            eos_token_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
            while generated < max_tokens:
                token = int(y.item() if hasattr(y, "item") else y)
                if token in eos_token_ids:
                    break
                detokenizer.add_token(token)
                generated += 1
                yield SimpleNamespace(
                    text=detokenizer.last_segment,
                    tokens=[token],
                    prompt_tokens=len(prompt_tokens),
                    prefill_seconds=prefill_seconds,
                    generation_tokens=generated,
                    generation_tps=generated
                    / max(time.perf_counter() - decode_started, 1e-9),
                    finish_reason=None,
                    proposed_tokens=0,
                    accepted_tokens=0,
                    speculative_steps=0,
                    avg_acceptance_ratio=0.0,
                    block_size_history=(),
                    avg_tree_node_count=0.0,
                    ddtree_fast_path_ratio=0.0,
                    tree_budget=0,
                    ngram_acceptance_ratio=0.0,
                    ngram_cycles=0,
                    ngram_fallback_cycles=0,
                    ngram_tool_guard_cycles=0,
                    cache_hit_type=fetch.hit_type,
                    cached_tokens=cached_tokens,
                    phase_timings_us={},
                )
                y, logprobs_2d = _step(mx.array([token], dtype=mx.uint32))
                mx.async_eval(y, logprobs_2d)
                if generated % 256 == 0:
                    mx.clear_cache()

            detokenizer.finalize()
            yield SimpleNamespace(
                text=detokenizer.last_segment,
                tokens=[],
                prompt_tokens=len(prompt_tokens),
                prefill_seconds=prefill_seconds,
                generation_tokens=generated,
                generation_tps=generated
                / max(time.perf_counter() - decode_started, 1e-9),
                finish_reason="stop",
                proposed_tokens=0,
                accepted_tokens=0,
                speculative_steps=0,
                avg_acceptance_ratio=0.0,
                block_size_history=(),
                avg_tree_node_count=0.0,
                ddtree_fast_path_ratio=0.0,
                tree_budget=0,
                ngram_acceptance_ratio=0.0,
                ngram_cycles=0,
                ngram_fallback_cycles=0,
                ngram_tool_guard_cycles=0,
                cache_hit_type=fetch.hit_type,
                cached_tokens=cached_tokens,
                phase_timings_us={},
            )

    async def _stream_target_cached(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        tools_requested: bool,
        logits_processor_factories: list[Any] | None = None,
        prefix_boundaries: list[int] | None = None,
    ):
        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "DFlashEngine not started"

        def _make_processors():
            processors = []
            for factory in logits_processor_factories or ():
                processor = factory()
                if processor is not None:
                    processors.append(processor)
            return processors

        processors = await loop.run_in_executor(executor, _make_processors)
        gen = await loop.run_in_executor(
            executor,
            lambda: self._stream_target_cached_sync(
                prompt,
                max_tokens,
                temperature,
                top_p,
                tools_requested,
                processors,
                prefix_boundaries,
            ),
        )
        sentinel = object()
        stop_after_strings = list(_TOOL_STOP_AFTER_STRINGS) if tools_requested else []
        stop_strings = [*(stop or ()), *stop_after_strings]
        cumulative = ""

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        def _close_gen():
            close = getattr(gen, "close", None)
            if close is not None:
                close()

        try:
            while True:
                resp = await loop.run_in_executor(executor, _next)
                if resp is sentinel:
                    return
                cumulative += resp.text or ""
                finish_reason = resp.finish_reason
                if stop_strings and any(marker in cumulative for marker in stop_strings):
                    finish_reason = "stop"
                if resp.cached_tokens:
                    logger.info(
                        "[target-prefix-cache] cache_fetch HIT cached_tokens=%d prompt_tokens=%d",
                        resp.cached_tokens,
                        resp.prompt_tokens,
                    )
                yield SimpleNamespace(
                    **{**resp.__dict__, "finish_reason": finish_reason}
                )
                if finish_reason:
                    return
        finally:
            try:
                await loop.run_in_executor(executor, _close_gen)
            except RuntimeError as exc:
                logger.warning("[target-prefix-cache] close skipped: %s", exc)

    def _store_ddtree_cache_result(
        self,
        result: dict[str, Any],
        prompt_token_ids: list[int],
        prefix_boundary: int,
    ) -> None:
        if not self._ddtree_capture_cache:
            return
        if result.get("prompt_cache_state") is not None:
            self._ddtree_prefix_cache.store(
                list(result.get("prompt_token_ids") or ()),
                result.get("prompt_cache_state"),
            )
        if result.get("extended_prompt_cache_state") is not None:
            self._ddtree_prefix_cache.store(
                [
                    *list(result.get("prompt_token_ids") or ()),
                    *list(result.get("generated_token_ids") or ()),
                ],
                result.get("extended_prompt_cache_state"),
            )
        if (
            prefix_boundary > 0
            and prefix_boundary < len(prompt_token_ids)
            and result.get("prefix_boundary_state") is not None
        ):
            self._ddtree_prefix_cache.store(
                prompt_token_ids[:prefix_boundary],
                result.get("prefix_boundary_state"),
            )

    def _should_enable_ngram_first(
        self, prompt: str, phase: str | None = None
    ) -> bool:
        agentic_long_text_ngram = (
            self._agentic_speculative_policy == "auto"
            and phase == "long_text_or_code"
            and self._ddtree_budget > 0
            and _env_bool("DFLASH_AGENTIC_NGRAM_LONG_TEXT", True)
        )
        if not (self._ngram_first_enabled or agentic_long_text_ngram):
            return False
        if agentic_long_text_ngram or _env_bool("DFLASH_NGRAM_FORCE", False):
            return True
        lowered = prompt.lower()
        structured_markers = (
            "json",
            "xml",
            "crud",
            "schema",
            "typescript type",
            "interface ",
            "tests",
            "boilerplate",
        )
        if any(marker in lowered for marker in structured_markers):
            return True
        words = [word for word in lowered.replace("\n", " ").split(" ") if word]
        if len(words) < 16:
            return False
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio <= 0.55

    def _maybe_compress_ddtree_suffix(
        self,
        prompt_token_ids: list[int],
        cached_tokens: int,
        phase: str | None,
    ) -> list[int]:
        if not self._speculative_prefill.config.enabled:
            return prompt_token_ids
        if phase in {"tool_json", "repair", "validation", "finalization"}:
            self._speculative_prefill.last_result = None
            return prompt_token_ids
        cached_tokens = max(0, min(int(cached_tokens or 0), len(prompt_token_ids)))
        suffix_tokens = prompt_token_ids[cached_tokens:]
        if not suffix_tokens:
            self._speculative_prefill.last_result = None
            return prompt_token_ids
        try:
            suffix_text = self._tokenizer.decode(suffix_tokens)
        except Exception:
            self._speculative_prefill.last_result = None
            return prompt_token_ids
        result = self._speculative_prefill.compress(suffix_text, self._tokenizer)
        if not result.applied:
            return prompt_token_ids
        compressed_suffix = list(self._tokenizer.encode(result.prompt))
        if len(compressed_suffix) >= len(suffix_tokens):
            return prompt_token_ids
        logger.info(
            "[speculative-prefill] ddtree suffix compressed cached=%d suffix=%d->%d",
            cached_tokens,
            len(suffix_tokens),
            len(compressed_suffix),
        )
        return [*prompt_token_ids[:cached_tokens], *compressed_suffix]

    def _run_ddtree_sync(
        self,
        prompt: str,
        max_tokens: int,
        stop: list[str] | None,
        tools_requested: bool,
        prefix_boundary: int = 0,
        on_step: Any = None,
        should_stop: Any = None,
        emit_step_text: bool = False,
        logits_processors: list[Any] | None = None,
        agentic_phase: str | None = None,
    ) -> dict[str, Any]:
        from ..speculative.ddtree.engine import generate_ddtree, tokenize_prompt

        prompt_array = tokenize_prompt(self._tokenizer, prompt)
        ngram_first_enabled = self._should_enable_ngram_first(
            prompt,
            agentic_phase,
        )
        prompt_token_ids = []
        if self._ddtree_capture_cache:
            prompt_token_ids = (
                prompt_array.tolist()
                if hasattr(prompt_array, "tolist")
                else list(prompt_array)
            )
        cache_fetch = (
            self._ddtree_prefix_cache.fetch(prompt_token_ids)
            if self._ddtree_capture_cache
            else SimpleNamespace(state=None, hit_type=None, cached_tokens=0)
        )
        logger.info(
            "[DDTree] cache_fetch %s cached_tokens=%d prompt_tokens=%d",
            "HIT" if getattr(cache_fetch, "cached_tokens", 0) else "MISS",
            int(getattr(cache_fetch, "cached_tokens", 0) or 0),
            len(prompt_token_ids),
        )
        if prompt_token_ids:
            prompt_token_ids = self._maybe_compress_ddtree_suffix(
                prompt_token_ids,
                int(getattr(cache_fetch, "cached_tokens", 0) or 0),
                agentic_phase,
            )
            prompt_array = mx.array(prompt_token_ids)
            if prefix_boundary > int(getattr(cache_fetch, "cached_tokens", 0) or 0):
                prefix_boundary = 0

        result = generate_ddtree(
            target_model=self._model,
            draft_model=self._drafter,
            tokenizer=self._tokenizer,
            prompt_tokens=prompt_array,
            max_new_tokens=max_tokens,
            tree_budget=self._ddtree_budget,
            block_size=self._effective_ddtree_block_size(),
            adaptive_block_size=self._adaptive_cfg,
            prefix_state=cache_fetch.state,
            capture_prefill_state=self._ddtree_capture_cache,
            target_turboquant_bits=self._turboquant_bits,
            stop_strings=stop,
            stop_after_strings=(
                list(_TOOL_STOP_AFTER_STRINGS) if tools_requested else None
            ),
            prefix_boundary=prefix_boundary,
            ngram_num_draft_tokens=(
                self._ngram_num_draft_tokens if ngram_first_enabled else None
            ),
            ngram_size=self._ngram_size if ngram_first_enabled else None,
            ngram_min_matches=(
                self._ngram_min_matches if ngram_first_enabled else None
            ),
            ngram_disable_threshold=(
                self._ngram_disable_threshold if ngram_first_enabled else None
            ),
            ngram_disable_window=(
                self._ngram_disable_window if ngram_first_enabled else None
            ),
            ngram_disable_cooldown=(
                self._ngram_disable_cooldown if ngram_first_enabled else None
            ),
            thinking_ngram_num_draft_tokens=self._thinking_ngram_num_draft_tokens,
            thinking_ngram_size=self._thinking_ngram_size,
            thinking_ngram_min_matches=self._thinking_ngram_min_matches,
            logits_processors=logits_processors,
            should_stop=should_stop,
            on_step=on_step,
            emit_step_text=emit_step_text,
        )
        self._store_ddtree_cache_result(result, prompt_token_ids, prefix_boundary)
        self._ddtree_last = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "prompt_cache_state",
                "prompt_token_ids",
                "prefix_boundary_state",
                "extended_prompt_cache_state",
            }
        }
        result["cache_hit_type"] = cache_fetch.hit_type
        result["cached_tokens"] = cache_fetch.cached_tokens
        return result

    async def _stream_ddtree(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        tools_requested: bool,
        prefix_boundary: int = 0,
        logits_processor_factories: list[Any] | None = None,
        agentic_phase: str | None = None,
    ):
        """Run the Rapid-MLX DDTree loop on the DFlash MLX worker thread."""
        if not _is_greedy_sampling(temperature, top_p):
            logger.debug(
                "[DDTree] greedy DDTree path ignores sampler settings: temperature=%s top_p=%s",
                temperature,
                top_p,
            )

        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "DFlashEngine not started"
        stop_event = threading.Event()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _make_processors() -> list[Any]:
            processors = []
            for factory in logits_processor_factories or ():
                processor = factory()
                if processor is not None:
                    if hasattr(processor, "prompt_token_count"):
                        processor.prompt_token_count = 0
                    processors.append(processor)
            return processors

        def _run():
            logits_processors = _make_processors()

            def _on_step(step: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, step)

            try:
                result = self._run_ddtree_sync(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    stop=stop,
                    tools_requested=tools_requested,
                    prefix_boundary=prefix_boundary,
                    on_step=_on_step,
                    should_stop=stop_event.is_set,
                    emit_step_text=True,
                    logits_processors=logits_processors,
                    agentic_phase=agentic_phase,
                )
                loop.call_soon_threadsafe(queue.put_nowait, result)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        future = loop.run_in_executor(executor, _run)
        saw_step = False
        try:
            while True:
                try:
                    if _DDTREE_STREAM_IDLE_TIMEOUT_SECONDS > 0:
                        item = await asyncio.wait_for(
                            queue.get(),
                            timeout=_DDTREE_STREAM_IDLE_TIMEOUT_SECONDS,
                        )
                    else:
                        item = await queue.get()
                except TimeoutError:
                    if future.done():
                        continue
                    stop_event.set()
                    logger.warning(
                        "[DDTree] no stream events for %.1fs; aborting request",
                        _DDTREE_STREAM_IDLE_TIMEOUT_SECONDS,
                    )
                    break
                if item is sentinel:
                    break
                is_final = "finish_reason" in item
                item_text = "" if is_final and saw_step else item.get("text", "")
                saw_step = saw_step or not is_final
                yield SimpleNamespace(
                    text=item_text,
                    tokens=list(item.get("generated_token_ids", []) or []),
                    prompt_tokens=int(item.get("prompt_tokens") or 0),
                    prefill_seconds=float(item.get("prefill_seconds") or 0.0),
                    generation_tokens=int(item.get("generated_tokens") or 0),
                    generation_tps=float(item.get("generation_tps") or 0.0),
                    finish_reason=item.get("finish_reason") if is_final else None,
                    proposed_tokens=int(item.get("proposed_tokens") or 0),
                    accepted_tokens=int(item.get("accepted_tokens") or 0),
                    speculative_steps=int(item.get("speculative_steps") or 0),
                    avg_acceptance_ratio=float(
                        item.get("avg_acceptance_ratio") or 0.0
                    ),
                    block_size_history=tuple(item.get("block_size_history") or ()),
                    avg_tree_node_count=float(
                        item.get("avg_tree_node_count") or 0.0
                    ),
                    ddtree_fast_path_ratio=float(
                        item.get("ddtree_fast_path_ratio") or 0.0
                    ),
                    tree_budget=int(item.get("tree_budget") or self._ddtree_budget),
                    ngram_acceptance_ratio=float(
                        item.get("ngram_acceptance_ratio") or 0.0
                    ),
                    ngram_cycles=int(item.get("ngram_cycles_completed") or 0),
                    ngram_fallback_cycles=int(
                        item.get("ngram_fallback_cycles") or 0
                    ),
                    ngram_tool_guard_cycles=int(
                        item.get("ngram_tool_guard_cycles") or 0
                    ),
                    cache_hit_type=item.get("cache_hit_type"),
                    cached_tokens=int(item.get("cached_tokens") or 0),
                    phase_timings_us=dict(item.get("ddtree_phase_timings_us") or {}),
                )
            if future.done():
                await future
            else:
                stop_event.set()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=_DDTREE_STOP_WAIT_SECONDS,
                    )
                except TimeoutError:
                    logger.warning(
                        "[DDTree] worker did not stop within %.1fs after abort; "
                        "releasing stream request",
                        _DDTREE_STOP_WAIT_SECONDS,
                    )
                except Exception:
                    logger.exception("[DDTree] worker failed after abort")
        except BaseException:
            stop_event.set()
            future.cancel()
            raise

    def _make_ddtree_logits_processors(
        self, logits_processor_factories: list[Any] | None
    ) -> list[Any]:
        processors = []
        for factory in logits_processor_factories or ():
            processor = factory()
            if processor is None:
                continue
            if hasattr(processor, "prompt_token_count"):
                processor.prompt_token_count = 0
            processors.append(processor)
        return processors

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        if images or videos:
            raise ValueError("DFlash mode does not support images or videos.")
        prompt = self._maybe_compress_prompt(prompt, kwargs)

        greedy_request = _is_greedy_sampling(temperature, top_p)
        if self._ddtree_budget > 0 and greedy_request:
            self._inflight += 1
            try:
                async with self._lock:
                    tools_requested = bool(kwargs.pop("tools_requested", False))
                    prefix_boundary = int(kwargs.pop("prefix_boundary", 0) or 0)
                    prefix_boundaries = kwargs.pop("prefix_boundaries", None)
                    agentic_phase = kwargs.pop("agentic_phase", None)
                    kwargs.pop("agentic_speculative_policy", None)
                    kwargs.pop("tool_call_expected", None)
                    logits_processor_factories = kwargs.pop(
                        "logits_processor_factories", None
                    )
                    forced_mode, policy_reason, policy_metadata = (
                        self._agentic_policy_decision(
                            prompt,
                            tools_requested=tools_requested,
                            max_tokens=max_tokens,
                            greedy_request=greedy_request,
                            phase=agentic_phase,
                            cached_tokens=self._ddtree_cached_token_count(prompt),
                        )
                    )
                    self._last_agentic_policy = policy_metadata | {
                        "decision": forced_mode,
                        "reason": policy_reason,
                    }
                    if forced_mode == "target-fallback" or (
                        forced_mode is None
                        and self._should_use_agentic_target_fallback(
                            prompt, tools_requested
                        )
                    ):
                        use_prefix_cache = self._should_use_target_prefix_cache(
                            temperature,
                            top_p,
                        )
                        self._track_request_start(
                            "target-prefix-cache"
                            if use_prefix_cache
                            else "target-fallback"
                        )
                        if self._active is not None:
                            self._active.agentic_phase = agentic_phase
                            self._active.agentic_policy_decision = (
                                "target-prefix-cache"
                                if use_prefix_cache
                                else "target-fallback"
                            )
                            self._active.agentic_policy_reason = policy_reason
                        try:
                            last_resp = None
                            text_parts: list[str] = []
                            if use_prefix_cache:
                                response_stream = self._stream_target_cached(
                                    prompt,
                                    max_tokens,
                                    temperature,
                                    top_p,
                                    stop,
                                    tools_requested,
                                    logits_processor_factories,
                                    prefix_boundaries,
                                )
                            else:
                                response_stream = self._stream_target(
                                    prompt,
                                    max_tokens,
                                    temperature,
                                    top_p,
                                    stop,
                                    tools_requested,
                                    logits_processor_factories,
                                )
                            async for resp in response_stream:
                                last_resp = resp
                                if resp.text:
                                    text_parts.append(resp.text)
                                self._update_active(resp, new_text=resp.text or "")
                            return GenerationOutput(
                                text=clean_output_text("".join(text_parts)),
                                prompt_tokens=last_resp.prompt_tokens
                                if last_resp
                                else 0,
                                completion_tokens=last_resp.generation_tokens
                                if last_resp
                                else 0,
                                finished=True,
                                finish_reason=(
                                    last_resp.finish_reason if last_resp else None
                                )
                                or "stop",
                            )
                        finally:
                            self._track_request_end()
                    mode = forced_mode or (
                        "ddtree-ngram"
                        if self._should_enable_ngram_first(prompt, agentic_phase)
                        or self._thinking_ngram_enabled
                        else "ddtree"
                    )
                    self._track_request_start(mode)
                    if self._active is not None:
                        self._active.agentic_phase = agentic_phase
                        self._active.agentic_policy_decision = mode
                        self._active.agentic_policy_reason = policy_reason
                    try:
                        loop = asyncio.get_running_loop()
                        executor = self._executor
                        assert executor is not None, "DFlashEngine not started"
                        stop_event = threading.Event()
                        future = loop.run_in_executor(
                            executor,
                            lambda: self._run_ddtree_sync(
                                prompt=prompt,
                                max_tokens=max_tokens,
                                stop=stop,
                                tools_requested=tools_requested,
                                prefix_boundary=prefix_boundary,
                                emit_step_text=False,
                                should_stop=stop_event.is_set,
                                logits_processors=self._make_ddtree_logits_processors(
                                    logits_processor_factories
                                ),
                                agentic_phase=agentic_phase,
                            ),
                        )
                        try:
                            result = await future
                        except BaseException:
                            stop_event.set()
                            try:
                                await asyncio.wait_for(asyncio.shield(future), timeout=2.0)
                            except Exception:
                                pass
                            raise
                        resp = SimpleNamespace(
                            tokens=list(result.get("generated_token_ids", []) or []),
                            prompt_tokens=int(result.get("prompt_tokens") or 0),
                            prefill_seconds=float(result.get("prefill_seconds") or 0.0),
                            generation_tokens=int(result.get("generated_tokens") or 0),
                            generation_tps=float(result.get("generation_tps") or 0.0),
                            finish_reason=result.get("finish_reason") or "stop",
                            proposed_tokens=int(result.get("proposed_tokens") or 0),
                            accepted_tokens=int(result.get("accepted_tokens") or 0),
                            speculative_steps=int(result.get("speculative_steps") or 0),
                            avg_acceptance_ratio=float(
                                result.get("avg_acceptance_ratio") or 0.0
                            ),
                            block_size_history=tuple(result.get("block_size_history") or ()),
                            avg_tree_node_count=float(
                                result.get("avg_tree_node_count") or 0.0
                            ),
                            ddtree_fast_path_ratio=float(
                                result.get("ddtree_fast_path_ratio") or 0.0
                            ),
                            ngram_acceptance_ratio=float(
                                result.get("ngram_acceptance_ratio") or 0.0
                            ),
                            ngram_cycles=int(result.get("ngram_cycles_completed") or 0),
                            ngram_fallback_cycles=int(
                                result.get("ngram_fallback_cycles") or 0
                            ),
                            ngram_tool_guard_cycles=int(
                                result.get("ngram_tool_guard_cycles") or 0
                            ),
                            cache_hit_type=result.get("cache_hit_type"),
                            cached_tokens=int(result.get("cached_tokens") or 0),
                            phase_timings_us=dict(
                                result.get("ddtree_phase_timings_us") or {}
                            ),
                        )
                        self._update_active(resp, new_text=result.get("text", ""))
                        return GenerationOutput(
                            text=clean_output_text(result.get("text", "")),
                            prompt_tokens=resp.prompt_tokens,
                            completion_tokens=resp.generation_tokens,
                            finished=True,
                            finish_reason=resp.finish_reason,
                        )
                    finally:
                        self._track_request_end()
            finally:
                self._inflight -= 1

        last = None
        text_parts: list[str] = []
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            images=images,
            videos=videos,
            **kwargs,
        ):
            last = output
            if output.new_text:
                text_parts.append(output.new_text)
        return GenerationOutput(
            text=clean_output_text("".join(text_parts)),
            prompt_tokens=last.prompt_tokens if last else 0,
            completion_tokens=last.completion_tokens if last else 0,
            finished=True,
            finish_reason=(last.finish_reason if last else None) or "stop",
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        if images or videos:
            raise ValueError("DFlash mode does not support images or videos.")
        prompt = self._maybe_compress_prompt(prompt, kwargs)

        self._inflight += 1
        try:
            async with self._lock:
                tools_requested = bool(kwargs.pop("tools_requested", False))
                prefix_boundary = int(kwargs.pop("prefix_boundary", 0) or 0)
                prefix_boundaries = kwargs.pop("prefix_boundaries", None)
                agentic_phase = kwargs.pop("agentic_phase", None)
                kwargs.pop("agentic_speculative_policy", None)
                kwargs.pop("tool_call_expected", None)
                logits_processor_factories = kwargs.pop(
                    "logits_processor_factories", None
                )
                greedy_request = _is_greedy_sampling(temperature, top_p)
                forced_mode, policy_reason, policy_metadata = (
                    self._agentic_policy_decision(
                        prompt,
                        tools_requested=tools_requested,
                        max_tokens=max_tokens,
                        greedy_request=greedy_request,
                        phase=agentic_phase,
                        cached_tokens=self._ddtree_cached_token_count(prompt),
                    )
                )
                self._last_agentic_policy = policy_metadata | {
                    "decision": forced_mode,
                    "reason": policy_reason,
                }
                use_target_fallback = forced_mode == "target-fallback" or (
                    forced_mode is None
                    and self._should_use_agentic_target_fallback(
                        prompt, tools_requested
                    )
                )
                if self._ddtree_budget > 0 and not greedy_request:
                    logger.warning(
                        "[DDTree] falling back to DFlash for non-greedy request: temperature=%s top_p=%s",
                        temperature,
                        top_p,
                    )
                if use_target_fallback:
                    mode = (
                        "target-prefix-cache"
                        if self._should_use_target_prefix_cache(
                            temperature,
                            top_p,
                        )
                        else "target-fallback"
                    )
                else:
                    mode = forced_mode or (
                        "ddtree-ngram"
                        if (
                            (
                                self._should_enable_ngram_first(prompt, agentic_phase)
                                or self._thinking_ngram_enabled
                            )
                            and greedy_request
                        )
                        else (
                            "ddtree"
                            if self._ddtree_budget > 0 and greedy_request
                            else "dflash"
                        )
                    )
                if logits_processor_factories and mode == "dflash":
                    raise RuntimeError(
                        "Structured CoT logits masking requires greedy DDTree mode "
                        "when using DFlash. Set --dflash-ddtree-budget and "
                        "--default-temperature 0, or disable --structured-cot."
                    )
                self._track_request_start(mode)
                if self._active is not None:
                    self._active.agentic_phase = agentic_phase
                    self._active.agentic_policy_decision = mode
                    self._active.agentic_policy_reason = policy_reason
                    logger.info(
                        "[agentic-speculative-policy] phase=%s mode=%s reason=%s "
                        "prompt_tokens=%s acceptance=%s cooldown=%s",
                        agentic_phase,
                        mode,
                        policy_reason,
                        policy_metadata.get("prompt_tokens"),
                        policy_metadata.get("recent_acceptance"),
                        policy_metadata.get("cooldown"),
                    )
                cumulative = ""
                last_resp = None
                try:
                    if mode == "target-fallback":
                        response_stream = self._stream_target(
                            prompt,
                            max_tokens,
                            temperature,
                            top_p,
                            stop,
                            tools_requested,
                            logits_processor_factories,
                        )
                    elif mode == "target-prefix-cache":
                        response_stream = self._stream_target_cached(
                            prompt,
                            max_tokens,
                            temperature,
                            top_p,
                            stop,
                            tools_requested,
                            logits_processor_factories,
                            prefix_boundaries,
                        )
                    elif mode in ("ddtree", "ddtree-ngram"):
                        stream = self._stream_ddtree
                        response_stream = stream(
                            prompt,
                            max_tokens,
                            temperature,
                            top_p,
                            stop,
                            tools_requested,
                            prefix_boundary,
                            logits_processor_factories,
                            agentic_phase,
                        )
                    else:
                        stream = self._stream_dflash
                        response_stream = stream(prompt, max_tokens, temperature, top_p)
                    async for resp in response_stream:
                        last_resp = resp
                        new_text = resp.text or ""
                        cumulative += new_text
                        self._update_active(resp, new_text=new_text)

                        yield GenerationOutput(
                            text=clean_output_text(cumulative),
                            new_text=new_text,
                            tokens=list(resp.tokens) if resp.tokens else [],
                            prompt_tokens=resp.prompt_tokens,
                            completion_tokens=resp.generation_tokens,
                            finished=bool(resp.finish_reason),
                            finish_reason=resp.finish_reason,
                        )
                finally:
                    self._track_request_end()

                # Some clients expect a final yield with finished=True
                if last_resp is not None and not last_resp.finish_reason:
                    yield GenerationOutput(
                        text=clean_output_text(cumulative),
                        new_text="",
                        tokens=[],
                        prompt_tokens=last_resp.prompt_tokens,
                        completion_tokens=last_resp.generation_tokens,
                        finished=True,
                        finish_reason="stop",
                    )
        finally:
            self._inflight -= 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _track_request_start(self, mode: str = "dflash") -> None:
        self._active = _ActiveRequest(
            started_at=time.time(),
            mode=mode,
            block_size=self._current_block_size,
            tree_budget=(
                self._ddtree_budget if mode in ("ddtree", "ddtree-ngram") else 0
            ),
        )

    def _track_request_end(self) -> None:
        if self._active is not None:
            self._lifetime_responses += 1
            self._lifetime_proposed += self._active.proposed_tokens
            self._lifetime_accepted += self._active.accepted_tokens
            if self._active.mode in ("ddtree", "ddtree-ngram"):
                self._ddtree_responses += 1
            if self._active.agentic_policy_decision:
                self._agentic_policy_history.append(
                    {
                        "mode": self._active.mode,
                        "phase": self._active.agentic_phase,
                        "decision": self._active.agentic_policy_decision,
                        "reason": self._active.agentic_policy_reason,
                        "acceptance_ratio": self._active.acceptance_ratio,
                        "proposed_tokens": self._active.proposed_tokens,
                        "accepted_tokens": self._active.accepted_tokens,
                        "generated_tokens": self._active.generated_tokens,
                        "prompt_tokens": self._active.prompt_tokens,
                    }
                )
        self._active = None

    def _update_active(self, resp: Any, new_text: str = "") -> None:
        a = self._active
        if a is None:
            return
        if a.first_token_at is None and (new_text or resp.tokens):
            a.first_token_at = time.time()
        a.prompt_tokens = int(resp.prompt_tokens or 0)
        a.prefill_seconds = float(getattr(resp, "prefill_seconds", 0.0) or 0.0)
        a.generated_tokens = int(resp.generation_tokens or 0)
        a.generation_tps = float(getattr(resp, "generation_tps", 0.0) or 0.0)
        a.proposed_tokens = int(resp.proposed_tokens or 0)
        a.accepted_tokens = int(resp.accepted_tokens or 0)
        a.speculative_steps = int(resp.speculative_steps or 0)
        a.acceptance_ratio = float(resp.avg_acceptance_ratio or 0.0)
        a.avg_tree_node_count = float(getattr(resp, "avg_tree_node_count", 0.0) or 0.0)
        a.ddtree_fast_path_ratio = float(
            getattr(resp, "ddtree_fast_path_ratio", 0.0) or 0.0
        )
        a.ngram_acceptance_ratio = float(
            getattr(resp, "ngram_acceptance_ratio", 0.0) or 0.0
        )
        a.ngram_cycles = int(getattr(resp, "ngram_cycles", 0) or 0)
        a.ngram_fallback_cycles = int(
            getattr(resp, "ngram_fallback_cycles", 0) or 0
        )
        a.ngram_tool_guard_cycles = int(
            getattr(resp, "ngram_tool_guard_cycles", 0) or 0
        )
        a.cache_hit_type = getattr(resp, "cache_hit_type", None)
        a.cached_tokens = int(getattr(resp, "cached_tokens", 0) or 0)
        a.phase_timings_us = dict(getattr(resp, "phase_timings_us", {}) or {})
        history = list(resp.block_size_history or ())
        if history:
            a.block_history = history
            a.block_size = int(history[-1])
            self._current_block_size = a.block_size
            mn = min(history)
            mx_ = max(history)
            self._observed_block_min = (
                mn if self._observed_block_min == 0 else min(self._observed_block_min, mn)
            )
            self._observed_block_max = max(self._observed_block_max, mx_)

    def get_stats(self) -> dict[str, Any]:
        active_mem_gb, peak_mem_gb, cache_mem_gb = self._last_memory_stats
        if self._active is None or _env_bool("DFLASH_STATS_MEMORY_DURING_ACTIVE", False):
            try:
                import mlx.core as mx

                active_mem_gb = mx.get_active_memory() / 1e9
                peak_mem_gb = mx.get_peak_memory() / 1e9
                cache_mem_gb = mx.get_cache_memory() / 1e9
                self._last_memory_stats = (active_mem_gb, peak_mem_gb, cache_mem_gb)
            except Exception:
                active_mem_gb = peak_mem_gb = cache_mem_gb = 0.0

        running_requests: list[dict[str, Any]] = []
        if self._active is not None:
            now = time.time()
            elapsed = now - self._active.started_at
            ttft = self._active.prefill_seconds if self._active.prefill_seconds > 0 else None
            if ttft is None and self._active.first_token_at:
                ttft = self._active.first_token_at - self._active.started_at
            tps = self._active.generation_tps if self._active.generation_tps > 0 else None
            if (
                tps is None
                and self._active.first_token_at
                and self._active.generated_tokens > 0
            ):
                window = now - self._active.first_token_at
                if window > 0.01:
                    tps = self._active.generated_tokens / window
            running_requests.append(
                {
                    "request_id": f"{self._active.mode}-active",
                    "mode": self._active.mode,
                    "status": "running",
                    "phase": (
                        "generation" if self._active.first_token_at else "prefill"
                    ),
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": self._active.prompt_tokens,
                    "completion_tokens": self._active.generated_tokens,
                    "max_tokens": 0,
                    "tokens_per_second": tps,
                    "ttft_s": ttft,
                    "acceptance_ratio": self._active.acceptance_ratio,
                    "block_size": self._active.block_size,
                    "speculative_steps": self._active.speculative_steps,
                    "accepted_tokens": self._active.accepted_tokens,
                    "proposed_tokens": self._active.proposed_tokens,
                    "tree_budget": self._active.tree_budget,
                    "avg_tree_node_count": self._active.avg_tree_node_count,
                    "ddtree_fast_path_ratio": self._active.ddtree_fast_path_ratio,
                    "ngram_acceptance_ratio": self._active.ngram_acceptance_ratio,
                    "ngram_cycles": self._active.ngram_cycles,
                    "ngram_fallback_cycles": self._active.ngram_fallback_cycles,
                    "ngram_tool_guard_cycles": self._active.ngram_tool_guard_cycles,
                    "cache_hit_type": self._active.cache_hit_type,
                    "cached_tokens": self._active.cached_tokens,
                    "agentic_phase": self._active.agentic_phase,
                    "agentic_policy_decision": self._active.agentic_policy_decision,
                    "agentic_policy_reason": self._active.agentic_policy_reason,
                    "phase_timings_us": self._active.phase_timings_us,
                    "progress": 0.0,
                }
            )

        lifetime_ratio = (
            (self._lifetime_accepted / self._lifetime_proposed)
            if self._lifetime_proposed > 0
            else 0.0
        )

        num_running = 1 if self._active is not None else 0
        num_waiting = max(0, self._inflight - num_running)
        prefill_result = self._speculative_prefill.last_result

        return {
            "engine_type": "dflash",
            "model_name": self._model_name,
            "is_mllm": False,
            "loaded": self._loaded,
            "running": self._active is not None,
            "uptime_seconds": time.time() - self._start_time,
            "num_running": num_running,
            "num_waiting": num_waiting,
            "total_requests_processed": self._lifetime_responses,
            "metal_active_memory_gb": active_mem_gb,
            "metal_peak_memory_gb": peak_mem_gb,
            "metal_cache_memory_gb": cache_mem_gb,
            "requests": running_requests,
            "speculative_prefill": {
                "enabled": self._speculative_prefill.config.enabled,
                "last_applied": bool(prefill_result and prefill_result.applied),
                "last_reason": prefill_result.reason if prefill_result else None,
                "last_original_tokens": (
                    prefill_result.original_tokens if prefill_result else 0
                ),
                "last_compressed_tokens": (
                    prefill_result.compressed_tokens if prefill_result else 0
                ),
                "last_tokens_saved": (
                    prefill_result.tokens_saved if prefill_result else 0
                ),
                "last_draft_model_used": (
                    bool(prefill_result and prefill_result.draft_model_used)
                ),
            },
            "dflash": {
                "mode": (
                    "ddtree-ngram"
                    if self._ngram_first_enabled or self._thinking_ngram_enabled
                    else ("ddtree" if self._ddtree_budget > 0 else "dflash")
                ),
                "lifetime_acceptance_ratio": lifetime_ratio,
                "current_block_size": self._current_block_size,
                "adaptive_enabled": self._adaptive_enabled,
                "adaptive_min": self._adaptive_min,
                "adaptive_max": self._adaptive_max,
                "observed_block_min": self._observed_block_min,
                "observed_block_max": self._observed_block_max,
                "ddtree_budget": self._ddtree_budget,
                "ddtree_block_size": self._ddtree_block_size,
                "ddtree_requests": self._ddtree_responses,
                "ddtree_last_fast_path_ratio": self._ddtree_last.get(
                    "ddtree_fast_path_ratio", 0.0
                ),
                "ddtree_last_avg_tree_node_count": self._ddtree_last.get(
                    "avg_tree_node_count", 0.0
                ),
                "ddtree_last_generation_tps": self._ddtree_last.get(
                    "generation_tps", 0.0
                ),
                "ddtree_last_phase_timings_us": self._ddtree_last.get(
                    "ddtree_phase_timings_us", {}
                ),
                "ngram_first_enabled": self._ngram_first_enabled,
                "ngram_num_draft_tokens": self._ngram_num_draft_tokens,
                "ngram_size": self._ngram_size,
                "ngram_min_matches": self._ngram_min_matches,
                "ngram_disable_threshold": self._ngram_disable_threshold,
                "ngram_disable_window": self._ngram_disable_window,
                "ngram_disable_cooldown": self._ngram_disable_cooldown,
                "thinking_ngram_enabled": self._thinking_ngram_enabled,
                "thinking_ngram_num_draft_tokens": (
                    self._thinking_ngram_num_draft_tokens or 0
                ),
                "thinking_ngram_size": self._thinking_ngram_size or 0,
                "thinking_ngram_min_matches": (
                    self._thinking_ngram_min_matches or 0
                ),
                "agentic_target_fallback_enabled": _env_bool(
                    "DFLASH_AGENTIC_TARGET_FALLBACK", True
                ),
                "target_prefix_cache_enabled": self._target_prefix_cache_enabled,
                "target_prefix_cache": self._target_prefix_cache.get_stats(),
                "agentic_speculative_policy": self._agentic_speculative_policy,
                "agentic_policy_cooldown": self._agentic_policy_cooldown,
                "agentic_policy_last": self._last_agentic_policy,
                "agentic_policy_history": list(self._agentic_policy_history),
                "agentic_target_fallback_min_prompt_tokens": int(
                    os.environ.get(
                        "DFLASH_AGENTIC_TARGET_FALLBACK_MIN_PROMPT_TOKENS",
                        "4096",
                    )
                ),
                "ngram_last_acceptance_ratio": self._ddtree_last.get(
                    "ngram_acceptance_ratio", 0.0
                ),
                "ngram_last_cycles": self._ddtree_last.get(
                    "ngram_cycles_completed", 0
                ),
                "ngram_last_fallback_cycles": self._ddtree_last.get(
                    "ngram_fallback_cycles", 0
                ),
                "ngram_last_tool_guard_cycles": self._ddtree_last.get(
                    "ngram_tool_guard_cycles", 0
                ),
            },
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return {
            "ddtree": self._ddtree_prefix_cache.get_stats(),
            "target": self._target_prefix_cache.get_stats(),
        }
