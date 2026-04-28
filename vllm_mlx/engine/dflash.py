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
import logging
import sys
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..api.utils import clean_output_text
from .base import GenerationOutput
from .batched import BatchedEngine

logger = logging.getLogger(__name__)

_MLX_BATCH_NONE_LEN_ERROR = "object of type 'NoneType' has no len()"


def _is_mlx_batch_terminal_error(exc: Exception) -> bool:
    return _MLX_BATCH_NONE_LEN_ERROR in str(exc)


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
    generated_tokens: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    speculative_steps: int = 0
    acceptance_ratio: float = 0.0
    block_size: int = 0
    block_history: list[int] = field(default_factory=list)


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
        fallback_mode: str = "ngram",
        disable_threshold: float = 0.55,
        disable_window: int = 4,
        disable_cooldown: int = 8,
        ngram_num_draft_tokens: int = 4,
        ngram_size: int = 3,
        ngram_min_matches: int = 1,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=None,
            stream_interval=1,
            force_mllm=False,
            gpu_memory_utilization=gpu_memory_utilization,
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
        self._fallback_mode = fallback_mode
        self._disable_threshold = float(disable_threshold)
        self._disable_window = max(1, int(disable_window))
        self._disable_cooldown = max(0, int(disable_cooldown))
        self._ngram_num_draft_tokens = int(ngram_num_draft_tokens)
        self._ngram_size = int(ngram_size)
        self._ngram_min_matches = int(ngram_min_matches)

        self._lock = asyncio.Lock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._active: _ActiveRequest | None = None
        self._inflight = 0

        self._adaptive_cfg: Any = None
        self._current_block_size = 0
        self._observed_block_min = 0
        self._observed_block_max = 0

        self._lifetime_proposed = 0
        self._lifetime_accepted = 0
        self._lifetime_responses = 0
        self._fallback_responses = 0
        self._fallback_proposed = 0
        self._fallback_accepted = 0
        self._recent_acceptance: deque[float] = deque(maxlen=self._disable_window)
        self._dflash_disabled_remaining = 0
        self._dflash_disabled_reason = ""
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

        from mlx_lm import load as mlx_load

        logger.info("[DFlash] Loading target model: %s", self._model_name)
        self._model, self._tokenizer = mlx_load(self._model_name)

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
            self._executor.shutdown(wait=False)
            self._executor = None
        self._drafter = None

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
        seen_response = False

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel
            except TypeError as exc:
                if (
                    seen_response
                    and "object of type 'NoneType' has no len()" in str(exc)
                ):
                    logger.warning(
                        "[DFlash] treating terminal detokenizer TypeError as clean stop: %s",
                        exc,
                    )
                    return sentinel
                raise

        while True:
            resp = await loop.run_in_executor(executor, _next)
            if resp is sentinel:
                return
            seen_response = True
            yield resp

    async def _stream_scheduler_fallback(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        mode: str,
    ) -> AsyncIterator[GenerationOutput]:
        """Run AR or n-gram fallback in the DFlash MLX worker thread."""
        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "DFlashEngine not started"

        def _make_gen():
            return self._scheduler_fallback_iter(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                mode=mode,
            )

        gen = await loop.run_in_executor(executor, _make_gen)
        sentinel = object()
        terminal_error = object()

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel
            except TypeError as exc:
                if _is_mlx_batch_terminal_error(exc):
                    logger.warning(
                        "[DFlash] fallback %s hit terminal BatchGenerator error; "
                        "ending stream cleanly: %s",
                        mode,
                        exc,
                    )
                    return terminal_error
                raise

        cumulative = ""
        last_prompt_tokens = 0
        last_completion_tokens = 0
        while True:
            item = await loop.run_in_executor(executor, _next)
            if item is sentinel:
                return
            if item is terminal_error:
                yield GenerationOutput(
                    text=clean_output_text(cumulative),
                    new_text="",
                    tokens=[],
                    prompt_tokens=last_prompt_tokens,
                    completion_tokens=last_completion_tokens,
                    finished=True,
                    finish_reason="stop",
                )
                return
            kind, payload, stats = item
            if kind == "stats":
                self._record_fallback_stats(stats)
                continue
            if kind == "finish":
                self._record_fallback_stats(stats)
                yield GenerationOutput(
                    text=clean_output_text(cumulative),
                    new_text="",
                    tokens=[],
                    prompt_tokens=last_prompt_tokens,
                    completion_tokens=last_completion_tokens,
                    finished=True,
                    finish_reason="stop",
                )
                return

            output = payload
            cumulative += output.new_text or ""
            self._update_active_fallback(output, stats, mode)
            last_prompt_tokens = output.prompt_tokens or last_prompt_tokens
            last_completion_tokens = (
                output.completion_tokens or last_completion_tokens
            )
            yield GenerationOutput(
                text=clean_output_text(cumulative),
                new_text=output.new_text,
                tokens=list(output.new_token_ids or []),
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finished=output.finished,
                finish_reason=output.finish_reason,
                logprobs=output.logprobs,
            )

    def _scheduler_fallback_iter(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        mode: str,
    ):
        from ..request import Request, SamplingParams
        from ..scheduler import Scheduler, SchedulerConfig

        spec_type = "ngram-mod" if mode == "ngram" else None
        scheduler_config = SchedulerConfig(
            max_num_seqs=1,
            prefill_batch_size=1,
            completion_batch_size=1,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            spec_type=spec_type,
            ngram_num_draft_tokens=self._ngram_num_draft_tokens,
            ngram_size=self._ngram_size,
            ngram_min_matches=self._ngram_min_matches,
        )
        scheduler = Scheduler(
            model=self._model,
            tokenizer=self._tokenizer,
            config=scheduler_config,
        )
        request = Request(
            request_id="dflash-fallback",
            prompt=prompt,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )
        scheduler.add_request(request)

        try:
            while scheduler.has_requests():
                try:
                    output = scheduler.step()
                except TypeError as exc:
                    if _is_mlx_batch_terminal_error(exc):
                        logger.warning(
                            "[DFlash] fallback %s scheduler ended on terminal "
                            "BatchGenerator error: %s",
                            mode,
                            exc,
                        )
                        yield "finish", None, scheduler.get_stats()
                        break
                    raise
                stats = scheduler.get_stats()
                for req_output in output.outputs:
                    yield "output", req_output, stats
            yield "stats", None, scheduler.get_stats()
        finally:
            try:
                scheduler.deep_reset()
            except TypeError as exc:
                if _is_mlx_batch_terminal_error(exc):
                    logger.debug(
                        "[DFlash] ignoring fallback %s cleanup BatchGenerator "
                        "terminal error: %s",
                        mode,
                        exc,
                    )
                else:
                    raise

    def _choose_generation_mode(self) -> str:
        if self._dflash_disabled_remaining > 0:
            self._dflash_disabled_remaining -= 1
            if self._fallback_mode != "none":
                return self._fallback_mode
        return "dflash"

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

        self._inflight += 1
        try:
            async with self._lock:
                mode = self._choose_generation_mode()
                self._track_request_start(mode)
                cumulative = ""
                raw_dflash_text = ""
                last_resp = None
                try:
                    if mode == "dflash":
                        async for resp in self._stream_dflash(
                            prompt, max_tokens, temperature, top_p
                        ):
                            last_resp = resp
                            raw_text = resp.text or ""
                            if raw_text and raw_text.startswith(raw_dflash_text):
                                new_text = raw_text[len(raw_dflash_text) :]
                                raw_dflash_text = raw_text
                            else:
                                new_text = raw_text
                                raw_dflash_text += raw_text
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
                    else:
                        async for output in self._stream_scheduler_fallback(
                            prompt, max_tokens, temperature, top_p, mode
                        ):
                            last_resp = output
                            cumulative = output.text
                            yield output
                finally:
                    self._track_request_end()

                # Some clients expect a final yield with finished=True
                if last_resp is not None and not last_resp.finish_reason:
                    final_completion = int(
                        getattr(
                            last_resp,
                            "generation_tokens",
                            getattr(last_resp, "completion_tokens", 0),
                        )
                        or 0
                    )
                    final_prompt = int(getattr(last_resp, "prompt_tokens", 0) or 0)
                    final_reason = (
                        "length"
                        if max_tokens > 0 and final_completion >= max_tokens
                        else "stop"
                    )
                    yield GenerationOutput(
                        text=clean_output_text(cumulative),
                        new_text="",
                        tokens=[],
                        prompt_tokens=final_prompt,
                        completion_tokens=final_completion,
                        finished=True,
                        finish_reason=final_reason,
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
            block_size=self._current_block_size if mode == "dflash" else 0,
        )

    def _track_request_end(self) -> None:
        a = self._active
        if a is not None:
            if a.mode == "dflash":
                self._lifetime_responses += 1
                self._lifetime_proposed += a.proposed_tokens
                self._lifetime_accepted += a.accepted_tokens
                if a.proposed_tokens > 0:
                    self._recent_acceptance.append(a.acceptance_ratio)
                    self._maybe_disable_dflash()
            else:
                self._fallback_responses += 1
                self._fallback_proposed += a.proposed_tokens
                self._fallback_accepted += a.accepted_tokens
        self._active = None

    def _maybe_disable_dflash(self) -> None:
        if self._fallback_mode == "none" or self._disable_cooldown <= 0:
            return
        if len(self._recent_acceptance) < self._disable_window:
            return
        recent = sum(self._recent_acceptance) / len(self._recent_acceptance)
        if recent >= self._disable_threshold:
            return
        self._dflash_disabled_remaining = self._disable_cooldown
        self._dflash_disabled_reason = (
            f"recent_acceptance={recent:.1%}<threshold={self._disable_threshold:.1%}"
        )
        self._recent_acceptance.clear()
        logger.info(
            "[DFlash] disabling for %d request(s); %s; fallback=%s",
            self._disable_cooldown,
            self._dflash_disabled_reason,
            self._fallback_mode,
        )

    def _update_active(self, resp: Any, new_text: str = "") -> None:
        a = self._active
        if a is None:
            return
        if a.first_token_at is None and (new_text or resp.tokens):
            a.first_token_at = time.time()
        a.prompt_tokens = int(resp.prompt_tokens or 0)
        a.generated_tokens = int(resp.generation_tokens or 0)
        a.proposed_tokens = int(resp.proposed_tokens or 0)
        a.accepted_tokens = int(resp.accepted_tokens or 0)
        a.speculative_steps = int(resp.speculative_steps or 0)
        a.acceptance_ratio = float(resp.avg_acceptance_ratio or 0.0)
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

    def _update_active_fallback(self, output: Any, stats: dict[str, Any], mode: str):
        a = self._active
        if a is None:
            return
        if a.first_token_at is None and (output.new_text or output.new_token_ids):
            a.first_token_at = time.time()
        a.mode = mode
        a.prompt_tokens = int(output.prompt_tokens or 0)
        a.generated_tokens = int(output.completion_tokens or 0)
        ngram_stats = stats.get("ngram_mod") or {}
        proposed = int(ngram_stats.get("proposed_tokens") or 0)
        accepted = int(ngram_stats.get("accepted_tokens") or 0)
        a.proposed_tokens = proposed
        a.accepted_tokens = accepted
        a.acceptance_ratio = accepted / proposed if proposed > 0 else 0.0
        a.block_size = 0

    def _record_fallback_stats(self, stats: dict[str, Any]) -> None:
        a = self._active
        if a is None:
            return
        ngram_stats = stats.get("ngram_mod") or {}
        if not ngram_stats:
            return
        proposed = int(ngram_stats.get("proposed_tokens") or 0)
        accepted = int(ngram_stats.get("accepted_tokens") or 0)
        a.proposed_tokens = proposed
        a.accepted_tokens = accepted
        a.acceptance_ratio = accepted / proposed if proposed > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        try:
            import mlx.core as mx

            active_mem_gb = mx.get_active_memory() / 1e9
            peak_mem_gb = mx.get_peak_memory() / 1e9
            cache_mem_gb = mx.get_cache_memory() / 1e9
        except Exception:
            active_mem_gb = peak_mem_gb = cache_mem_gb = 0.0

        running_requests: list[dict[str, Any]] = []
        if self._active is not None:
            now = time.time()
            elapsed = now - self._active.started_at
            ttft = (
                (self._active.first_token_at - self._active.started_at)
                if self._active.first_token_at
                else None
            )
            tps = None
            if (
                self._active.first_token_at
                and self._active.generated_tokens > 0
            ):
                window = now - self._active.first_token_at
                if window > 0.01:
                    tps = self._active.generated_tokens / window
            running_requests.append(
                {
                    "request_id": f"{self._active.mode}-active",
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
                    "cache_hit_type": None,
                    "cached_tokens": 0,
                    "progress": 0.0,
                }
            )

        lifetime_ratio = (
            (self._lifetime_accepted / self._lifetime_proposed)
            if self._lifetime_proposed > 0
            else 0.0
        )
        fallback_ratio = (
            (self._fallback_accepted / self._fallback_proposed)
            if self._fallback_proposed > 0
            else 0.0
        )
        recent_ratio = (
            sum(self._recent_acceptance) / len(self._recent_acceptance)
            if self._recent_acceptance
            else 0.0
        )

        num_running = 1 if self._active is not None else 0
        num_waiting = max(0, self._inflight - num_running)

        return {
            "engine_type": "dflash",
            "model_name": self._model_name,
            "is_mllm": False,
            "loaded": self._loaded,
            "running": self._active is not None,
            "uptime_seconds": time.time() - self._start_time,
            "num_running": num_running,
            "num_waiting": num_waiting,
            "total_requests_processed": self._lifetime_responses
            + self._fallback_responses,
            "metal_active_memory_gb": active_mem_gb,
            "metal_peak_memory_gb": peak_mem_gb,
            "metal_cache_memory_gb": cache_mem_gb,
            "requests": running_requests,
            "dflash": {
                "lifetime_acceptance_ratio": lifetime_ratio,
                "current_block_size": self._current_block_size,
                "adaptive_enabled": self._adaptive_enabled,
                "adaptive_min": self._adaptive_min,
                "adaptive_max": self._adaptive_max,
                "observed_block_min": self._observed_block_min,
                "observed_block_max": self._observed_block_max,
                "fallback_mode": self._fallback_mode,
                "fallback_requests": self._fallback_responses,
                "fallback_acceptance_ratio": fallback_ratio,
                "disable_threshold": self._disable_threshold,
                "disable_window": self._disable_window,
                "disable_cooldown": self._disable_cooldown,
                "disabled_remaining": self._dflash_disabled_remaining,
                "disabled_reason": self._dflash_disabled_reason,
                "recent_acceptance_ratio": recent_ratio,
            },
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None
