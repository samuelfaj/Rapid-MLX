# SPDX-License-Identifier: Apache-2.0
"""Speculative prefill prompt compression for MLX generation.

The compressor is deliberately conservative: it only rewrites prompts when it
can reduce token count while preserving protected semantic anchors. If the
draft-model scorer is unavailable or the safety checks fail, it returns the
original prompt.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


_PROTECTED_PATTERNS = (
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`\n]+`"),
    re.compile(r'"[^"\n]{1,240}"'),
    re.compile(r"'[^'\n]{1,240}'"),
    re.compile(r"https?://\S+"),
    re.compile(r"/(?:[\w.-]+/)+[\w.-]+"),
    re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b"),
    re.compile(r"\b\d+(?:\.\d+)?\b"),
)


@dataclass
class SpeculativePrefillConfig:
    enabled: bool = False
    draft_model_path: str | None = None
    target_token_ratio: float = 0.85
    min_prompt_tokens: int = 128
    preserve_first_tokens: int = 32
    preserve_last_tokens: int = 64


@dataclass
class SpeculativePrefillResult:
    original_prompt: str
    prompt: str
    original_tokens: int
    compressed_tokens: int
    enabled: bool
    applied: bool
    reason: str
    draft_model_used: bool = False

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


class SpeculativePrefillCompressor:
    """Token-importance compressor with optional small draft-model scoring."""

    def __init__(
        self,
        config: SpeculativePrefillConfig | None = None,
        score_fn: Callable[[list[int], Any], list[float]] | None = None,
    ) -> None:
        self.config = config or SpeculativePrefillConfig()
        self._score_fn = score_fn
        self._draft_model: Any | None = None
        self._draft_tokenizer: Any | None = None
        self._draft_load_failed = False
        self.last_result: SpeculativePrefillResult | None = None

    def compress(self, prompt: str, tokenizer: Any) -> SpeculativePrefillResult:
        cfg = self.config
        tokens = _encode(tokenizer, prompt)
        original_tokens = len(tokens)
        if not cfg.enabled:
            return self._record(prompt, prompt, original_tokens, original_tokens, False, "disabled")
        if original_tokens < max(1, cfg.min_prompt_tokens):
            return self._record(
                prompt,
                prompt,
                original_tokens,
                original_tokens,
                True,
                "below_min_prompt_tokens",
            )

        target_count = int(original_tokens * min(max(cfg.target_token_ratio, 0.05), 1.0))
        protected = self._protected_token_indexes(prompt, tokenizer, tokens)
        protected.update(range(min(cfg.preserve_first_tokens, original_tokens)))
        protected.update(
            range(max(0, original_tokens - cfg.preserve_last_tokens), original_tokens)
        )
        if len(protected) >= target_count:
            return self._record(
                prompt,
                prompt,
                original_tokens,
                original_tokens,
                True,
                "protected_floor",
            )

        scores, draft_used = self._score_tokens(tokens, tokenizer)
        keep = set(protected)
        candidates = [
            (scores[idx], idx)
            for idx in range(original_tokens)
            if idx not in keep
        ]
        candidates.sort(reverse=True)
        for _, idx in candidates[: max(0, target_count - len(keep))]:
            keep.add(idx)

        kept_tokens = [tok for idx, tok in enumerate(tokens) if idx in keep]
        compressed = _decode(tokenizer, kept_tokens)
        compressed_tokens = len(_encode(tokenizer, compressed))
        if compressed_tokens >= original_tokens:
            return self._record(
                prompt,
                prompt,
                original_tokens,
                original_tokens,
                True,
                "no_token_saving",
            )
        if not self._semantic_anchors_preserved(prompt, compressed):
            return self._record(
                prompt,
                prompt,
                original_tokens,
                original_tokens,
                True,
                "semantic_anchor_mismatch",
            )

        result = self._record(
            prompt,
            compressed,
            original_tokens,
            compressed_tokens,
            True,
            "compressed",
            draft_model_used=draft_used,
        )
        logger.info(
            "[speculative-prefill] compressed prompt: %d -> %d tokens (ratio %.3f, draft=%s)",
            result.original_tokens,
            result.compressed_tokens,
            result.compression_ratio,
            result.draft_model_used,
        )
        return result

    def _score_tokens(self, tokens: list[int], tokenizer: Any) -> tuple[list[float], bool]:
        if self._score_fn is not None:
            scores = self._score_fn(tokens, tokenizer)
            if len(scores) == len(tokens):
                return list(scores), False

        draft_scores = self._draft_model_scores(tokens)
        if draft_scores is not None:
            return draft_scores, True

        return self._lexical_scores(tokens, tokenizer), False

    def _draft_model_scores(self, tokens: list[int]) -> list[float] | None:
        cfg = self.config
        if not cfg.draft_model_path or self._draft_load_failed:
            return None
        try:
            import mlx.core as mx
            import numpy as np

            if self._draft_model is None:
                from ..utils.tokenizer import load_model_with_fallback

                self._draft_model, self._draft_tokenizer = load_model_with_fallback(
                    cfg.draft_model_path,
                    tokenizer_config={"trust_remote_code": True},
                )

            if len(tokens) < 2:
                return [1.0] * len(tokens)
            input_ids = mx.array([tokens[:-1]])
            output = self._draft_model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            logits = logits.astype(mx.float32)
            mx.eval(logits)
            logits_np = np.asarray(logits[0])
            next_ids = np.asarray(tokens[1:], dtype=np.int64)
            logits_np = logits_np - logits_np.max(axis=-1, keepdims=True)
            exp = np.exp(logits_np)
            probs = exp / exp.sum(axis=-1, keepdims=True)
            surprisal = -np.log(np.maximum(probs[np.arange(len(next_ids)), next_ids], 1e-9))
            scores = [float(surprisal[0]) if len(surprisal) else 1.0]
            scores.extend(float(v) for v in surprisal)
            mx.clear_cache()
            return scores[: len(tokens)]
        except Exception as exc:
            self._draft_load_failed = True
            logger.warning("[speculative-prefill] draft scorer disabled: %s", exc)
            return None

    @staticmethod
    def _lexical_scores(tokens: list[int], tokenizer: Any) -> list[float]:
        scores: list[float] = []
        for token in tokens:
            piece = _decode(tokenizer, [token])
            score = 0.1
            if any(ch.isalnum() for ch in piece):
                score += 0.4
            if any(ch in piece for ch in ".?!:;{}[]()<>/\\"):
                score += 0.3
            if any(ch.isupper() for ch in piece):
                score += 0.2
            scores.append(score)
        return scores

    @staticmethod
    def _protected_token_indexes(
        prompt: str,
        tokenizer: Any,
        tokens: list[int],
    ) -> set[int]:
        spans: list[tuple[int, int]] = []
        for pattern in _PROTECTED_PATTERNS:
            spans.extend((match.start(), match.end()) for match in pattern.finditer(prompt))
        if not spans:
            return set()

        protected: set[int] = set()
        cursor = 0
        for idx, token in enumerate(tokens):
            piece = _decode(tokenizer, [token])
            start = cursor
            end = cursor + len(piece)
            cursor = end
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                protected.add(idx)
        return protected

    @staticmethod
    def _semantic_anchors_preserved(original: str, compressed: str) -> bool:
        for pattern in _PROTECTED_PATTERNS:
            for match in pattern.finditer(original):
                anchor = match.group(0)
                if anchor and anchor not in compressed:
                    return False
        return True

    def _record(
        self,
        original: str,
        prompt: str,
        original_tokens: int,
        compressed_tokens: int,
        enabled: bool,
        reason: str,
        draft_model_used: bool = False,
    ) -> SpeculativePrefillResult:
        result = SpeculativePrefillResult(
            original_prompt=original,
            prompt=prompt,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            enabled=enabled,
            applied=prompt != original,
            reason=reason,
            draft_model_used=draft_model_used,
        )
        self.last_result = result
        return result


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    encoded = tokenizer.encode(text)
    return encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)


def _decode(tokenizer: Any, tokens: list[int]) -> str:
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    return tokenizer.decode(tokens)
