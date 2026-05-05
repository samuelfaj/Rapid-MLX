# SPDX-License-Identifier: Apache-2.0
"""Structured Chain-of-Thought constrained decoding.

Implements the fixed grammars from https://github.com/andthattoo/structured-cot
without a general GBNF parser.  The processor constrains only the opening
``<think>`` plan.  After ``</think>\n\n`` generation is unrestricted again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx

StructuredCoTMode = Literal["basic", "plan"]

_BASIC_LABELS = ("GOAL", "APPROACH", "EDGE")
_PLAN_LABELS = ("GOAL", "STATE", "ALGO", "EDGE", "VERIFY")
_CLOSE = "</think>\n\n"


def normalize_structured_cot_mode(value: bool | str | None) -> StructuredCoTMode | None:
    """Normalize request value to a supported mode."""
    if value is None or value is False:
        return None
    if value is True:
        return "plan"
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"basic", "fsm", "goals"}:
        return "basic"
    if mode in {"plan", "fsm_plan", "lcb_plan", "livecodebench"}:
        return "plan"
    raise ValueError(
        "structured_cot must be true, false, 'basic', 'plan', or 'lcb_plan'"
    )


def _labels_for_mode(mode: StructuredCoTMode) -> tuple[str, ...]:
    return _BASIC_LABELS if mode == "basic" else _PLAN_LABELS


def _consume_literal(text: str, pos: int, literal: str) -> tuple[bool, int, bool]:
    """Consume literal. Returns (valid, new_pos, still_prefix)."""
    remaining = text[pos:]
    if len(remaining) < len(literal):
        return literal.startswith(remaining), pos + len(remaining), True
    if remaining.startswith(literal):
        return True, pos + len(literal), False
    return False, pos, False


def is_structured_cot_prefix(text: str, mode: StructuredCoTMode) -> bool:
    """Return true if text can still become a valid structured CoT output."""
    pos = 0
    valid, pos, prefix = _consume_literal(text, pos, "<think>\n")
    if not valid:
        return False
    if prefix:
        return True

    for label in _labels_for_mode(mode):
        valid, pos, prefix = _consume_literal(text, pos, f"{label}: ")
        if not valid:
            return False
        if prefix:
            return True

        remaining = text[pos:]
        newline_at = remaining.find("\n")
        if newline_at == -1:
            return "\n" not in remaining
        if newline_at == 0:
            return False
        pos += newline_at + 1

    valid, pos, prefix = _consume_literal(text, pos, _CLOSE)
    if not valid:
        return False
    if prefix:
        return True
    return True


def structured_cot_complete(text: str, mode: StructuredCoTMode) -> bool:
    """Return true once the constrained think block is complete."""
    if not is_structured_cot_prefix(text, mode):
        return False
    return _CLOSE in text


@dataclass
class StructuredCoTLogitsProcessor:
    """Logits processor that enforces a compact structured ``<think>`` block."""

    tokenizer: Any
    mode: StructuredCoTMode = "plan"
    prompt_token_count: int = 0

    def __post_init__(self) -> None:
        self._tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        try:
            vocab_size = len(self._tokenizer)
        except TypeError:
            vocab_size = int(self._tokenizer.vocab_size)
        self._token_texts = tuple(self._decode_token(i) for i in range(vocab_size))
        self._allowed_cache: dict[str, tuple[int, ...]] = {}

    def _decode_token(self, token_id: int) -> str:
        try:
            return self._tokenizer.decode([token_id])
        except Exception:
            return ""

    def _decode_history(self, tokens: Any) -> str:
        try:
            token_ids = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
            if self.prompt_token_count and len(token_ids) >= self.prompt_token_count:
                token_ids = token_ids[self.prompt_token_count :]
            return self._tokenizer.decode(token_ids)
        except Exception:
            return ""

    def _allowed_token_ids(self, prefix: str) -> tuple[int, ...]:
        cached = self._allowed_cache.get(prefix)
        if cached is not None:
            return cached
        if structured_cot_complete(prefix, self.mode):
            allowed = tuple(range(len(self._token_texts)))
            self._allowed_cache[prefix] = allowed
            return allowed

        allowed = []
        for token_id, token_text in enumerate(self._token_texts):
            if not token_text:
                continue
            candidate = prefix + token_text
            if is_structured_cot_prefix(candidate, self.mode):
                allowed.append(token_id)
        result = tuple(allowed)
        if len(self._allowed_cache) >= 4096:
            self._allowed_cache.clear()
        self._allowed_cache[prefix] = result
        return result

    def __call__(self, tokens: Any, logits: mx.array) -> mx.array:
        prefix = self._decode_history(tokens)
        if structured_cot_complete(prefix, self.mode):
            return logits

        allowed = self._allowed_token_ids(prefix)
        if not allowed:
            return logits

        mask = mx.full(logits.shape, -float("inf"))
        allowed_idx = mx.array(allowed, dtype=mx.int32)
        values = mx.take(logits[0], allowed_idx)
        mask = mx.put_along_axis(mask, allowed_idx[None, :], values[None, :], axis=-1)
        return mask
