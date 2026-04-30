# SPDX-License-Identifier: Apache-2.0
"""Structured chain-of-thought constrained decoding.

This implements the local equivalent of the structured-cot GBNF constraint:
the thinking prelude must be a terse, one-line-per-field plan, then the answer
channel becomes unconstrained again.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np


STRUCTURED_COT_LABELS = ("GOAL: ", "APPROACH: ", "EDGE: ")
STRUCTURED_COT_END = "</think>\n\n"


def prompt_thinking_prefix(prompt: str) -> str:
    """Return literal still expected before GOAL based on prompt state."""
    last_open = prompt.rfind("<think>")
    last_close = prompt.rfind("</think>")
    if last_open <= last_close:
        return "<think>\n"

    after_open = prompt[last_open + len("<think>") :]
    if after_open.endswith("\n"):
        return ""
    return "\n"


class StructuredCoTLogitsProcessor:
    """Mask logits so generated thinking follows GOAL/APPROACH/EDGE."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        prefix: str = "<think>\n",
        token_budget: int = 256,
        labels: tuple[str, ...] = STRUCTURED_COT_LABELS,
        prompt_token_count: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.labels = labels
        self.prompt_token_count = max(0, int(prompt_token_count))
        self.max_line_chars = max(24, min(192, int(token_budget) * 4 // len(labels)))
        self._token_texts: list[str] = []
        self._mask_cache: dict[tuple[str, int], mx.array] = {}

    def __call__(self, tokens: Any, logits: mx.array) -> mx.array:
        generated = self._decode_tokens(tokens)
        if self._is_done(generated):
            return logits

        vocab_size = int(logits.shape[-1])
        self._ensure_token_texts(vocab_size)
        key = (generated, vocab_size)
        mask = self._mask_cache.get(key)
        if mask is None:
            allowed = [
                token_id
                for token_id, token_text in enumerate(self._token_texts[:vocab_size])
                if token_text and self._is_valid_partial(generated + token_text)
            ]
            if not allowed:
                return logits
            mask_np = np.full((vocab_size,), -1e9, dtype=np.float32)
            mask_np[allowed] = 0.0
            mask = mx.array(mask_np)[None, :]
            self._mask_cache[key] = mask
        return logits + mask.astype(logits.dtype)

    def mask_logits_for_tokens(self, generated_tokens: list[int], logits: mx.array) -> mx.array:
        """Apply the same constraint when caller owns the greedy decode loop."""
        return self(generated_tokens, logits)

    def _decode_tokens(self, tokens: Any) -> str:
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, int):
            token_ids = [tokens]
        else:
            token_ids = [int(token) for token in (tokens or [])]
        if self.prompt_token_count:
            token_ids = token_ids[self.prompt_token_count :]
        if not token_ids:
            return ""
        try:
            return self.tokenizer.decode(token_ids)
        except TypeError:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _ensure_token_texts(self, vocab_size: int) -> None:
        start = len(self._token_texts)
        if start >= vocab_size:
            return
        for token_id in range(start, vocab_size):
            try:
                text = self.tokenizer.decode([token_id])
            except TypeError:
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                text = ""
            self._token_texts.append(text)

    def _is_done(self, text: str) -> bool:
        return self._structured_end_index(text) is not None

    def _structured_end_index(self, text: str) -> int | None:
        start = len(self.prefix)
        if not text.startswith(self.prefix):
            return None
        pos = start
        for label in self.labels:
            if not text.startswith(label, pos):
                return None
            pos += len(label)
            newline = text.find("\n", pos)
            if newline < 0:
                return None
            if newline == pos:
                return None
            pos = newline + 1
        if text.startswith(STRUCTURED_COT_END, pos):
            return pos + len(STRUCTURED_COT_END)
        return None

    def _is_valid_partial(self, text: str) -> bool:
        end_index = self._structured_end_index(text)
        if end_index is not None:
            return True

        if not self.prefix.startswith(text[: len(self.prefix)]):
            return False
        if len(text) < len(self.prefix):
            return True

        pos = len(self.prefix)
        for label in self.labels:
            remaining = text[pos:]
            if len(remaining) < len(label):
                return label.startswith(remaining)
            if not remaining.startswith(label):
                return False
            pos += len(label)

            newline = text.find("\n", pos)
            if newline < 0:
                content = text[pos:]
                return (
                    0 <= len(content) <= self.max_line_chars
                    and "</think>" not in content
                )
            if newline == pos or newline - pos > self.max_line_chars:
                return False
            content = text[pos:newline]
            if "</think>" in content:
                return False
            pos = newline + 1

        remaining = text[pos:]
        return STRUCTURED_COT_END.startswith(remaining)


def create_structured_cot_logits_processor(
    tokenizer: Any,
    *,
    prompt: str = "",
    token_budget: int = 256,
    prompt_token_count: int = 0,
) -> StructuredCoTLogitsProcessor:
    return StructuredCoTLogitsProcessor(
        tokenizer,
        prefix=prompt_thinking_prefix(prompt),
        token_budget=token_budget,
        prompt_token_count=prompt_token_count,
    )
