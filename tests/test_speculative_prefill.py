# SPDX-License-Identifier: Apache-2.0
"""Tests for speculative prefill prompt compression."""

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.speculative.prefill import (
    SpeculativePrefillCompressor,
    SpeculativePrefillConfig,
)


class WordTokenizer:
    def encode(self, text):
        return list(range(len(text.split())))

    def decode(self, tokens):
        return " ".join(f"tok{token}" for token in tokens)


class StableWordTokenizer:
    def __init__(self):
        self._ids = {}
        self._pieces = {}

    def encode(self, text):
        ids = []
        for piece in text.split():
            if piece not in self._ids:
                token_id = len(self._ids)
                self._ids[piece] = token_id
                self._pieces[token_id] = piece
            ids.append(self._ids[piece])
        return ids

    def decode(self, tokens):
        return " ".join(self._pieces[token] for token in tokens)


def test_speculative_prefill_disabled_is_noop():
    compressor = SpeculativePrefillCompressor(SpeculativePrefillConfig(enabled=False))
    result = compressor.compress("alpha beta gamma", WordTokenizer())

    assert result.prompt == "alpha beta gamma"
    assert result.applied is False
    assert result.reason == "disabled"


def test_speculative_prefill_compresses_with_importance_scores():
    tokenizer = StableWordTokenizer()

    def score(tokens, _tokenizer):
        return [10.0 if idx in {0, 2, 5, 9} else 0.1 for idx, _ in enumerate(tokens)]

    compressor = SpeculativePrefillCompressor(
        SpeculativePrefillConfig(
            enabled=True,
            target_token_ratio=0.5,
            min_prompt_tokens=4,
            preserve_first_tokens=1,
            preserve_last_tokens=1,
        ),
        score_fn=score,
    )

    result = compressor.compress(
        "keep filler keep filler filler keep filler filler filler keep",
        tokenizer,
    )

    assert result.applied is True
    assert result.compressed_tokens < result.original_tokens
    assert result.tokens_saved > 0


def test_speculative_prefill_preserves_protected_anchors():
    tokenizer = StableWordTokenizer()
    compressor = SpeculativePrefillCompressor(
        SpeculativePrefillConfig(
            enabled=True,
            target_token_ratio=0.5,
            min_prompt_tokens=4,
            preserve_first_tokens=1,
            preserve_last_tokens=1,
        ),
        score_fn=lambda tokens, _tokenizer: [0.1] * len(tokens),
    )

    result = compressor.compress(
        "start filler filler https://example.com/api filler 42 filler end",
        tokenizer,
    )

    assert "https://example.com/api" in result.prompt
    assert "42" in result.prompt


def test_batched_engine_compresses_only_last_user_message():
    tokenizer = StableWordTokenizer()
    engine = BatchedEngine(
        "dummy",
        speculative_prefill_config=SpeculativePrefillConfig(
            enabled=True,
            target_token_ratio=0.5,
            min_prompt_tokens=4,
            preserve_first_tokens=1,
            preserve_last_tokens=1,
        ),
    )
    engine._tokenizer = tokenizer
    engine._speculative_prefill._score_fn = lambda tokens, _tokenizer: [
        10.0 if idx in {0, 7} else 0.1 for idx, _ in enumerate(tokens)
    ]

    messages = [
        {"role": "system", "content": "system text should stay"},
        {"role": "user", "content": "alpha beta gamma delta epsilon zeta eta omega"},
    ]

    compressed = engine._maybe_compress_last_user_message(messages)

    assert compressed[0]["content"] == "system text should stay"
    assert compressed[1]["content"] != messages[1]["content"]


def test_batched_engine_does_not_compress_tool_requests_by_default():
    engine = BatchedEngine(
        "dummy",
        speculative_prefill_config=SpeculativePrefillConfig(
            enabled=True,
            target_token_ratio=0.5,
            min_prompt_tokens=4,
            preserve_first_tokens=1,
            preserve_last_tokens=1,
        ),
    )
    engine._tokenizer = StableWordTokenizer()
    engine._speculative_prefill._score_fn = lambda tokens, _tokenizer: [0.1] * len(
        tokens
    )
    messages = [
        {"role": "user", "content": "alpha beta gamma delta epsilon zeta eta omega"},
    ]

    compressed = engine._maybe_compress_last_user_message(
        messages, tools_requested=True
    )

    assert compressed == messages
