# SPDX-License-Identifier: Apache-2.0
"""Focused tests for server-wired n-gram speculative decoding."""

from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx

from vllm_mlx.scheduler import _install_ngram_mod


class FakeCache:
    def __init__(self):
        self.trimmed: list[int] = []

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trimmed.append(n)
        return n

    def extract(self, _idx):
        return self


class FakeUntrimmableCache:
    def __init__(self):
        self.seen: list[int] = []

    @property
    def state(self):
        return list(self.seen)

    @state.setter
    def state(self, value):
        self.seen = list(value)

    def is_trimmable(self):
        return False

    def extract(self, _idx):
        return self


class CopyableState:
    def __init__(self, values=None):
        self.values = list(values or [])

    def copy(self):
        return CopyableState(self.values)


class FakeCopyableStateCache:
    def __init__(self):
        self.holder = CopyableState()

    @property
    def state(self):
        return [self.holder]

    @state.setter
    def state(self, value):
        self.holder = value[0]

    @property
    def seen(self):
        return self.holder.values

    def is_trimmable(self):
        return False

    def extract(self, _idx):
        return self


class FakeModel:
    def __call__(self, input_tokens, cache=None):
        # For verify input [3, 4, 1, 2], target samples [4, 1, 9, 0].
        if cache:
            for cache_layer in cache:
                if hasattr(cache_layer, "seen"):
                    cache_layer.seen.extend(input_tokens[0].tolist())
        samples = [4, 1, 9, 0]
        logits = mx.full((1, input_tokens.shape[1], 12), -1000.0)
        for pos in range(input_tokens.shape[1]):
            logits[:, pos, samples[pos]] = 0.0
        return logits


class FakeBatch:
    def __init__(self, cache):
        self.uids = [7]
        self.y = mx.array([3], mx.uint32)
        self.logprobs = [mx.zeros((12,))]
        self.max_tokens = [20]
        self.num_tokens = [0]
        self.cache = [cache]
        self.samplers = [None]
        self.logits_processors = [[]]
        self.tokens = [mx.array([1, 2, 3, 4, 1, 2], mx.uint32)]

    def __len__(self):
        return len(self.uids)

    def extract_cache(self, idx):
        return [c.extract(idx) for c in self.cache]


@dataclass
class FakeResponse:
    uid: int
    token: int
    logprobs: mx.array
    finish_reason: str | None
    prompt_cache: list | None


class FakeBatchGenerator:
    Response = FakeResponse

    def __init__(self):
        self.cache = FakeCache()
        self.active_batch = FakeBatch(self.cache)
        self.unprocessed_prompts = []
        self.stop_tokens = set()
        self.sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
        self.model = FakeModel()
        self._stats = SimpleNamespace(generation_time=0.0, generation_tokens=0)

    def _next(self):
        raise AssertionError("ngram path should not fall back")

    def remove(self, _uids, return_prompt_caches=False):
        return {} if return_prompt_caches else None


class FakeFallbackBatchGenerator(FakeBatchGenerator):
    def _next(self):
        batch = self.active_batch
        token = batch.y[0].item()
        batch.tokens[0] = mx.concatenate((batch.tokens[0], batch.y))
        batch.y = mx.array([8], mx.uint32)
        batch.num_tokens[0] += 1
        return [self.Response(batch.uids[0], token, batch.logprobs[0], None, None)]


class FakeToolTokenizer:
    def decode(self, tokens, *args, **kwargs):
        if 4 in tokens:
            return "<tool_call><function=bash>"
        return "plain text"


def test_ngram_mod_accepts_prefix_and_trims_rejected_suffix():
    bg = FakeBatchGenerator()
    stats = _install_ngram_mod(
        bg,
        num_draft_tokens=3,
        ngram_size=3,
        min_matches=1,
    )

    responses = bg._next()

    assert [r.token for r in responses] == [3, 4, 1]
    assert [r.finish_reason for r in responses] == [None, None, None]
    assert bg.cache.trimmed == [1]
    assert bg.active_batch.y.tolist() == [9]
    assert bg.active_batch.num_tokens == [3]
    assert bg.active_batch.tokens[0].tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 1]
    assert stats["proposed_tokens"] == 3
    assert stats["accepted_tokens"] == 2
    assert stats["rejected_tokens"] == 1


def test_ngram_mod_replays_accepted_tokens_for_untrimmable_cache():
    bg = FakeBatchGenerator()
    bg.cache = FakeUntrimmableCache()
    bg.active_batch.cache = [bg.cache]
    stats = _install_ngram_mod(
        bg,
        num_draft_tokens=3,
        ngram_size=3,
        min_matches=1,
    )

    responses = bg._next()

    assert [r.token for r in responses] == [3, 4, 1]
    assert bg.cache.seen == [3, 4, 1]
    assert bg.active_batch.y.tolist() == [9]
    assert stats["attempts"] == 1
    assert stats["accepted_tokens"] == 2
    assert stats["rejected_tokens"] == 1


def test_ngram_mod_snapshots_copyable_cache_state_before_verify():
    bg = FakeBatchGenerator()
    bg.cache = FakeCopyableStateCache()
    bg.active_batch.cache = [bg.cache]
    _install_ngram_mod(
        bg,
        num_draft_tokens=3,
        ngram_size=3,
        min_matches=1,
    )

    responses = bg._next()

    assert [r.token for r in responses] == [3, 4, 1]
    assert bg.cache.seen == [3, 4, 1]


def test_ngram_mod_rewinds_trimmable_layers_before_replay():
    bg = FakeBatchGenerator()
    kv_cache = FakeCache()
    state_cache = FakeCopyableStateCache()
    bg.active_batch.cache = [kv_cache, state_cache]
    _install_ngram_mod(
        bg,
        num_draft_tokens=3,
        ngram_size=3,
        min_matches=1,
    )

    responses = bg._next()

    assert [r.token for r in responses] == [3, 4, 1]
    assert kv_cache.trimmed == [4]
    assert state_cache.seen == [3, 4, 1]


def test_ngram_mod_falls_back_for_tool_call_drafts():
    bg = FakeFallbackBatchGenerator()
    stats = _install_ngram_mod(
        bg,
        num_draft_tokens=3,
        ngram_size=3,
        min_matches=1,
        tokenizer=FakeToolTokenizer(),
    )

    responses = bg._next()

    assert [r.token for r in responses] == [3]
    assert bg.active_batch.y.tolist() == [8]
    assert bg.active_batch.num_tokens == [1]
    assert stats["attempts"] == 0
    assert stats["tool_guard_steps"] == 1
    assert stats["proposed_tokens"] == 0


def test_ngram_mod_finishes_batch_on_speculative_length_stop():
    bg = FakeBatchGenerator()
    bg.active_batch.max_tokens = [3]
    _install_ngram_mod(
        bg,
        num_draft_tokens=3,
        ngram_size=3,
        min_matches=1,
    )

    responses = bg._next()

    assert [r.token for r in responses] == [3, 4, 1]
    assert [r.finish_reason for r in responses] == [None, None, "length"]
    assert bg.active_batch is None
