# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-request n-gram drafter and think-block gating."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm_mlx.scheduler import Scheduler, SchedulerConfig
from vllm_mlx.speculative.ngram_drafter import (
    NgramRequestState,
    ThinkStateTracker,
    lookup_think_token_ids,
)


THINK_START = 100
THINK_END = 101


def _make_state(prompt, **kwargs):
    defaults = dict(
        think_start_id=THINK_START,
        think_end_id=THINK_END,
        num_draft_tokens=4,
        ngram_size=3,
        min_matches=2,
        only_in_think=True,
    )
    defaults.update(kwargs)
    return NgramRequestState(prompt_tokens=prompt, **defaults)


class TestThinkStateTracker:
    def test_starts_outside_think(self):
        t = ThinkStateTracker(THINK_START, THINK_END)
        assert t.thinking is False

    def test_enters_on_think_start(self):
        t = ThinkStateTracker(THINK_START, THINK_END)
        t.feed(THINK_START)
        assert t.thinking is True

    def test_exits_on_think_end(self):
        t = ThinkStateTracker(THINK_START, THINK_END)
        t.feed(THINK_START)
        t.feed(7)
        t.feed(THINK_END)
        assert t.thinking is False

    def test_unknown_tokens_dont_change_state(self):
        t = ThinkStateTracker(THINK_START, THINK_END)
        t.feed(7)
        t.feed(8)
        assert t.thinking is False
        t.feed(THINK_START)
        t.feed(7)
        assert t.thinking is True

    def test_no_think_tokens_means_static_state(self):
        t = ThinkStateTracker(None, None)
        t.feed(THINK_START)
        t.feed(THINK_END)
        assert t.thinking is False


class TestNgramRequestState:
    def test_initial_state_inside_think_when_prompt_ends_in_think(self):
        # Qwen-style prompt: <think> tag at the end of the assistant prefix
        prompt = [1, 2, 3, 4, 5, THINK_START]
        s = _make_state(prompt)
        assert s.thinking is True
        assert s.should_draft() is True

    def test_initial_state_outside_think_for_plain_prompt(self):
        prompt = [1, 2, 3, 4, 5]
        s = _make_state(prompt)
        assert s.thinking is False
        assert s.should_draft() is False

    def test_drafts_returned_for_repeating_pattern(self):
        prompt = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, THINK_START]
        s = _make_state(prompt)
        # Re-establish (1,2,3) trigram by feeding three tokens
        s.feed_token(1)
        s.feed_token(2)
        s.feed_token(3)
        drafts = s.get_drafts()
        # Trigram (1,2,3) appears earlier with continuation 4,5,6,1
        assert drafts == [4, 5, 6, 1]

    def test_drafts_empty_when_no_match(self):
        prompt = [10, 20, 30, 40, 50]
        s = _make_state(prompt, only_in_think=False)
        # Last 3-gram is (30,40,50) — no earlier occurrence → no drafts
        assert s.get_drafts() == []

    def test_only_in_think_gate_blocks_outside(self):
        prompt = [1, 2, 3, 4, 5, 6, 1, 2, 3]
        s = _make_state(prompt, only_in_think=True)
        # Tracker is outside think — even with a match, should_draft() is False
        assert s.should_draft() is False

    def test_only_in_think_off_drafts_anywhere(self):
        prompt = [1, 2, 3, 4, 5, 6, 1, 2, 3]
        s = _make_state(prompt, only_in_think=False)
        assert s.should_draft() is True
        # Trigram (1,2,3) at position 0 has continuation 4,5,6,1
        # (drafter takes up to num_draft_tokens=4 following tokens)
        assert s.get_drafts() == [4, 5, 6, 1]

    def test_record_outcome_tracks_acceptance(self):
        prompt = [1, 2, 3]
        s = _make_state(prompt, only_in_think=False)
        s.record_outcome(drafted=4, accepted=3)
        s.record_outcome(drafted=4, accepted=4)
        stats = s.get_stats()
        assert stats["drafts_attempted"] == 2
        assert stats["tokens_drafted"] == 8
        assert stats["tokens_accepted"] == 7
        assert stats["acceptance_rate"] == pytest.approx(7 / 8)

    def test_feed_token_updates_both_tracker_and_decoder(self):
        prompt = [1, 2, 3]
        s = _make_state(prompt, only_in_think=True)
        # Start outside think
        assert s.thinking is False
        s.feed_token(THINK_START)
        assert s.thinking is True
        # Feeding more tokens grows decoder history (no exception)
        s.feed_token(4)
        s.feed_token(5)
        assert s.thinking is True

    def test_token_outside_think_does_not_draft_after_close(self):
        prompt = [1, 2, 3, 4, 5, 6, 1, 2, 3, THINK_START]
        s = _make_state(prompt, only_in_think=True)
        assert s.should_draft() is True
        s.feed_token(THINK_END)
        # After </think>, drafting is gated off even though n-gram match exists
        assert s.should_draft() is False

    def test_lookup_with_pending_finds_match(self):
        # History: prompt=[1,2,3,4,5,6], then feed 1,2.
        # Trigram (1,2,3) is at position 0; continuation = history[3:3+K] (K=4).
        # history after feeds = [1,2,3,4,5,6,1,2] → continuation = [4,5,6,1]
        prompt = [1, 2, 3, 4, 5, 6]
        s = _make_state(prompt, only_in_think=False)
        s.feed_token(1)
        s.feed_token(2)
        drafts = s.lookup_drafts_with_pending(3)
        assert drafts == [4, 5, 6, 1]
        # State is unchanged (no commit) — last appended was 2, not 3
        assert s.decoder._token_history[-1] == 2

    def test_lookup_with_pending_returns_empty_when_no_match(self):
        prompt = [1, 2, 3, 4, 5, 6]
        s = _make_state(prompt, only_in_think=False)
        s.feed_token(99)
        # Query (1,99,X) doesn't appear → empty
        assert s.lookup_drafts_with_pending(50) == []

    def test_lookup_with_pending_does_not_mutate_state(self):
        prompt = [1, 2, 3, 4, 5, 6]
        s = _make_state(prompt, only_in_think=False)
        s.feed_token(1)
        s.feed_token(2)
        history_before = list(s.decoder._token_history)
        index_size_before = len(s.decoder._ngram_index)
        s.lookup_drafts_with_pending(3)
        s.lookup_drafts_with_pending(99)
        s.lookup_drafts_with_pending(3)
        # No tokens added; no n-gram index entries created
        assert s.decoder._token_history == history_before
        assert len(s.decoder._ngram_index) == index_size_before


class TestLookupThinkTokenIds:
    def test_returns_ids_when_present(self):
        tok = MagicMock()
        tok.get_vocab = lambda: {
            "<think>": THINK_START,
            "</think>": THINK_END,
            "x": 5,
        }
        assert lookup_think_token_ids(tok) == (THINK_START, THINK_END)

    def test_returns_none_when_absent(self):
        tok = MagicMock()
        tok.get_vocab = lambda: {"x": 5}
        assert lookup_think_token_ids(tok) == (None, None)

    def test_handles_processor_wrapper(self):
        inner = MagicMock()
        inner.get_vocab = lambda: {
            "<think>": THINK_START,
            "</think>": THINK_END,
        }
        wrapper = MagicMock(spec=["tokenizer"])
        wrapper.tokenizer = inner
        # Wrapper has no get_vocab, must descend into .tokenizer
        assert lookup_think_token_ids(wrapper) == (THINK_START, THINK_END)


class TestSchedulerNgramWiring:
    """Integration-light: verify Scheduler init wires the drafter registry."""

    def _make_tokenizer(self, *, with_think: bool = True) -> MagicMock:
        tok = MagicMock()
        vocab = {"<eos>": 0, "x": 5}
        if with_think:
            vocab.update({"<think>": THINK_START, "</think>": THINK_END})
        tok.get_vocab = lambda: vocab
        tok.encode = lambda s: [1, 2, 3]
        tok.decode = lambda ids: "x"
        tok.eos_token_id = 0
        tok.eos_token_ids = None
        return tok

    def test_scheduler_init_resolves_think_tokens(self):
        tok = self._make_tokenizer(with_think=True)
        sch = Scheduler(
            model=MagicMock(mtp=None),
            tokenizer=tok,
            config=SchedulerConfig(
                enable_ngram=True,
                enable_prefix_cache=False,
            ),
        )
        assert sch._ngram_think_start_id == THINK_START
        assert sch._ngram_think_end_id == THINK_END
        assert sch._ngram_drafters == {}

    def test_scheduler_init_warns_when_think_tokens_missing(self, caplog):
        tok = self._make_tokenizer(with_think=False)
        with caplog.at_level("WARNING"):
            sch = Scheduler(
                model=MagicMock(mtp=None),
                tokenizer=tok,
                config=SchedulerConfig(
                    enable_ngram=True,
                    ngram_only_in_think=True,
                    enable_prefix_cache=False,
                ),
            )
        assert sch._ngram_think_start_id is None
        assert sch._ngram_think_end_id is None
        assert any(
            "no <think>/</think> tokens" in r.message for r in caplog.records
        )

    def test_disabled_ngram_skips_lookup(self):
        tok = self._make_tokenizer(with_think=False)
        sch = Scheduler(
            model=MagicMock(mtp=None),
            tokenizer=tok,
            config=SchedulerConfig(
                enable_ngram=False,
                enable_prefix_cache=False,
            ),
        )
        assert sch._ngram_drafters == {}
        assert sch._ngram_think_start_id is None
        assert sch._ngram_think_end_id is None

    def test_get_stats_exposes_ngram_counters(self):
        tok = self._make_tokenizer(with_think=True)
        sch = Scheduler(
            model=MagicMock(mtp=None),
            tokenizer=tok,
            config=SchedulerConfig(
                enable_ngram=True,
                enable_mtp=False,
                enable_prefix_cache=False,
            ),
        )
        # Simulate a few ngram cycles via the global stats dict
        sch._mtp_global_stats["ngram_drafts"] = 3
        sch._mtp_global_stats["ngram_tokens_drafted"] = 12
        sch._mtp_global_stats["ngram_tokens_accepted"] = 9

        stats = sch.get_stats()
        mtp = stats["mtp"]
        assert mtp["ngram_drafts"] == 3
        assert mtp["ngram_tokens_drafted"] == 12
        assert mtp["ngram_tokens_accepted"] == 9
        assert mtp["ngram_acceptance_ratio"] == pytest.approx(9 / 12)

    def test_get_stats_no_ngram_returns_none_ratio(self):
        tok = self._make_tokenizer(with_think=True)
        sch = Scheduler(
            model=MagicMock(mtp=None),
            tokenizer=tok,
            config=SchedulerConfig(
                enable_ngram=False,
                enable_mtp=False,
                enable_prefix_cache=False,
            ),
        )
        stats = sch.get_stats()
        assert stats["mtp"]["ngram_acceptance_ratio"] is None
        assert stats["mtp"]["ngram_enabled"] is False


class TestRecorderNgramFields:
    def test_recorder_records_ngram_acceptance(self):
        from vllm_mlx.request_metrics import RequestRecorder

        rec = RequestRecorder()
        rid = rec.start("/v1/chat/completions")
        rec.update(rid, generated_tokens=100, prompt_tokens=50)
        rec.finish(
            rid,
            finish_reason="stop",
            generated_tokens=100,
            prompt_tokens=50,
            ngram_enabled=True,
            ngram_drafts=10,
            ngram_tokens_drafted=40,
            ngram_tokens_accepted=32,
        )
        last = rec.last()
        assert last is not None
        assert last["ngram_enabled"] is True
        assert last["ngram_cycles"] == 10
        assert last["ngram_tokens_drafted"] == 40
        assert last["ngram_tokens_accepted"] == 32
        assert last["ngram_acceptance_ratio"] == pytest.approx(32 / 40)
        # Reused speculative_* keys feed _spec_path/_avg_accept_tokens
        assert last["speculative_proposed_tokens"] == 40
        assert last["speculative_accepted_tokens"] == 32

    def test_recorder_no_ngram_returns_none_ratio(self):
        from vllm_mlx.request_metrics import RequestRecorder

        rec = RequestRecorder()
        rid = rec.start("/v1/chat/completions")
        rec.finish(rid, finish_reason="stop", generated_tokens=10, prompt_tokens=5)
        last = rec.last()
        assert last is not None
        assert last["ngram_enabled"] is False
        assert last["ngram_cycles"] == 0
        assert last["ngram_acceptance_ratio"] is None
