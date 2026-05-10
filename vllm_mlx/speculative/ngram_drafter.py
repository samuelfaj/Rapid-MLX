# SPDX-License-Identifier: Apache-2.0
"""
Per-request n-gram speculative draft state, gated by `<think>` block.

This module wires `PromptLookupDecoder` into the MTP scheduler path with
two extras: a token-level state machine that tracks whether generation is
currently inside a `<think>...</think>` block, and a uid-keyed registry
managed by the scheduler so each request keeps its own n-gram history.

When layered on top of native MTP the drafter runs first; if it finds a
match it returns up to K candidate tokens to verify in a single forward
pass, falling back to MTP-head draft when there is no match.
"""

from __future__ import annotations

import logging

from .prompt_lookup import PromptLookupDecoder

logger = logging.getLogger(__name__)


class ThinkStateTracker:
    """Track `<think>...</think>` nesting via raw token IDs.

    We only watch for two tokens. Any token that is neither sets nothing
    so the tracker is cheap to call on every emitted token.
    """

    def __init__(
        self, think_start_id: int | None, think_end_id: int | None
    ) -> None:
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.thinking = False

    def feed(self, token_id: int) -> None:
        if self.think_start_id is not None and token_id == self.think_start_id:
            self.thinking = True
        elif self.think_end_id is not None and token_id == self.think_end_id:
            self.thinking = False

    def feed_many(self, token_ids) -> None:
        for t in token_ids:
            self.feed(int(t))


class NgramRequestState:
    """Per-request n-gram drafter + think-block gate.

    The decoder is seeded with the prompt tokens so first-step lookups
    can hit prefix patterns. The tracker is also fed the prompt so
    initial `thinking` state matches the chat template (Qwen reasoning
    models inject `<think>` in the assistant prefix → tracker starts in
    thinking=True, ready to draft on the very first generated token).
    """

    def __init__(
        self,
        prompt_tokens,
        think_start_id: int | None,
        think_end_id: int | None,
        num_draft_tokens: int = 4,
        ngram_size: int = 3,
        min_matches: int = 2,
        only_in_think: bool = True,
    ) -> None:
        self.decoder = PromptLookupDecoder(
            num_draft_tokens=num_draft_tokens,
            ngram_size=ngram_size,
            min_matches=min_matches,
        )
        self.tracker = ThinkStateTracker(think_start_id, think_end_id)
        self.only_in_think = only_in_think

        prompt_list = [int(t) for t in prompt_tokens]
        self.decoder.add_prompt_tokens(prompt_list)
        self.tracker.feed_many(prompt_list)

        # Stats — exposed via get_stats() for aggregate logging.
        self.drafts_attempted = 0
        self.tokens_drafted = 0
        self.tokens_accepted = 0

    @property
    def thinking(self) -> bool:
        return self.tracker.thinking

    def feed_token(self, token_id: int) -> None:
        """Add an emitted token to history; updates decoder + tracker."""
        tid = int(token_id)
        self.decoder.add_generated_token(tid)
        self.tracker.feed(tid)

    def feed_many(self, token_ids) -> None:
        for t in token_ids:
            self.feed_token(t)

    def should_draft(self) -> bool:
        if not self.only_in_think:
            return True
        return self.tracker.thinking

    def get_drafts(self) -> list[int]:
        """Return up to num_draft_tokens candidates, or [] when no match.

        The decoder enforces its own `min_matches` threshold internally,
        so this returns either a usable list or empty.
        """
        return self.decoder.get_draft_tokens()

    def lookup_drafts_with_pending(self, pending_token: int) -> list[int]:
        """Return drafts treating ``pending_token`` as a virtual tail token.

        Used at draft time when the just-sampled primary has not yet been
        committed to history but should still complete the n-gram query
        (so drafts predict positions AFTER the primary, not at it).
        Does not mutate decoder/tracker state.
        """
        decoder = self.decoder
        history = decoder._token_history
        n = decoder.ngram_size
        if len(history) + 1 < n:
            return []
        if n == 1:
            query = (int(pending_token),)
        else:
            query = tuple(history[-(n - 1):]) + (int(pending_token),)

        positions = decoder._ngram_index.get(query, [])
        if not positions:
            return []
        # The virtual occurrence of the query starts at len(history) + 1 - n
        # (one past the real history end). Skip any position that equals the
        # virtual current_start (cannot happen here since no real n-gram has
        # been indexed yet for the query, but kept for parity with
        # PromptLookupDecoder.get_draft_tokens).
        current_start_virtual = len(history) + 1 - n
        K = decoder.num_draft_tokens
        drafts: list[int] = []
        best_len = 0
        for start in positions:
            if start == current_start_virtual:
                continue
            cont_begin = start + n
            cont_end = min(cont_begin + K, len(history))
            cont = history[cont_begin:cont_end]
            if len(cont) > best_len:
                best_len = len(cont)
                drafts = list(cont[:K])
        if len(drafts) >= decoder.min_matches:
            decoder.total_drafts += 1
            decoder.total_draft_tokens += len(drafts)
            return drafts
        return []

    def record_outcome(self, drafted: int, accepted: int) -> None:
        if drafted <= 0:
            return
        self.drafts_attempted += 1
        self.tokens_drafted += drafted
        self.tokens_accepted += accepted

    def get_stats(self) -> dict:
        rate = (
            self.tokens_accepted / self.tokens_drafted
            if self.tokens_drafted > 0
            else 0.0
        )
        return {
            "drafts_attempted": self.drafts_attempted,
            "tokens_drafted": self.tokens_drafted,
            "tokens_accepted": self.tokens_accepted,
            "acceptance_rate": rate,
        }


def lookup_think_token_ids(tokenizer) -> tuple[int | None, int | None]:
    """Resolve `<think>` and `</think>` token IDs from a tokenizer vocab.

    Returns (None, None) if the model doesn't have these tokens. In that
    case n-gram-only-in-think gating cannot work and the caller should
    skip ngram setup (or set only_in_think=False).
    """
    actual = tokenizer
    if hasattr(actual, "tokenizer") and not hasattr(actual, "get_vocab"):
        actual = actual.tokenizer
    try:
        vocab = actual.get_vocab()
    except Exception:
        return (None, None)
    return vocab.get("<think>"), vocab.get("</think>")
