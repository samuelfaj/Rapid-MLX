# SPDX-License-Identifier: Apache-2.0
"""Tests for service helper resolution logic."""

import pytest

from vllm_mlx.config import reset_config
from vllm_mlx.service.helpers import (
    _STRUCTURED_COT_SUFFIX,
    _STRUCTURED_COT_TOOL_SUFFIX,
    _resolve_enable_thinking,
    _resolve_max_tokens,
    _structured_cot_suffix,
)


@pytest.fixture(autouse=True)
def _fresh_config():
    reset_config()
    yield
    reset_config()


def test_resolve_max_tokens_adds_reasoning_budget_only_when_thinking_enabled():
    cfg = reset_config()
    cfg.reasoning_parser_name = "qwen3"
    cfg.thinking_token_budget = 2048

    assert _resolve_max_tokens(32) == 2080
    assert _resolve_max_tokens(32, enable_thinking=False) == 32


def test_resolve_max_tokens_uses_short_structured_cot_budget():
    cfg = reset_config()
    cfg.reasoning_parser_name = "qwen3"
    cfg.structured_cot = True
    cfg.structured_cot_token_budget = 128

    assert _resolve_max_tokens(32, enable_thinking=True) == 160


def test_resolve_max_tokens_respects_no_thinking_flag():
    cfg = reset_config()
    cfg.reasoning_parser_name = "qwen3"
    cfg.no_thinking = True

    assert _resolve_max_tokens(32) == 32


def test_qwen3_tool_calls_disable_thinking_by_default():
    cfg = reset_config()
    cfg.tool_call_parser = "qwen3_coder_xml"

    assert _resolve_enable_thinking(None, tools_requested=True) is False
    assert _resolve_enable_thinking(True, tools_requested=True) is True


def test_structured_cot_tool_calls_keep_qwen3_template_thinking_closed():
    cfg = reset_config()
    cfg.tool_call_parser = "qwen3_coder_xml"
    cfg.structured_cot = True
    cfg.structured_cot_tools = True

    assert _resolve_enable_thinking(None, tools_requested=True) is False


def test_structured_cot_non_tool_requests_enable_thinking():
    cfg = reset_config()
    cfg.tool_call_parser = "qwen3_coder_xml"
    cfg.structured_cot = True

    assert _resolve_enable_thinking(None, tools_requested=False) is True


def test_structured_cot_tool_calls_respect_no_thinking():
    cfg = reset_config()
    cfg.tool_call_parser = "qwen3_coder_xml"
    cfg.structured_cot = True
    cfg.structured_cot_tools = True
    cfg.no_thinking = True

    assert _resolve_enable_thinking(None, tools_requested=True) is False


def test_non_qwen_tool_calls_keep_template_default():
    cfg = reset_config()
    cfg.tool_call_parser = "hermes"

    assert _resolve_enable_thinking(None, tools_requested=True) is None


def test_structured_cot_suffix_respects_tool_gate():
    cfg = reset_config()
    cfg.structured_cot = True

    assert _structured_cot_suffix(tools_requested=False) == _STRUCTURED_COT_SUFFIX
    assert _structured_cot_suffix(tools_requested=True) is None

    cfg.structured_cot_tools = True
    assert _structured_cot_suffix(tools_requested=True) == _STRUCTURED_COT_TOOL_SUFFIX
