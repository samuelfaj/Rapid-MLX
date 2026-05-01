from types import SimpleNamespace

import pytest

from vllm_mlx.cli import _finalize_tool_call_args


def test_tool_call_parser_implies_auto_tool_choice():
    args = SimpleNamespace(
        enable_auto_tool_choice=False,
        tool_call_parser="qwen3_coder_xml",
    )

    _finalize_tool_call_args(args)

    assert args.enable_auto_tool_choice is True


def test_auto_tool_choice_still_requires_parser():
    args = SimpleNamespace(
        enable_auto_tool_choice=True,
        tool_call_parser=None,
    )

    with pytest.raises(SystemExit):
        _finalize_tool_call_args(args)
