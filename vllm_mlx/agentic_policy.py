# SPDX-License-Identifier: Apache-2.0
"""Lightweight agentic speculative policy helpers."""

from __future__ import annotations

from typing import Any

VALIDATION_MARKERS = (
    "bun test",
    "bun run build",
    "npm test",
    "pytest",
    "tsc",
    "validation",
)
FAILURE_MARKERS = (
    "error",
    "failed",
    "fail",
    "traceback",
    "exception",
    "not found",
    "cannot find",
)
SUCCESS_MARKERS = (
    " 0 fail",
    "0 fail",
    "tests passed",
    "all checks passed",
    "exit\":0",
    "exit:0",
)


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content") or ""
    else:
        content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    return str(content)


def classify_agentic_phase(messages: list[Any], tools_requested: bool) -> str:
    """Classify a tool workflow phase from local transcript signals."""

    if not tools_requested:
        return "long_text_or_code"

    tool_messages = [
        message for message in messages if _message_role(message) == "tool"
    ]
    assistant_tool_calls = [
        message
        for message in messages
        if _message_role(message) == "assistant"
        and (
            bool(message.get("tool_calls")) if isinstance(message, dict) else False
        )
    ]
    transcript_tail = "\n".join(_message_text(message) for message in messages[-8:])
    lowered_tail = transcript_tail.lower()

    validation_seen = any(marker in lowered_tail for marker in VALIDATION_MARKERS)
    failure_seen = any(marker in lowered_tail for marker in FAILURE_MARKERS)
    success_seen = any(marker in lowered_tail for marker in SUCCESS_MARKERS)

    if validation_seen and success_seen:
        return "finalization"
    if failure_seen:
        return "repair"
    if validation_seen:
        return "validation"
    if not tool_messages or len(assistant_tool_calls) <= 2:
        return "initial_scaffold"
    if len(transcript_tail) < 3000:
        return "tool_json"
    return "long_text_or_code"
