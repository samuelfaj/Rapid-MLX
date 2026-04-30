# SPDX-License-Identifier: Apache-2.0
"""Tests for chat streaming tool-continuation retry behavior."""

from unittest.mock import MagicMock

import pytest

from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.domain.events import StreamEvent
from vllm_mlx.engine import GenerationOutput
from vllm_mlx.routes.chat import (
    _TOOL_CALL_REPEAT_BUFFER_MAX_ARGUMENT_CHARS,
    stream_chat_completion,
)
from vllm_mlx.service.helpers import (
    _TOOL_CALL_JSON_RETRY_PROMPT,
    _TOOL_CALL_REQUIRED_RETRY_PROMPT,
    _TOOL_CONTINUATION_REPEATED_TOOL_PROMPT,
    _TOOL_CONTINUATION_RETRY_PROMPT,
)


class _EngineThatExhaustsThenCallsTool:
    def __init__(self):
        self.calls = 0
        self.messages_seen = []
        self.tokenizer = MagicMock()

    async def stream_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        if self.calls == 1:
            yield GenerationOutput(
                text="Thinking only.",
                new_text="Thinking only.",
                prompt_tokens=10,
                completion_tokens=2,
                finished=False,
                finish_reason=None,
            )
            return

        yield GenerationOutput(
            text="<tool_call>",
            new_text="<tool_call>",
            prompt_tokens=12,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
        )


class _EngineThatRepeatsToolThenAnswers:
    def __init__(self):
        self.calls = 0
        self.messages_seen = []
        self.kwargs_seen = []
        self.tokenizer = MagicMock()

    async def stream_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        self.kwargs_seen.append(kwargs)
        if self.calls == 1:
            yield GenerationOutput(
                text="<tool_call>",
                new_text="<tool_call>",
                prompt_tokens=10,
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
            )
            return

        yield GenerationOutput(
            text="Already wrote it.",
            new_text="Already wrote it.",
            prompt_tokens=12,
            completion_tokens=4,
            finished=True,
            finish_reason="stop",
        )


class _FakeStreamingPostProcessor:
    instances = 0

    def __init__(self, *args, **kwargs):
        self.index = _FakeStreamingPostProcessor.instances
        _FakeStreamingPostProcessor.instances += 1

    def set_thinking_model(self, model_name):
        pass

    def reset(self):
        pass

    def process_chunk(self, output):
        if self.index == 0:
            return [StreamEvent(type="content", content="Thinking only.")]
        return [
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                tool_calls_detected=True,
            )
        ]

    def finalize(self):
        return []


class _FakeRepeatedWritePostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        if self.index == 0:
            return [
                StreamEvent(
                    type="tool_call",
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_repeat",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": "",
                            },
                        }
                    ],
                    tool_calls_detected=True,
                ),
                StreamEvent(
                    type="tool_call",
                    tool_calls=[
                        {
                            "index": 0,
                            "function": {
                                "arguments": (
                                    '{"filePath":"src/main.tsx","content":"same"}'
                                ),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                    tool_calls_detected=True,
                ),
            ]
        return [
            StreamEvent(type="content", content="Already wrote it."),
            StreamEvent(type="finish", finish_reason="stop"),
        ]


class _FakeLargeWritePostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        large_arguments = '{"filePath":"src/App.tsx","content":"' + (
            "x" * (_TOOL_CALL_REPEAT_BUFFER_MAX_ARGUMENT_CHARS + 1)
        )
        return [
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_large",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": "",
                        },
                    }
                ],
                tool_calls_detected=True,
            ),
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "function": {
                            "arguments": large_arguments,
                        },
                    }
                ],
                tool_calls_detected=True,
            ),
        ]


class _FakeRepetitionPostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        if self.index == 0:
            return [StreamEvent(type="content", content="point " * 40)]
        return [
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_after_retry",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                tool_calls_detected=True,
            )
        ]


class _EngineThatTruncatesThenCallsTool:
    def __init__(self):
        self.calls = 0
        self.messages_seen = []
        self.tokenizer = MagicMock()

    async def stream_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        if self.calls == 1:
            yield GenerationOutput(
                text="<tool_call><function=write>{",
                new_text="<tool_call><function=write>{",
                prompt_tokens=10,
                completion_tokens=8192,
                finished=False,
                finish_reason=None,
            )
            yield GenerationOutput(
                text="<tool_call><function=write>{",
                new_text="",
                prompt_tokens=10,
                completion_tokens=8192,
                finished=True,
                finish_reason="length",
            )
            return

        yield GenerationOutput(
            text="<tool_call><function=write>{}</function></tool_call>",
            new_text="<tool_call><function=write>{}</function></tool_call>",
            prompt_tokens=12,
            completion_tokens=10,
            finished=True,
            finish_reason="stop",
        )


class _FakeTruncatedToolPostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        if self.index == 0:
            if output.finished:
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                ]
            return [
                StreamEvent(
                    type="tool_call",
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_truncated",
                            "type": "function",
                            "function": {"name": "write", "arguments": ""},
                        }
                    ],
                    tool_calls_detected=True,
                ),
                StreamEvent(
                    type="tool_call",
                    tool_calls=[
                        {
                            "index": 0,
                            "function": {"arguments": "{"},
                        }
                    ],
                    tool_calls_detected=True,
                ),
            ]

        return [
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_complete",
                        "type": "function",
                        "function": {"name": "write", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                tool_calls_detected=True,
            )
        ]


@pytest.mark.asyncio
async def test_tool_continuation_retries_when_stream_exhausts_without_finish(
    monkeypatch,
):
    """Retry when engine stream ends after narration without finish event."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeStreamingPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "do work"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="required",
    )
    engine = _EngineThatExhaustsThenCallsTool()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [{"role": "tool", "content": "done"}],
            request,
            tool_continuation_retry=True,
            max_tokens=16,
        )
    ]

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CONTINUATION_RETRY_PROMPT
    assert not any("Thinking only." in chunk for chunk in chunks)
    assert any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_tool_request_retries_when_model_repeats_text_before_tool_call(
    monkeypatch,
):
    """Retry first tool turn when model enters repeated-text loop."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeRepetitionPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "do work"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="required",
    )
    engine = _EngineThatExhaustsThenCallsTool()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [{"role": "user", "content": "Build app"}],
            request,
            tool_continuation_retry=False,
            max_tokens=16,
        )
    ]

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CALL_REQUIRED_RETRY_PROMPT
    assert not any("point point" in chunk for chunk in chunks)
    assert any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_tool_request_retries_when_tool_call_json_is_truncated(monkeypatch):
    """Retry instead of streaming partial function arguments to the client."""
    from vllm_mlx.service import postprocessor

    _FakeTruncatedToolPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeTruncatedToolPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "write file"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="required",
    )
    engine = _EngineThatTruncatesThenCallsTool()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [{"role": "user", "content": "Build app"}],
            request,
            tool_continuation_retry=False,
            max_tokens=16,
        )
    ]

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CALL_JSON_RETRY_PROMPT
    assert not any("call_truncated" in chunk for chunk in chunks)
    assert any("call_complete" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_tool_auto_allows_text_final_after_tool_result(monkeypatch):
    """Do not force another tool call when auto tool choice produces final text."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeStreamingPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "do work"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )
    engine = _EngineThatExhaustsThenCallsTool()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [{"role": "tool", "content": "done"}],
            request,
            tool_continuation_retry=True,
            max_tokens=16,
        )
    ]

    assert engine.calls == 1
    assert any("Thinking only." in chunk for chunk in chunks)
    assert not any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_repeated_tool_call_retries_with_anti_loop_prompt(monkeypatch):
    """Do not emit the same tool call again after its result."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeRepeatedWritePostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "write app"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="required",
    )
    engine = _EngineThatRepeatsToolThenAnswers()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_prev",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": (
                                    '{"filePath":"src/main.tsx","content":"same"}'
                                ),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_prev", "content": "ok"},
            ],
            request,
            tool_continuation_retry=True,
            max_tokens=16,
            tools=[{"type": "function", "function": {"name": "write"}}],
        )
    ]

    assert engine.calls == 2
    assert "tools" in engine.kwargs_seen[0]
    assert "tools" in engine.kwargs_seen[1]
    assert any("Already wrote it." in chunk for chunk in chunks)
    assert not any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_large_auto_tool_call_streams_after_repeat_buffer_limit(monkeypatch):
    """Do not hold large non-required tool calls while checking for repeats."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeLargeWritePostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "write app"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )
    engine = _EngineThatRepeatsToolThenAnswers()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_prev",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": (
                                    '{"filePath":"src/main.tsx","content":"same"}'
                                ),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_prev", "content": "ok"},
            ],
            request,
            tool_continuation_retry=True,
            max_tokens=16,
            tools=[{"type": "function", "function": {"name": "write"}}],
        )
    ]

    assert engine.calls == 1
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any("call_large" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_repeated_tool_prompt_allows_text_final_with_required_tool_choice(
    monkeypatch,
):
    """Do not force another tool after anti-loop repeated-tool prompt."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeStreamingPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "do work"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="required",
    )
    engine = _EngineThatExhaustsThenCallsTool()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [
                {"role": "tool", "content": "npm error"},
                {"role": "user", "content": _TOOL_CONTINUATION_REPEATED_TOOL_PROMPT},
            ],
            request,
            tool_continuation_retry=True,
            max_tokens=16,
        )
    ]

    assert engine.calls == 1
    assert any("Thinking only." in chunk for chunk in chunks)
    assert not any('"tool_calls"' in chunk for chunk in chunks)
