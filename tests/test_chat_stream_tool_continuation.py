# SPDX-License-Identifier: Apache-2.0
"""Tests for chat streaming tool-continuation retry behavior."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.domain.events import StreamEvent
from vllm_mlx.engine import GenerationOutput
from vllm_mlx.routes.chat import (
    _AGENTIC_DIAGNOSTIC_COMMAND,
    _AGENTIC_DEPENDENCY_INSTALL_COMMAND,
    _AGENTIC_MAX_TOOL_RESULTS_AFTER_FAILURE_BEFORE_DIAGNOSTIC,
    _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC,
    _AGENTIC_MAX_TOOL_RESULTS_BEFORE_DIAGNOSTIC,
    _AGENTIC_NO_TOOL_RETRY_MAX_TOKENS,
    _AGENTIC_REPAIR_USER_PROMPT,
    _TOOL_CALL_REPEAT_BUFFER_MAX_ARGUMENT_CHARS,
    _agentic_max_same_command_tools_since_latest_failure,
    _agentic_max_same_path_tools_since_latest_failure,
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
        self.kwargs_seen = []
        self.tokenizer = MagicMock()

    async def stream_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        self.kwargs_seen.append(kwargs)
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


class _EngineThatStallsThenCallsTool:
    def __init__(self):
        self.calls = 0
        self.messages_seen = []
        self.tokenizer = MagicMock()

    async def stream_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        if self.calls == 1:
            await asyncio.sleep(60)
            return

        yield GenerationOutput(
            text="<tool_call>",
            new_text="<tool_call>",
            prompt_tokens=12,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
        )


class _EngineThatAlwaysNarrates:
    def __init__(self):
        self.calls = 0
        self.messages_seen = []
        self.tokenizer = MagicMock()

    async def stream_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        yield GenerationOutput(
            text="Still planning.",
            new_text="Still planning.",
            prompt_tokens=10,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
        )


class _EngineThatStreamsTwoChunks:
    def __init__(self):
        self.calls = 0
        self.messages_seen = []
        self.tokenizer = MagicMock()

    async def stream_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        yield GenerationOutput(
            text="first",
            new_text="first",
            prompt_tokens=10,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
        )
        yield GenerationOutput(
            text="second",
            new_text="second",
            prompt_tokens=10,
            completion_tokens=2,
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


class _FakeRepeatedWriteSamePathPostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        if self.index == 0:
            return [
                StreamEvent(
                    type="tool_call",
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_repeat_path",
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
                                    '{"filePath":"src/main.tsx","content":"changed"}'
                                ),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                    tool_calls_detected=True,
                ),
            ]
        return [
            StreamEvent(type="content", content="Already handled."),
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


class _FakeLargeRepeatedPathWritePostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        if self.index == 0:
            large_arguments = '{"filePath":"src/main.tsx","content":"' + (
                "x" * (_TOOL_CALL_REPEAT_BUFFER_MAX_ARGUMENT_CHARS + 1)
            )
            return [
                StreamEvent(
                    type="tool_call",
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_large_repeat",
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
        return [
            StreamEvent(type="content", content="Moved on."),
            StreamEvent(type="finish", finish_reason="stop"),
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


class _FakeLongTextPostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        if self.index == 0:
            return [StreamEvent(type="content", content="x" * 5000)]
        return [
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_after_long_text",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                tool_calls_detected=True,
            )
        ]


class _FakeAlwaysLongTextPostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        return [StreamEvent(type="content", content="x" * 5000)]


class _FakeFinishOnlyPostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        return [StreamEvent(type="finish", finish_reason="stop")]


class _FakeAgenticLongTextThenToolPostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        if self.index == 0:
            self.index += 1
            long_text = " ".join(f"step{index}" for index in range(900))
            return [StreamEvent(type="content", content=long_text)]
        return [
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_after_agentic_long_text",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"npm test"}',
                        },
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
async def test_structured_cot_tools_does_not_enable_agentic_guard(monkeypatch):
    """Structured CoT for tools should not force agentic retries by itself."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    get_config().structured_cot_tools = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeStreamingPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [
                    {
                        "role": "user",
                        "content": (
                            "Build the requested project and validate it."
                        ),
                    }
                ],
                request,
                tool_continuation_retry=False,
                max_tokens=4096,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 1
    assert any("Thinking only." in chunk for chunk in chunks)
    assert not any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_agentic_guard_retries_auto_text_for_agentic_prompt(monkeypatch):
    """Agentic repair retries remain available when explicitly enabled."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    cfg = get_config()
    cfg.agentic_guard = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeStreamingPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [
                    {
                        "role": "user",
                        "content": (
                            "Build the requested project and validate it."
                        ),
                    }
                ],
                request,
                tool_continuation_retry=False,
                max_tokens=4096,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CALL_REQUIRED_RETRY_PROMPT
    assert engine.kwargs_seen[1]["max_tokens"] == _AGENTIC_NO_TOOL_RETRY_MAX_TOKENS
    assert not any("Thinking only." in chunk for chunk in chunks)
    assert any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_agentic_guard_allows_long_text_before_tool_call(monkeypatch):
    """Agentic guard should keep buffering longer planning text until tool call."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    cfg = get_config()
    cfg.agentic_guard = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeAgenticLongTextThenToolPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    engine = _EngineThatStreamsTwoChunks()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [{"role": "user", "content": "Build the requested project."}],
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 1
    assert not any('"content":"' + ("x" * 20) in chunk for chunk in chunks)
    assert any("call_after_agentic_long_text" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_agentic_repair_allows_long_text_before_tool_call(monkeypatch):
    """Repair mode should use the same agentic buffer budget as normal agentic mode."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    cfg = get_config()
    cfg.agentic_guard = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeAgenticLongTextThenToolPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "tool", "content": "VALIDATION_FAILED: tests failed"},
    ]
    engine = _EngineThatStreamsTwoChunks()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 1
    assert any("call_after_agentic_long_text" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_agentic_guard_retries_repetitive_text_without_structured_cot(monkeypatch):
    """Agentic guard should suppress repeated text loops even without structured CoT."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    get_config().agentic_guard = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeRepetitionPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [{"role": "user", "content": "Build the requested project."}],
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CALL_REQUIRED_RETRY_PROMPT
    assert not any("point point" in chunk for chunk in chunks)
    assert any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_agentic_guard_retry_exhaustion_emits_diagnostic_tool(monkeypatch):
    """Agentic guard should keep the client moving with a validation tool call."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    get_config().agentic_guard = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeAlwaysLongTextPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [{"role": "user", "content": "Build the requested project."}],
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 3
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any("AGENTIC_DIAGNOSTIC" in chunk for chunk in chunks)
    assert not any("could not produce a valid tool call" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_retry_exhaustion_without_buffer_emits_diagnostic(
    monkeypatch,
):
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    get_config().agentic_guard = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeFinishOnlyPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    engine = _EngineThatStreamsTwoChunks()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [{"role": "user", "content": "Build the requested project."}],
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 3
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any("AGENTIC_DIAGNOSTIC" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_forces_dependency_install_after_missing_dependency():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [
                    {"role": "user", "content": "Build the requested project."},
                    {
                        "role": "tool",
                        "content": (
                            "VALIDATION_FAILED\n"
                            "error: Cannot find package 'express'"
                        ),
                    },
                ],
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 0
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any("bun add express" in chunk for chunk in chunks)
    assert _AGENTIC_DEPENDENCY_INSTALL_COMMAND


@pytest.mark.asyncio
async def test_agentic_guard_forces_dependency_install_after_install_package_hint():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [
                    {"role": "user", "content": "Build the requested project."},
                    {
                        "role": "tool",
                        "content": (
                            "VALIDATION_FAILED\n"
                            "error: Please install package manually: sqlite3"
                        ),
                    },
                ],
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 0
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any("bun add sqlite3" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_agentic_guard_does_not_install_for_relative_module_import():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [
                    {"role": "user", "content": "Build the requested project."},
                    {
                        "role": "tool",
                        "content": (
                            "VALIDATION_FAILED\n"
                            "error: Cannot find module '../../orders/order.model'"
                        ),
                    },
                ],
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls > 0
    assert chunks[-1] == "data: [DONE]\n\n"
    assert not any("bun install" in chunk for chunk in chunks)


def test_agentic_diagnostic_and_repair_prompt_are_generic():
    assert "package.json" in _AGENTIC_DIAGNOSTIC_COMMAND
    assert "pyproject.toml" in _AGENTIC_DIAGNOSTIC_COMMAND
    assert "go.mod" in _AGENTIC_DIAGNOSTIC_COMMAND
    assert "Cargo.toml" in _AGENTIC_DIAGNOSTIC_COMMAND
    assert "missing package.json" not in _AGENTIC_DIAGNOSTIC_COMMAND
    assert "NEXT_ACTION:" in _AGENTIC_DIAGNOSTIC_COMMAND
    assert "VALIDATION_PASSED" in _AGENTIC_DIAGNOSTIC_COMMAND
    assert "VALIDATION_FAILED" in _AGENTIC_DIAGNOSTIC_COMMAND
    assert "express" not in _AGENTIC_DIAGNOSTIC_COMMAND.lower()
    assert "sequelize" not in _AGENTIC_DIAGNOSTIC_COMMAND.lower()
    assert "Do not expand scope" in _AGENTIC_REPAIR_USER_PROMPT
    assert "package.json or install" not in _AGENTIC_REPAIR_USER_PROMPT
    assert "no tests were found" in _AGENTIC_REPAIR_USER_PROMPT
    assert "source and test directory structure" in _AGENTIC_REPAIR_USER_PROMPT
    assert "named import/export errors" in _AGENTIC_REPAIR_USER_PROMPT
    assert "duplicate declarations" in _AGENTIC_REPAIR_USER_PROMPT
    assert "no tests were found" in _AGENTIC_DIAGNOSTIC_COMMAND


@pytest.mark.asyncio
async def test_agentic_guard_forces_diagnostic_after_many_tool_results():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [{"role": "user", "content": "Build the requested project."}]
    for index in range(_AGENTIC_MAX_TOOL_RESULTS_BEFORE_DIAGNOSTIC):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": f'{{"path":"src/file{index}.ts"}}',
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "successfully wrote file"})
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 0
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any("AGENTIC_DIAGNOSTIC" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_forces_diagnostic_after_stale_failed_validation():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "assistant", "tool_calls": []},
        {"role": "tool", "content": "VALIDATION_FAILED: cannot find module"},
    ]
    for index in range(_AGENTIC_MAX_TOOL_RESULTS_AFTER_FAILURE_BEFORE_DIAGNOSTIC):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": f'{{"path":"src/file{index}.ts"}}',
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "edit applied"})
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 0
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any("AGENTIC_DIAGNOSTIC" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_forces_diagnostic_after_repeated_same_path_repairs():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "tool", "content": "VALIDATION_FAILED: cannot find module"},
    ]
    for index in range(
        _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
    ):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": (
                                '{"path":"src/config/db.ts",'
                                f'"edits":[{{"oldText":"a{index}",'
                                f'"newText":"b{index}"}}]}}'
                            ),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "edit applied"})
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls > 0
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_failed_diagnostic_anchors_repair_window():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {
            "role": "tool",
            "content": "AGENTIC_DIAGNOSTIC\nVALIDATION_FAILED: tests failed",
        },
    ]
    for index in range(_AGENTIC_MAX_TOOL_RESULTS_AFTER_FAILURE_BEFORE_DIAGNOSTIC):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": f'{{"path":"src/file{index}.ts"}}',
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "successfully wrote file"})
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 0
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any("AGENTIC_DIAGNOSTIC" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_failed_diagnostic_anchors_same_path_window():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {
            "role": "tool",
            "content": "AGENTIC_DIAGNOSTIC\nVALIDATION_FAILED: tests failed",
        },
    ]
    for index in range(
        _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
    ):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": (
                                '{"path":"src/controllers/order.ts",'
                                f'"edits":[{{"oldText":"a{index}",'
                                f'"newText":"b{index}"}}]}}'
                            ),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "edit applied"})
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls > 0
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_failed_diagnostic_resets_same_path_window():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "tool", "content": "VALIDATION_FAILED: cannot find module"},
    ]
    for index in range(
        _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
    ):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": (
                                '{"path":"src/config/db.ts",'
                                f'"edits":[{{"oldText":"a{index}",'
                                f'"newText":"b{index}"}}]}}'
                            ),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "edit applied"})
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"AGENTIC_DIAGNOSTIC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "AGENTIC_DIAGNOSTIC\nVALIDATION_FAILED: still broken",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": (
                                '{"path":"src/config/db.ts",'
                                '"edits":[{"oldText":"c","newText":"d"}]}'
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "content": "edit applied"},
        ]
    )
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls > 0
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_repeated_same_path_uses_repair_prompt_not_diagnostic():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "tool", "content": "VALIDATION_FAILED: cannot find module"},
    ]
    for index in range(
        _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
    ):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": (
                                '{"path":"src/models/user.ts",'
                                f'"edits":[{{"oldText":"a{index}",'
                                f'"newText":"b{index}"}}]}}'
                            ),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "edit applied"})
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls > 0
    assert any(
        "use write with the complete corrected file content" in message.get(
            "content", ""
        )
        for message in engine.messages_seen[0]
        if message.get("role") == "user"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agentic_guard_blocks_repeated_same_path_after_repair_prompt(monkeypatch):
    """In repeated-path repair mode, do not stream another write to that path."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    get_config().agentic_guard = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeRepeatedWriteSamePathPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "build project and run tests"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        tool_choice="auto",
    )
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "tool", "content": "VALIDATION_FAILED: cannot find module"},
    ]
    for index in range(
        _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
    ):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_prev_{index}",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": (
                                '{"filePath":"src/main.tsx",'
                                f'"content":"changed {index}"}}'
                            ),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "write applied"})

    engine = _EngineThatRepeatsToolThenAnswers()
    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=True,
                max_tokens=16,
                tools=request.tools,
            )
        ]
    finally:
        reset_config()

    assert not any("call_repeat_path" in chunk for chunk in chunks)
    assert any("AGENTIC_DIAGNOSTIC" in chunk for chunk in chunks)


def test_agentic_same_path_count_includes_failed_edits():
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "tool", "content": "VALIDATION_FAILED: tests failed"},
    ]
    for index in range(4):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": (
                                '{"path":"tests/service.test.ts",'
                                f'"edits":[{{"oldText":"a{index}",'
                                f'"newText":"b{index}"}}]}}'
                            ),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "Edit failed: oldText not found"})

    assert _agentic_max_same_path_tools_since_latest_failure(messages) == 4


def test_agentic_same_command_count_includes_repeated_inspection_after_failure():
    command = 'find /tmp/work -type f -name "*.ts" -o -name "*.js" | grep -v node_modules'
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "tool", "content": "VALIDATION_FAILED: tests failed"},
    ]
    for _ in range(4):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": command}),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "/tmp/work/src/index.ts"})

    assert _agentic_max_same_command_tools_since_latest_failure(messages) == 4


@pytest.mark.asyncio
async def test_agentic_guard_allows_model_turn_after_diagnostic():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
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
    messages = [
        {"role": "user", "content": "Build the requested project."},
        {"role": "tool", "content": "VALIDATION_FAILED: cannot find module"},
    ]
    for index in range(_AGENTIC_MAX_TOOL_RESULTS_AFTER_FAILURE_BEFORE_DIAGNOSTIC):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": f'{{"path":"src/file{index}.ts"}}',
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "edit applied"})
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"AGENTIC_DIAGNOSTIC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "AGENTIC_DIAGNOSTIC\nVALIDATION_FAILED: still broken",
            },
        ]
    )
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 3
    assert any("AGENTIC_DIAGNOSTIC" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_agentic_guard_keeps_working_when_requested_artifact_missing():
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    get_config().agentic_guard = True

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "oi"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        tool_choice="auto",
    )
    messages = [
        {"role": "user", "content": "Create models, migrations, and unit tests."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": '{"path":"src/models/User.ts"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "Successfully wrote src/models/User.ts"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": '{"path":"test/User.test.ts"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "Successfully wrote test/User.test.ts"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"bun test"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "1 pass\n0 fail"},
    ]
    engine = _EngineThatAlwaysNarrates()

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                tool_continuation_retry=False,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert any(
        "artifact categories that are still absent" in message.get("content", "")
        for message in engine.messages_seen[0]
        if message.get("role") == "user"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_structured_cot_tools_retries_auto_text_after_tool_result(monkeypatch):
    """Agentic structured-COT mode keeps tool continuation moving for auto tools."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    get_config().structured_cot_tools = True
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

    try:
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
    finally:
        reset_config()

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CONTINUATION_RETRY_PROMPT
    assert not any("Thinking only." in chunk for chunk in chunks)
    assert any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_agentic_guard_allows_final_text_after_validation_passes(monkeypatch):
    """Once validation evidence exists, structured tool mode should allow final text."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    cfg = get_config()
    cfg.agentic_guard = True
    cfg.structured_cot_tools = True
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

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [
                    {"role": "user", "content": "Build and validate project."},
                    {"role": "tool", "content": "VALIDATION_PASSED"},
                ],
                request,
                tool_continuation_retry=True,
                max_tokens=16,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 1
    assert any("Thinking only." in chunk for chunk in chunks)
    assert not any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_structured_cot_tools_retries_long_auto_text_before_tool_call(monkeypatch):
    """Agentic continuation does not wait for max_tokens of text before retrying."""
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.service import postprocessor

    reset_config()
    get_config().structured_cot_tools = True
    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeLongTextPostProcessor,
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

    try:
        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                engine,
                [{"role": "tool", "content": "done"}],
                request,
                tool_continuation_retry=True,
                max_tokens=32768,
            )
        ]
    finally:
        reset_config()

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CONTINUATION_RETRY_PROMPT
    assert not any('"xxxxx' in chunk for chunk in chunks)
    assert any("call_after_long_text" in chunk for chunk in chunks)


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
async def test_repeated_write_to_same_path_retries_even_when_content_changes(monkeypatch):
    """Do not rewrite the same file path after its tool result."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeRepeatedWriteSamePathPostProcessor,
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
    assert any("Already handled." in chunk for chunk in chunks)
    assert not any("call_repeat_path" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_repeated_path_is_allowed_after_failed_tool_result(monkeypatch):
    """Allow another edit to the same path when the previous edit failed."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeRepeatedWriteSamePathPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "fix app"}],
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
                                "name": "edit",
                                "arguments": (
                                    '{"filePath":"src/main.tsx","edits":[]}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_prev",
                    "content": "Edit failed: oldText not found",
                },
            ],
            request,
            tool_continuation_retry=True,
            max_tokens=16,
            tools=[{"type": "function", "function": {"name": "write"}}],
        )
    ]

    assert engine.calls == 1
    assert any("call_repeat_path" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_repeated_large_write_to_same_path_retries_from_partial_path(monkeypatch):
    """Detect repeated file paths before holding a huge incomplete JSON buffer."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeLargeRepeatedPathWritePostProcessor,
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
    assert any("Moved on." in chunk for chunk in chunks)
    assert not any("call_large_repeat" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_tool_continuation_retries_when_stream_idles(monkeypatch):
    """A stalled stream attempt should not leave the client waiting forever."""
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
        timeout=0.01,
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
    engine = _EngineThatStallsThenCallsTool()

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
    assert any('"tool_calls"' in chunk for chunk in chunks)


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
async def test_large_required_tool_call_streams_after_buffer_limit(monkeypatch):
    """Large write arguments should stream instead of waiting for full JSON."""
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
        tool_choice="required",
    )
    engine = _EngineThatRepeatsToolThenAnswers()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [{"role": "user", "content": "Build app"}],
            request,
            tool_continuation_retry=False,
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
