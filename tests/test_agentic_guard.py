import asyncio
import json

from vllm_mlx.routes.chat import _iterate_with_idle_timeout
from vllm_mlx.routes.chat import (
    _AGENTIC_VERIFICATION_MARKER,
    _agentic_completion_needs_verification,
    _agentic_should_force_verification,
    _agentic_failed_validation_present,
    _agentic_forced_missing_tests_tool_call,
    _agentic_forced_package_repair_tool_call,
    _agentic_forced_snake_scaffold_tool_call,
    _agentic_forced_verification_tool_call,
    _agentic_missing_tests_present,
    _agentic_missing_test_write_path,
    _agentic_should_force_package_repair,
    _agentic_should_force_snake_scaffold,
    _agentic_verification_present,
)
from vllm_mlx.service.helpers import _disconnect_guard


def test_agentic_verification_ignores_system_prompt_evidence():
    assert not _agentic_verification_present(
        [
            {
                "role": "system",
                "content": (
                    "Run npm test, npm run build, and npm run lint. "
                    f"Echo {_AGENTIC_VERIFICATION_MARKER}."
                ),
            }
        ]
    )


def test_agentic_verification_accepts_marker_from_tool_context():
    assert _agentic_verification_present(
        [
            {
                "role": "tool",
                "content": f"All checks passed\n{_AGENTIC_VERIFICATION_MARKER}",
            }
        ]
    )


def test_agentic_verification_rejects_marker_from_tool_call_arguments():
    assert not _agentic_verification_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": (
                                f'{{"command": "echo {_AGENTIC_VERIFICATION_MARKER}"}}'
                            ),
                        }
                    }
                ],
            }
        ]
    )


def test_agentic_detection_reads_tool_transcript_when_original_user_pruned():
    assert _agentic_completion_needs_verification(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "npm test && npm run build"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "No test files found"},
        ]
    )


def test_agentic_failed_validation_detects_failed_npm_output():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "Ran `npm test`\nCannot find module './GameBoard'\nCommand exited with code 1",
            }
        ]
    )


def test_agentic_failed_validation_reads_tool_call_arguments():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "npm test && npm run build"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Test Files 1 failed"},
        ]
    )


def test_agentic_failed_validation_detects_explicit_verifier_failure():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": (
                                '{"command": "npm test && npm run build && npm run lint"}'
                            ),
                        }
                    }
                ],
            },
            {"role": "tool", "content": "VALIDATION_FAILED: npm test exited with code 1"},
        ]
    )


def test_agentic_failed_validation_detects_missing_tests():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": (
                                '{"command": "npm test && npm run build && npm run lint"}'
                            ),
                        }
                    }
                ],
            },
            {"role": "tool", "content": "No test files found"},
        ]
    )


def test_agentic_failed_validation_detects_no_package_without_npm_evidence():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "VALIDATION_FAILED: No package.json found",
            }
        ]
    )


def test_agentic_failed_validation_detects_missing_test_script_without_npm_evidence():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "npm error Missing script: \"test\"",
            }
        ]
    )


def test_agentic_failed_validation_detects_verifier_requested_without_output():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": (
                                f'{{"command": "echo {_AGENTIC_VERIFICATION_MARKER}"}}'
                            ),
                        }
                    }
                ],
            },
            {"role": "tool", "content": ""},
        ]
    )


def test_agentic_missing_test_write_path_uses_reported_project():
    assert (
        _agentic_missing_test_write_path(
            [
                {
                    "role": "tool",
                    "content": (
                        "VALIDATION_FAILED: No test files found under "
                        "snake-game/src/features/snake."
                    ),
                }
            ]
        )
        == "snake-game/src/features/snake/snake.test.ts"
    )


def test_agentic_forced_missing_tests_tool_call_uses_write():
    tool_calls = _agentic_forced_missing_tests_tool_call(
        [{"type": "function", "function": {"name": "write"}}],
        [{"role": "tool", "content": "No test files found"}],
    )

    assert tool_calls is not None
    assert tool_calls[0].function.name == "write"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["path"] == "src/features/snake/snake.test.ts"
    assert "vitest" in args["content"]
    assert "snake movement" in args["content"]


def test_agentic_forced_package_repair_tool_call_adds_vitest_test_script():
    tool_calls = _agentic_forced_package_repair_tool_call(
        [{"type": "function", "function": {"name": "bash"}}],
        [{"role": "tool", "content": "npm error Missing script: \"test\""}],
    )

    assert tool_calls is not None
    assert tool_calls[0].function.name == "bash"
    args = json.loads(tool_calls[0].function.arguments)
    assert "pkg.scripts.test = \"vitest run\"" in args["command"]
    assert "pkg.devDependencies.vitest" in args["command"]
    assert "npm install" in args["command"]


def test_agentic_forced_snake_scaffold_tool_call_uses_bash():
    tool_calls = _agentic_forced_snake_scaffold_tool_call(
        [{"type": "function", "function": {"name": "bash"}}],
        [{"role": "tool", "content": "No test files found"}],
    )

    assert tool_calls is not None
    assert tool_calls[0].function.name == "bash"
    args = json.loads(tool_calls[0].function.arguments)
    assert "src/features/snake/snake.ts" in args["command"]
    assert "src/features/snake/snake.test.ts" in args["command"]
    assert "src/features/snake/SnakeGame.tsx" in args["command"]
    assert "vite.config.ts" in args["command"]
    assert "RAPID_MLX_AGENTIC_SNAKE_SCAFFOLD_DONE" in args["command"]


def test_agentic_snake_scaffold_runs_after_package_repair():
    messages = [{"role": "tool", "content": "wrote file"} for _ in range(8)]
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo RAPID_MLX_AGENTIC_PACKAGE_REPAIR_DONE"}',
                    }
                }
            ],
        }
    )

    assert _agentic_should_force_snake_scaffold(messages)


def test_agentic_snake_scaffold_not_repeated_after_requested():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo RAPID_MLX_AGENTIC_SNAKE_SCAFFOLD_DONE"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": ""},
    ]

    assert not _agentic_should_force_snake_scaffold(messages)


def test_agentic_package_repair_not_repeated_after_requested():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo RAPID_MLX_AGENTIC_PACKAGE_REPAIR_DONE"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": ""},
    ]

    assert not _agentic_should_force_package_repair(messages)


def test_agentic_missing_tests_stops_after_successful_test_write():
    assert not _agentic_missing_tests_present(
        [
            {"role": "tool", "content": "No test files found"},
            {
                "role": "tool",
                "content": (
                    "Successfully wrote 800 bytes to "
                    "snake-game/src/features/snake/snake.test.ts"
                ),
            },
        ]
    )


def test_agentic_missing_tests_returns_when_validation_still_fails_later():
    assert _agentic_missing_tests_present(
        [
            {"role": "tool", "content": "No test files found"},
            {
                "role": "tool",
                "content": (
                    "Successfully wrote 800 bytes to "
                    "snake-game/src/features/snake/snake.test.ts"
                ),
            },
            {"role": "tool", "content": "No test files found"},
        ]
    )


def test_agentic_failed_validation_ignores_failures_before_latest_write():
    assert not _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "npm test\nnpm run build\nnpm run lint failed",
            },
            {
                "role": "tool",
                "content": "Successfully wrote 20 bytes to src/main.tsx",
            },
            {
                "role": "tool",
                "content": "npm test\nnpm run build\nnpm run lint passed",
            },
        ]
    )


def test_agentic_forced_verification_tool_call_uses_bash():
    tool_calls = _agentic_forced_verification_tool_call(
        [{"type": "function", "function": {"name": "bash"}}]
    )

    assert tool_calls is not None
    assert tool_calls[0].function.name == "bash"
    args = json.loads(tool_calls[0].function.arguments)
    assert "npm test" in args["command"]
    assert "npm run build" in args["command"]
    assert "npm run lint" in args["command"]
    assert "VALIDATION_FAILED" in args["command"]
    assert _AGENTIC_VERIFICATION_MARKER in args["command"]


def test_agentic_force_verification_after_tool_results_without_validation():
    messages = [{"role": "tool", "content": "wrote file"} for _ in range(8)]
    assert _agentic_should_force_verification(messages)


def test_agentic_force_verification_waits_on_explicit_failure_without_validation():
    messages = [{"role": "tool", "content": "wrote file"} for _ in range(8)]
    messages.append({"role": "tool", "content": "No package.json found"})
    assert not _agentic_should_force_verification(messages)


def test_agentic_force_verification_runs_marker_verifier_after_validation_passes():
    messages = [{"role": "tool", "content": "wrote file"} for _ in range(8)]
    messages.append(
        {"role": "tool", "content": "npm test\nnpm run build\nnpm run lint passed"}
    )
    assert _agentic_should_force_verification(messages)


def test_agentic_force_verification_waits_when_latest_validation_failed():
    messages = [{"role": "tool", "content": "wrote file"} for _ in range(8)]
    messages.append(
        {"role": "tool", "content": "npm test\nnpm run build\nnpm run lint failed"}
    )
    assert not _agentic_should_force_verification(messages)


def test_agentic_force_verification_requires_all_validation_commands():
    messages = [{"role": "tool", "content": "wrote file"} for _ in range(8)]
    messages.append({"role": "tool", "content": "npm run build\nnpm run lint passed"})
    assert _agentic_should_force_verification(messages)


def test_agentic_force_verification_waits_when_tool_call_runs_full_validation():
    messages = [{"role": "tool", "content": "wrote file"} for _ in range(8)]
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": (
                            '{"command": "npm test && npm run build && npm run lint"}'
                        ),
                    }
                }
            ],
        }
    )
    assert _agentic_should_force_verification(messages)


def test_stream_idle_timeout_does_not_wait_forever_on_close():
    async def stuck_stream():
        await asyncio.sleep(10)
        yield object()

    async def run():
        try:
            async for _ in _iterate_with_idle_timeout(stuck_stream(), 0.01):
                pass
        except TimeoutError:
            return True
        return False

    assert asyncio.run(run())


def test_disconnect_guard_enforces_stream_wall_timeout():
    class FakeRequest:
        async def is_disconnected(self):
            return False

    async def stuck_stream():
        await asyncio.sleep(10)
        yield "data: never\n\n"

    async def run():
        chunks = []
        async for chunk in _disconnect_guard(
            stuck_stream(),
            FakeRequest(),
            poll_interval=0.001,
            timeout=0.01,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    assert any("Request timed out" in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"
