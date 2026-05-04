import asyncio
import json

from vllm_mlx.routes.chat import (
    _AGENTIC_TOOL_USE_SYSTEM_SUFFIX,
    _agentic_completion_needs_verification,
    _agentic_diagnostic_tool_call,
    _agentic_failed_validation_present,
    _agentic_requested_artifacts_missing,
    _agentic_task_terminal_ready,
    _agentic_verification_present,
    _iterate_with_idle_timeout,
    _last_tool_result_missing_dependency,
)
from vllm_mlx.service.helpers import _disconnect_guard


def test_agentic_verification_ignores_system_prompt_evidence():
    assert not _agentic_verification_present(
        [
            {
                "role": "system",
                "content": "Run the requested validation and report success.",
            }
        ]
    )


def test_agentic_tool_use_suffix_allows_final_after_success():
    assert "latest validation succeeded" in _AGENTIC_TOOL_USE_SYSTEM_SUFFIX
    assert "stop calling tools" in _AGENTIC_TOOL_USE_SYSTEM_SUFFIX


def test_agentic_task_terminal_ready_requires_success_and_artifacts():
    assert _agentic_task_terminal_ready(
        [
            {"role": "user", "content": "Create REST routes and tests."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/routes/user.routes.ts"}',
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/user.test.ts"}',
                        }
                    },
                ],
            },
            {"role": "tool", "content": "tests passed"},
        ]
    )


def test_agentic_task_terminal_ready_rejects_latest_failure():
    assert not _agentic_task_terminal_ready(
        [
            {"role": "user", "content": "Create REST routes and tests."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/routes/user.routes.ts"}',
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/user.test.ts"}',
                        }
                    },
                ],
            },
            {"role": "tool", "content": "bun test\n 1 fail"},
        ]
    )


def test_agentic_verification_accepts_marker_from_tool_context():
    assert _agentic_verification_present(
        [
            {
                "role": "tool",
                "content": "All checks passed",
            }
        ]
    )


def test_agentic_verification_rejects_success_from_tool_call_arguments():
    assert not _agentic_verification_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "echo all checks passed"}',
                        }
                    }
                ],
            }
        ]
    )


def test_agentic_verification_rejects_plain_command_success_exit_code():
    assert not _agentic_verification_present(
        [
            {
                "role": "tool",
                "content": "bun add completed\nCommand exited with code 0",
            }
        ]
    )


def test_agentic_verification_accepts_successful_validation_command():
    assert _agentic_verification_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "cd app && bun run test 2>&1 | tail -20"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Command exited with code 0",
            },
        ]
    )


def test_agentic_verification_accepts_runner_zero_fail_summary():
    assert _agentic_verification_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "cd app && bun test"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "17 pass\n0 fail",
            },
        ]
    )


def test_agentic_verification_rejects_success_before_later_failed_validation():
    assert not _agentic_verification_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "cd app && bun test"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "17 pass\n0 fail",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "cd app && bun test"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "23 pass\n9 fail",
            },
        ]
    )


def test_agentic_verification_accepts_success_after_earlier_failed_validation():
    assert _agentic_verification_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "cd app && bun test"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "23 pass\n9 fail",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "cd app && bun test"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "32 pass\n0 fail",
            },
        ]
    )


def test_agentic_verification_rejects_zero_fail_without_validation_command():
    assert not _agentic_verification_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "echo 0 fail"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "0 fail",
            },
        ]
    )


def test_agentic_requested_artifacts_missing_detects_named_categories():
    missing = _agentic_requested_artifacts_missing(
        [
            {
                "role": "user",
                "content": "Create models, seeders, migrations, and unit tests.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/models/User.ts"}',
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/scripts/seed.ts"}',
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "test/UserService.test.ts"}',
                        }
                    }
                ],
            },
        ]
    )

    assert missing == {"migration"}


def test_agentic_requested_artifacts_missing_accepts_all_named_categories():
    assert not _agentic_requested_artifacts_missing(
        [
            {
                "role": "user",
                "content": "Create seeders and migrations.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/seeders/user.seed.ts"}',
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/migrations/001-init.ts"}',
                        }
                    }
                ],
            },
        ]
    )


def test_agentic_requested_artifacts_missing_detects_vertical_slice_and_rest_api():
    assert _agentic_requested_artifacts_missing(
        [
            {
                "role": "user",
                "content": "Create a REST api that is vertically sliced.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/models/User.ts"}',
                        }
                    }
                ],
            },
        ]
    ) == {"route", "vertical_slice"}

    assert not _agentic_requested_artifacts_missing(
        [
            {
                "role": "user",
                "content": "Create a REST api that is vertically sliced.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": (
                                '{"path": "src/modules/users/routes/user.routes.ts"}'
                            ),
                        }
                    }
                ],
            },
        ]
    )


def test_agentic_requested_artifacts_missing_requires_tests_for_each_service():
    missing = _agentic_requested_artifacts_missing(
        [
            {
                "role": "user",
                "content": "Create services and unit tests for each service.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/services/UserService.ts"}',
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "src/services/ProductService.ts"}',
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": '{"path": "test/UserService.test.ts"}',
                        }
                    }
                ],
            },
        ]
    )

    assert missing == {"test"}


def test_agentic_requested_artifacts_missing_accepts_dot_service_test_names():
    assert not _agentic_requested_artifacts_missing(
        [
            {
                "role": "user",
                "content": (
                    "Create a REST app that is vertical sliced with models, "
                    "seeders, migrations, services, and unit tests for each service."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": (
                                '{"path": "src/features/users/models/User.model.ts"}'
                            ),
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": (
                                '{"path": "src/features/users/services/user.service.ts"}'
                            ),
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": (
                                '{"path": "src/features/users/services/user.service.test.ts"}'
                            ),
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": (
                                '{"path": "src/features/users/routes/user.routes.ts"}'
                            ),
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": (
                                '{"path": "src/features/users/migrations/001-users.ts"}'
                            ),
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": (
                                '{"path": "src/features/users/seeders/001-users.ts"}'
                            ),
                        }
                    }
                ],
            },
        ]
    )


def test_agentic_diagnostic_tool_call_uses_created_manifest_root():
    tool_call = _agentic_diagnostic_tool_call(
        [
            {"role": "user", "content": "Create the project."},
            {
                "role": "tool",
                "content": (
                    "Successfully wrote 42 bytes to "
                    "/tmp/rest-api-project/package.json"
                ),
            },
            {
                "role": "tool",
                "content": (
                    "Successfully wrote 42 bytes to "
                    "/tmp/rest-api-project/src/features/users/services/user.service.ts"
                ),
            },
        ]
    )
    arguments = json.loads(tool_call.function.arguments)

    assert arguments["command"].startswith("cd /tmp/rest-api-project && ")
    assert "VALIDATION_PASSED" in arguments["command"]


def test_agentic_detection_reads_tool_transcript_when_original_user_pruned():
    assert _agentic_completion_needs_verification(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "validate project"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Validation still needed"},
        ]
    )


def test_agentic_failed_validation_detects_failed_npm_output():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "Ran validation\nCannot find module './component'\nCommand exited with code 1",
            }
        ]
    )


def test_agentic_failed_validation_detects_failed_edit_output():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": (
                    "Found 3 occurrences of edits[1] in src/service.test.ts. "
                    "Each oldText must be unique."
                ),
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
                            "arguments": '{"command": "validate project"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Validation failed"},
        ]
    )


def test_agentic_failed_validation_detects_explicit_failure():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "validate project"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "VALIDATION_FAILED: exited with code 1"},
        ]
    )


def test_agentic_failed_validation_detects_missing_requested_tests():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "run tests"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "No test files found"},
        ]
    )


def test_agentic_failed_validation_detects_bun_no_tests_found():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": (
                    'No tests found. Tests need ".test", "_test_", ".spec" '
                    'or "_spec_" in the filename.'
                ),
            }
        ]
    )


def test_agentic_failed_validation_detects_explicit_validation_failure():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "VALIDATION_FAILED: project is not complete",
            }
        ]
    )


def test_agentic_failed_validation_detects_missing_test_script():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "Missing script: \"test\"",
            }
        ]
    )


def test_agentic_failed_validation_ignores_failures_before_latest_write():
    assert not _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "validation failed",
            },
            {
                "role": "tool",
                "content": "Successfully wrote 20 bytes to src/main.tsx",
            },
            {
                "role": "tool",
                "content": "validation passed",
            },
        ]
    )


def test_agentic_failed_validation_persists_after_write_until_validation_passes():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "VALIDATION_FAILED: no test files found",
            },
            {
                "role": "tool",
                "content": "Successfully wrote 20 bytes to src/main.ts",
            },
        ]
    )


def test_agentic_failed_validation_persists_after_plain_success_exit_code():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": "VALIDATION_FAILED: no tests found",
            },
            {
                "role": "tool",
                "content": "Command exited with code 0",
            },
        ]
    )


def test_agentic_failed_validation_failure_wins_over_same_output_success_marker():
    assert _agentic_failed_validation_present(
        [
            {
                "role": "tool",
                "content": (
                    "bun test v1.3.13\n"
                    "error: Cannot find package 'sequelize-typescript'\n"
                    "0 pass\n"
                    "3 fail\n"
                    "3 errors\n"
                    "VALIDATION_PASSED"
                ),
            }
        ]
    )


def test_agentic_missing_dependency_extracts_cannot_find_package_name():
    assert (
        _last_tool_result_missing_dependency(
            [
                {
                    "role": "tool",
                    "content": (
                        "error: Cannot find package 'sequelize-typescript' "
                        "from '/tmp/app/src/models/User.ts'"
                    ),
                }
            ]
        )
        == "sequelize-typescript"
    )


def test_agentic_force_verification_after_tool_results_without_validation():
    assert _agentic_completion_needs_verification(
        [{"role": "user", "content": "build the project and validate it"}]
    )


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
