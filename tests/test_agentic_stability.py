# SPDX-License-Identifier: Apache-2.0
"""Tests for the agentic stability harness."""

from pathlib import Path

import pytest

from vllm_mlx.agents.stability import (
    Classification,
    StabilityState,
    artifact_coverage,
    classify_turn,
    execute_tool,
    final_classification,
    is_read_only_tool,
    is_validation_command,
    normalize_tool_call,
    requested_artifact_groups,
    safe_workspace_path,
    snapshot_workspace,
    validation_result_passed,
)


def _response(content="", tool_calls=None):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls or [],
                }
            }
        ]
    }


def _tool_call(name="write", args=None):
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": args if isinstance(args, str) else __import__("json").dumps(args or {}),
        },
    }


def test_normalize_tool_call_rejects_malformed_arguments():
    with pytest.raises(ValueError, match="JSON object"):
        normalize_tool_call(_tool_call(args='["not", "object"]'))


def test_normalize_tool_call_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="write.content"):
        normalize_tool_call(_tool_call("write", {"path": "package.json"}))


def test_write_serializes_structured_json_content(tmp_path: Path):
    result = execute_tool(
        "write",
        {"path": "tsconfig.json", "content": {"compilerOptions": {"strict": True}}},
        tmp_path,
        5,
    )
    assert result["ok"]
    assert '"strict": true' in (tmp_path / "tsconfig.json").read_text(encoding="utf-8")


def test_workspace_path_cannot_escape_root(tmp_path: Path):
    with pytest.raises(ValueError, match="escapes"):
        safe_workspace_path(tmp_path, "../outside.txt")


def test_execute_write_read_and_edit_are_traced_operations(tmp_path: Path):
    assert execute_tool("write", {"path": "src/a.txt", "content": "one"}, tmp_path, 5)[
        "ok"
    ]
    assert execute_tool("read", {"path": "src/a.txt"}, tmp_path, 5)["content"] == "one"
    assert execute_tool(
        "edit", {"path": "src/a.txt", "old": "one", "new": "two"}, tmp_path, 5
    )["ok"]
    assert (tmp_path / "src" / "a.txt").read_text(encoding="utf-8") == "two"


def test_repeated_response_classified_as_loop(tmp_path: Path):
    state = StabilityState()
    snap = snapshot_workspace(tmp_path)
    result = Classification()
    for _ in range(3):
        result = classify_turn(state, _response(content="same"), snap, snap)
    assert result.category == "LOOP"


def test_repeated_tool_without_growth_classified_as_loop(tmp_path: Path):
    state = StabilityState()
    snap = snapshot_workspace(tmp_path)
    result = Classification()
    call = _tool_call("read", {"path": "missing.txt"})
    for _ in range(3):
        result = classify_turn(state, _response(tool_calls=[call]), snap, snap)
    assert result.category == "LOOP"


def test_directory_creation_counts_as_progress(tmp_path: Path):
    before = snapshot_workspace(tmp_path)
    (tmp_path / "src" / "users" / "models").mkdir(parents=True)
    after = snapshot_workspace(tmp_path)
    assert after.score > before.score


def test_api_error_classification(tmp_path: Path):
    state = StabilityState()
    snap = snapshot_workspace(tmp_path)
    result = classify_turn(state, None, snap, snap, api_error="timeout")
    assert result.category == "API_ERROR"


def test_artifact_coverage_and_partial_output(tmp_path: Path):
    required_groups = requested_artifact_groups(
        "Create models, migrations, seeders, services, and unit tests."
    )
    files = {
        "src/users/user.model.ts": "",
        "src/users/user.service.ts": "",
        "src/users/user.service.test.ts": "",
        "migrations/001-create-user.ts": "",
        "seeders/user.seeder.ts": "",
    }
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    assert all(artifact_coverage(snapshot_workspace(tmp_path), required_groups).values())
    assert (
        final_classification(tmp_path, required_groups=required_groups).category
        == "PARTIAL_OUTPUT"
    )
    assert (
        final_classification(
            tmp_path, validation_passed=True, required_groups=required_groups
        ).category
        is None
    )

    (tmp_path / "seeders" / "user.seeder.ts").unlink()
    partial = final_classification(tmp_path, required_groups=required_groups)
    assert partial.category == "PARTIAL_OUTPUT"
    assert "seeders" in partial.reason


def test_requested_artifact_groups_are_prompt_driven():
    assert requested_artifact_groups("Create a CLI tool.") == {}
    groups = requested_artifact_groups("Create REST routes and service tests.")
    assert set(groups) == {"routes", "services", "unit_tests"}


def test_bash_uses_pipefail(tmp_path: Path):
    result = execute_tool("bash", {"command": "false | head -1"}, tmp_path, 5)
    assert not result["ok"]


def test_snapshot_includes_directory_count(tmp_path: Path):
    (tmp_path / "src" / "users").mkdir(parents=True)
    snapshot = snapshot_workspace(tmp_path)
    assert "src/users" in snapshot.dirs


def test_validation_detection_ignores_heredoc_content():
    command = """cat > package.json << 'EOF'
{"scripts":{"test":"bun test"}}
EOF"""
    assert not is_validation_command(command)
    assert is_validation_command("npm run test:unit 2>&1 | head -100")


def test_validation_result_rejects_masked_failures():
    assert not validation_result_passed(
        {"ok": True, "returncode": 0, "stdout": "2 fail\nEXIT:1", "stderr": ""}
    )
    assert validation_result_passed(
        {"ok": True, "returncode": 0, "stdout": "10 pass", "stderr": ""}
    )


def test_read_only_detection_for_audit_batches():
    assert is_read_only_tool("read", {"path": "src/service.ts"})
    assert is_read_only_tool("bash", {"command": "find . -type f; cat package.json"})
    assert not is_read_only_tool("bash", {"command": "mkdir -p src"})
    assert not is_read_only_tool("bash", {"command": "bun test"})
