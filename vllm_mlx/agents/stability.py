# SPDX-License-Identifier: Apache-2.0
"""Agentic stability harness for local Rapid-MLX servers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Complete shell command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a UTF-8 text file from the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative workspace file path.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a UTF-8 text file in the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative workspace file path.",
                    },
                    "content": {
                        "description": (
                            "Complete file content. Prefer a string. Structured JSON values "
                            "are accepted and will be serialized for JSON-like files. Never "
                            "omit this field."
                        ),
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace exact text in a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative workspace file path.",
                    },
                    "old": {"type": "string", "description": "Exact text to replace."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
]
ALLOWED_TOOL_NAMES = {tool["function"]["name"] for tool in TOOL_SCHEMAS}

ARTIFACT_TERM_CATALOG = {
    "models": ("model", "models"),
    "migrations": ("migration", "migrations"),
    "seeders": ("seed", "seeder", "seeders"),
    "services": ("service", "services"),
    "unit_tests": ("test", "tests", "spec"),
    "controllers": ("controller", "controllers"),
    "routes": ("api", "endpoint", "endpoints", "rest", "route", "routes", "router"),
    "middleware": ("middleware", "middlewares"),
    "views": ("view", "views", "component", "components", "page", "pages"),
    "styles": ("style", "styles", "css", "tailwind"),
}
VALIDATION_COMMAND_RE = re.compile(
    r"\b("
    r"bun\s+(run\s+)?test|"
    r"npm\s+(run\s+)?(test|test:unit|build|typecheck)|"
    r"pnpm\s+(run\s+)?(test|build|typecheck)|"
    r"yarn\s+(run\s+)?(test|build|typecheck)|"
    r"pytest|python\s+-m\s+pytest|"
    r"go\s+test|cargo\s+test|mvn\s+test|gradle(w)?\s+test|"
    r"tsc(\s|$)"
    r")",
    re.IGNORECASE,
)
VALIDATION_FAILURE_RE = re.compile(
    r"("
    r"\b(exit|status)\s*:\s*[1-9]\d*\b|"
    r"\b[1-9]\d*\s+(fail|failed|failure|errors?)\b|"
    r"\b0\s+pass\b|"
    r"\b(error|failed|failure|cannot find module|missing script|no test files?)\b"
    r")",
    re.IGNORECASE,
)


@dataclass
class Classification:
    category: str | None = None
    reason: str = ""


@dataclass
class WorkspaceSnapshot:
    files: tuple[str, ...]
    total_bytes: int
    dirs: tuple[str, ...] = ()

    @property
    def score(self) -> int:
        return len(self.files) * 1000 + len(self.dirs) * 100 + self.total_bytes


@dataclass
class StabilityState:
    response_fingerprints: Counter[str] = field(default_factory=Counter)
    tool_signatures: Counter[str] = field(default_factory=Counter)
    recent_scores: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    invalid_tool_calls: int = 0
    api_errors: int = 0


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def fingerprint(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def safe_workspace_path(root: Path, rel_path: str) -> Path:
    candidate = (root / rel_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("path escapes workspace")
    return candidate


def snapshot_workspace(root: Path) -> WorkspaceSnapshot:
    files: list[str] = []
    dirs: list[str] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith((".git/", "node_modules/", "dist/", "build/")):
            continue
        if path.is_dir():
            dirs.append(rel)
            continue
        if not path.is_file():
            continue
        files.append(rel)
        total_bytes += path.stat().st_size
    return WorkspaceSnapshot(tuple(files), total_bytes, tuple(dirs))


def requested_artifact_groups(prompt: str) -> dict[str, tuple[str, ...]]:
    lowered = prompt.lower()
    groups: dict[str, tuple[str, ...]] = {}
    for group, terms in ARTIFACT_TERM_CATALOG.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
            groups[group] = terms
    return groups


def artifact_coverage(
    snapshot: WorkspaceSnapshot,
    required_groups: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, bool]:
    groups = required_groups or {}
    lowered = [p.lower() for p in snapshot.files]
    coverage: dict[str, bool] = {}
    for group, tokens in groups.items():
        coverage[group] = any(any(token in path for token in tokens) for path in lowered)
    return coverage


def is_validation_command(command: str) -> bool:
    command_headers = []
    for segment in command.split(";"):
        header = segment.split("<<", 1)[0].strip()
        if header:
            command_headers.append(header)
    return any(VALIDATION_COMMAND_RE.search(header) for header in command_headers)


def validation_result_passed(result: dict[str, Any]) -> bool:
    if result.get("ok") is not True or result.get("returncode") != 0:
        return False
    output = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}"
    return VALIDATION_FAILURE_RE.search(output) is None


def is_read_only_tool(name: str, args: dict[str, Any]) -> bool:
    if name == "read":
        return True
    if name != "bash":
        return False
    command = args.get("command")
    if not isinstance(command, str):
        return False
    headers = [segment.split("<<", 1)[0].strip() for segment in command.split(";")]
    write_markers = (">", "tee ", "mv ", "cp ", "rm ", "mkdir ", "npm ", "bun ")
    return all(
        header.startswith(("ls ", "find ", "cat ", "pwd", "head ", "sed "))
        and not any(marker in header for marker in write_markers)
        for header in headers
        if header
    )


def message_char_count(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str))


def compact_messages(
    system_message: dict[str, Any],
    user_message: dict[str, Any],
    workspace: Path,
    reason: str,
    required_groups: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    snapshot = snapshot_workspace(workspace)
    return [
        system_message,
        user_message,
        {
            "role": "user",
            "content": (
                f"Context compacted because {reason}. Continue from the actual "
                f"workspace state. Artifact coverage: {artifact_coverage(snapshot, required_groups)}. "
                f"Files: {list(snapshot.files)}. Do not recreate files that already "
                "exist unless validation output requires a targeted edit. Create only "
                "missing requested artifacts, then run one validation command and finish."
            ),
        },
    ]


def parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        parsed = json.loads(raw_args or "{}")
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("tool arguments must be a JSON object")


def normalize_tool_call(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ValueError("missing function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("missing function name")
    if name not in ALLOWED_TOOL_NAMES:
        raise ValueError(f"unknown tool: {name}")
    args = parse_tool_arguments(function.get("arguments", "{}"))
    validate_tool_arguments(name, args)
    call_id = str(tool_call.get("id") or fingerprint(tool_call))
    return name, args, call_id


def validate_tool_arguments(name: str, args: dict[str, Any]) -> None:
    required_by_tool = {
        "bash": {"command": str},
        "read": {"path": str},
        "write": {"path": str, "content": object},
        "edit": {"path": str, "old": str, "new": str},
    }
    required = required_by_tool.get(name)
    if required is None:
        raise ValueError(f"unknown tool: {name}")
    for arg_name, expected_type in required.items():
        value = args.get(arg_name)
        if value is None:
            raise ValueError(f"{name}.{arg_name} must be present")
        if not isinstance(value, expected_type) or (
            expected_type is str and arg_name != "content" and not value
        ):
            raise ValueError(f"{name}.{arg_name} must be {expected_type.__name__}")


def execute_tool(name: str, args: dict[str, Any], workspace: Path, timeout: int) -> dict[str, Any]:
    if name == "bash":
        command = args.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("bash.command must be a non-empty string")
        proc = subprocess.run(
            ["/bin/bash", "-o", "pipefail", "-c", command],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
        }
    if name == "read":
        path = args.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("read.path must be a non-empty string")
        target = safe_workspace_path(workspace, path)
        return {"ok": True, "content": target.read_text(encoding="utf-8")[-20000:]}
    if name == "write":
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path:
            raise ValueError("write.path must be a non-empty string")
        if content is None:
            raise ValueError("write.content must be present")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, ensure_ascii=False) + "\n"
        target = safe_workspace_path(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "bytes": len(content.encode("utf-8"))}
    if name == "edit":
        path = args.get("path")
        old = args.get("old")
        new = args.get("new")
        if not isinstance(path, str) or not path:
            raise ValueError("edit.path must be a non-empty string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError("edit.old and edit.new must be strings")
        target = safe_workspace_path(workspace, path)
        text = target.read_text(encoding="utf-8")
        if old not in text:
            raise ValueError("edit.old not found")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"ok": True, "replacements": 1}
    raise ValueError(f"unknown tool: {name}")


def classify_turn(
    state: StabilityState,
    response: dict[str, Any] | None,
    snapshot_before: WorkspaceSnapshot,
    snapshot_after: WorkspaceSnapshot,
    api_error: str | None = None,
) -> Classification:
    if api_error:
        state.api_errors += 1
        return Classification("API_ERROR", api_error)

    if response is None:
        return Classification("API_ERROR", "missing response")

    msg = response.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    response_fp = fingerprint({"content": content, "tool_calls": tool_calls})
    state.response_fingerprints[response_fp] += 1

    if state.response_fingerprints[response_fp] >= 3:
        return Classification("LOOP", "same assistant response repeated")

    if tool_calls:
        for tool_call in tool_calls:
            try:
                name, args, _ = normalize_tool_call(tool_call)
            except ValueError as exc:
                state.invalid_tool_calls += 1
                return Classification("INVALID_TOOL_CALL", str(exc))
            sig = fingerprint({"name": name, "args": args})
            state.tool_signatures[sig] += 1
            if state.tool_signatures[sig] >= 3 and snapshot_after.score <= snapshot_before.score:
                return Classification("LOOP", "same tool call repeated without artifact growth")

    state.recent_scores.append(snapshot_after.score)
    if (
        len(state.recent_scores) == state.recent_scores.maxlen
        and len(set(state.recent_scores)) == 1
        and not tool_calls
    ):
        return Classification("NO_PROGRESS", "assistant stopped without workspace progress")

    return Classification()


def final_classification(
    workspace: Path,
    validation_passed: bool = False,
    required_groups: dict[str, tuple[str, ...]] | None = None,
) -> Classification:
    snapshot = snapshot_workspace(workspace)
    coverage = artifact_coverage(snapshot, required_groups)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        return Classification("PARTIAL_OUTPUT", f"missing artifact groups: {', '.join(missing)}")
    if not validation_passed:
        return Classification("PARTIAL_OUTPUT", "validation has not passed")
    return Classification()


class ServerProcess:
    def __init__(self, command: list[str], log_path: Path):
        self.command = command
        self.log_path = log_path
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.open("a", encoding="utf-8")
        self.proc = subprocess.Popen(
            self.command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=20)


def wait_for_server(base_url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    health_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server health check timed out: {health_url}")


def run_benchmark(args: argparse.Namespace) -> int:
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not prompt:
        raise SystemExit("Provide --prompt or --prompt-file.")
    required_groups = requested_artifact_groups(prompt)

    run_dir = Path(args.log_dir) / utc_stamp()
    request_log = run_dir / "api.jsonl"
    tool_log = run_dir / "tools.jsonl"
    summary_path = run_dir / "summary.json"
    workspace = Path(args.workspace) if args.workspace else Path(tempfile.mkdtemp(prefix="rapidmlx-agentic-"))
    workspace.mkdir(parents=True, exist_ok=True)

    server: ServerProcess | None = None
    if args.start_server:
        if not args.server_command:
            raise SystemExit("--start-server requires --server-command.")
        command = args.server_command
        server = ServerProcess(command, run_dir / "server.log")
        server.start()

    state = StabilityState()
    classification = Classification()
    audit_retries = 0
    loop_retries = 0
    validation_passed = False
    system_message = {
        "role": "system",
        "content": (
            "You are an autonomous coding agent. Use tools to create and validate "
                f"the requested project in this workspace: {workspace}. All file paths "
                "passed to tools must be relative to that workspace. Keep going until "
                "the workspace contains the requested artifacts and tests. Use at most "
            "8 tool calls per response. Do not repeat inspection commands or read "
            "the same files multiple times. Prefer one validation command over many "
            "read-only commands once artifacts exist. Every tool call must satisfy "
            "the JSON schema exactly. For write calls, include both path and complete "
            "content, with path before content. For large files, use one bash heredoc "
            "command instead of an incomplete write call."
            " For underspecified app/domain requests, choose the smallest complete "
                "implementation that satisfies the requested artifact categories; use one "
                "resource unless the user explicitly names more. Keep imports relative to "
                "the actual file locations. Validation commands must not be piped through "
                "head/tail or followed by echo commands that hide the exit status. Prefer "
                "the project package manager's normal test command for validation instead "
                "of fragile glob filters."
        ),
    }
    user_message = {"role": "user", "content": prompt}
    messages: list[dict[str, Any]] = [system_message, user_message]

    try:
        wait_for_server(args.base_url, args.server_timeout)
        client = httpx.Client(timeout=args.api_timeout)
        for turn in range(args.max_turns):
            snapshot_before = snapshot_workspace(workspace)
            payload = {
                "model": args.model,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }
            api_error = None
            response = None
            try:
                started = time.time()
                resp = client.post(f"{args.base_url}/chat/completions", json=payload)
                elapsed_ms = (time.time() - started) * 1000
                resp.raise_for_status()
                response = resp.json()
                append_jsonl(
                    request_log,
                    {
                        "turn": turn,
                        "elapsed_ms": elapsed_ms,
                        "request": payload,
                        "response": response,
                    },
                )
            except Exception as exc:
                api_error = str(exc)
                append_jsonl(
                    request_log,
                    {"turn": turn, "request": payload, "api_error": api_error},
                )

            snapshot_after_api = snapshot_workspace(workspace)
            classification = classify_turn(
                state, response, snapshot_before, snapshot_after_api, api_error=api_error
            )
            if classification.category:
                if (
                    classification.category == "LOOP"
                    and loop_retries < args.max_loop_retries
                ):
                    loop_retries += 1
                    snapshot = snapshot_workspace(workspace)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Loop detected. The previous response was rejected because "
                                f"{classification.reason}. Current artifact coverage: "
                                f"{artifact_coverage(snapshot, required_groups)}. Current file count: "
                                f"{len(snapshot.files)}. Current files: "
                                f"{list(snapshot.files)}. Do not rewrite existing paths "
                                "with the same content. Missing artifact groups are "
                                f"{[group for group, present in artifact_coverage(snapshot, required_groups).items() if not present]}. "
                                "Create one file for a missing artifact group next. "
                                "If artifacts are complete, run at most one validation command "
                                "or give the final answer. If validation already failed, make "
                                "one targeted file change."
                            ),
                        }
                    )
                    classification = Classification()
                    continue
                break

            msg = response["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            messages.append(msg)
            if not tool_calls:
                classification = final_classification(
                    workspace, validation_passed, required_groups
                )
                if (
                    classification.category == "PARTIAL_OUTPUT"
                    and audit_retries < args.max_audit_retries
                ):
                    audit_retries += 1
                    snapshot = snapshot_workspace(workspace)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Completion audit failed. The original request is not done. "
                                f"Current artifact coverage: {artifact_coverage(snapshot, required_groups)}. "
                                f"Current files: {list(snapshot.files)}. "
                                "Use one or more tool calls now to create the missing requested "
                                "artifacts, then validate. Do not answer with prose until the "
                                "audit passes."
                            ),
                        }
                    )
                    classification = Classification()
                    continue
                break

            parsed_tool_calls: list[tuple[dict[str, Any], str, dict[str, Any], str]] = []
            for tool_call in tool_calls:
                try:
                    name, tool_args, call_id = normalize_tool_call(tool_call)
                except Exception as exc:
                    classification = Classification("INVALID_TOOL_CALL", str(exc))
                    parsed_tool_calls = []
                    break
                parsed_tool_calls.append((tool_call, name, tool_args, call_id))

            if classification.category:
                break

            current_audit = final_classification(
                workspace, validation_passed, required_groups
            )
            if (
                current_audit.category == "PARTIAL_OUTPUT"
                and current_audit.reason == "validation has not passed"
                and parsed_tool_calls
                and all(is_read_only_tool(name, tool_args) for _, name, tool_args, _ in parsed_tool_calls)
            ):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The artifact audit now passes, but validation has not passed. "
                            "Do not inspect or read files again. Your next response must be "
                            "one bash tool call that runs the project's test or typecheck "
                            "command with a real non-piped exit status. If it fails, make "
                            "one targeted file change."
                        ),
                    }
                )
                continue

            for tool_call, name, tool_args, call_id in parsed_tool_calls:
                try:
                    result = execute_tool(name, tool_args, workspace, args.tool_timeout)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                if (
                    name == "bash"
                    and isinstance(tool_args.get("command"), str)
                    and is_validation_command(tool_args["command"])
                    and validation_result_passed(result)
                ):
                    validation_passed = True
                validation_failed = (
                    name == "bash"
                    and isinstance(tool_args.get("command"), str)
                    and is_validation_command(tool_args["command"])
                    and not validation_result_passed(result)
                )
                current_snapshot = snapshot_workspace(workspace)
                coverage = artifact_coverage(current_snapshot, required_groups)
                result["workspace"] = {
                    "file_count": len(current_snapshot.files),
                    "dir_count": len(current_snapshot.dirs),
                    "files": list(current_snapshot.files)[:200],
                    "artifact_coverage": coverage,
                    "missing_artifact_groups": [
                        group for group, present in coverage.items() if not present
                    ],
                    "progress_score_delta": current_snapshot.score - snapshot_before.score,
                }

                append_jsonl(
                    tool_log,
                    {
                        "turn": turn,
                        "tool_call": tool_call,
                        "result": result,
                        "snapshot": current_snapshot.__dict__,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id") or call_id),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                if validation_failed:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Validation failed. Use the validation output above. "
                                "Your next response must be one targeted repair tool call. "
                                "Use write or edit for source/test/config failures. If the "
                                "failure is a missing runtime package and the manifest already "
                                "declares it, use one package-manager install command instead "
                                "of editing the manifest. Do not inspect files again unless "
                                "the failing file path is absent from the output. "
                                "If validation says no tests matched or no tests were found, "
                                "fix the test discovery command or test file locations/names."
                            ),
                        }
                    )
                elif result.get("ok") is False:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous tool call failed. Do not repeat the same "
                                "tool call. Use the error output above to choose a different "
                                "valid tool call. If an edit failed because old text was not "
                                "found, use write with complete corrected file content for "
                                "that path or run one targeted read of that exact path before "
                                "editing."
                            ),
                        }
                    )
                if classification.category:
                    break
            if message_char_count(messages) > args.compact_char_budget:
                messages = compact_messages(
                    system_message,
                    user_message,
                    workspace,
                    f"message history exceeded {args.compact_char_budget} characters",
                    required_groups,
                )
            if final_classification(workspace, validation_passed, required_groups).category is None:
                classification = Classification()
                break
            if classification.category:
                break
        else:
            classification = Classification("LOOP", "max turns reached before completion")

        if not classification.category:
            classification = final_classification(
                workspace, validation_passed, required_groups
            )

    finally:
        if server:
            server.stop()
        if args.clean_workspace and not args.workspace:
            shutil.rmtree(workspace, ignore_errors=True)

    summary = {
        "classification": classification.__dict__,
        "workspace": str(workspace),
        "artifact_coverage": artifact_coverage(snapshot_workspace(workspace), required_groups),
        "required_artifact_groups": sorted(required_groups),
        "snapshot": snapshot_workspace(workspace).__dict__,
        "logs": {
            "server": str(run_dir / "server.log"),
            "api": str(request_log),
            "tools": str(tool_log),
            "summary": str(summary_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if classification.category is None else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("--log-dir", default="logs/agentic-stability")
    parser.add_argument("--workspace")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--tool-timeout", type=int, default=120)
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--max-audit-retries", type=int, default=3)
    parser.add_argument("--max-loop-retries", type=int, default=12)
    parser.add_argument("--compact-char-budget", type=int, default=60000)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-command", nargs=argparse.REMAINDER)
    parser.add_argument("--clean-workspace", action="store_true")
    return parser


def main() -> int:
    return run_benchmark(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
