# SPDX-License-Identifier: Apache-2.0
"""Chat completion endpoints — /v1/chat/completions."""

import asyncio
import gc
import json
import logging
import re
import shlex
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceLogProbs,
    FunctionCall,
    TokenLogProb,
    ToolCall,
    Usage,
)
from ..api.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_schema_for_guided,
    parse_json_output,
)
from ..api.utils import (
    clean_output_text,
    extract_json_from_response,
    extract_multimodal_content,
    sanitize_output,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import GenerationOutput
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _TOOL_CALL_JSON_RETRY_PROMPT,
    _TOOL_CALL_REQUIRED_RETRY_PROMPT,
    _TOOL_CONTINUATION_REPEATED_TOOL_PROMPT,
    _TOOL_CONTINUATION_RETRY_PROMPT,
    _TOOL_USE_SYSTEM_SUFFIX,
    _append_tool_continuation_prompt,
    _assistant_tool_call_path_signature,
    _build_usage,
    _disconnect_guard,
    _extract_token_logprob,
    _inject_json_instruction,
    _last_assistant_tool_call_path_signature,
    _last_assistant_tool_call_signature,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
    _resolve_top_p,
    _tool_call_path_signature,
    _tool_call_signature,
    _validate_model_name,
    _validate_tool_call_params,
    _wait_with_disconnect,
    get_engine,
    get_usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_REPETITION_WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")
_AGENTIC_VALIDATION_COMMAND_RE = re.compile(
    r"(^|[;&|()\s])("
    r"bun\s+(run\s+)?test|"
    r"npm\s+(run\s+)?(test|lint|build|typecheck)|"
    r"pnpm\s+(run\s+)?(test|lint|build|typecheck)|"
    r"yarn\s+(run\s+)?(test|lint|build|typecheck)|"
    r"pytest|python\s+-m\s+pytest|"
    r"go\s+test|cargo\s+test|mvn\s+test|gradle(w)?\s+test|"
    r"tsc(\s|$)|eslint|ruff|mypy"
    r")\b",
    re.IGNORECASE,
)
_TOOL_TEXT_REPETITION_MIN_WORDS = 32
_TOOL_TEXT_REPETITION_MIN_COUNT = 24
_TOOL_TEXT_REPETITION_RATIO = 0.60
_TOOL_TEXT_BEFORE_TOOL_CALL_MAX_CHARS = 4096
_AGENTIC_TEXT_BEFORE_TOOL_CALL_MAX_CHARS = 20000
_TOOL_CALL_REPEAT_BUFFER_MAX_ARGUMENT_CHARS = 65536
_STREAM_IDLE_TIMEOUT_SECONDS = 60.0
_AGENTIC_MAX_TOOL_RESULTS_BEFORE_DIAGNOSTIC = 8
_AGENTIC_MAX_TOOL_RESULTS_AFTER_FAILURE_BEFORE_DIAGNOSTIC = 6
_AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC = 2
_AGENTIC_RETRY_MAX_TOKENS = 4096
_AGENTIC_NO_TOOL_RETRY_MAX_TOKENS = _AGENTIC_RETRY_MAX_TOKENS
_PARTIAL_TOOL_PATH_RE = re.compile(
    r'"(?:filePath|filepath|file_path|path)"\s*:\s*"((?:\\.|[^"\\])*)"',
)
_AGENTIC_COMPLETION_KEYWORDS = (
    "create",
    "implement",
    "modify",
    "edit",
    "write",
    "fix",
    "build",
    "test",
    "validate",
    "project",
)
_AGENTIC_TOOL_USE_SYSTEM_SUFFIX = (
    "\n\nFor coding-agent tasks that require creating or modifying a project, "
    "you MUST keep using tools until the requested work is complete. If the user "
    "asked for validation, do not give a final answer until tool output shows the "
    "requested validation succeeded. If validation fails, use the reported errors "
    "to change files before running another diagnostic command. Do not answer with "
    "prose when a required tool action remains. Once the requested work is complete "
    "and the latest validation succeeded, stop calling tools and answer with a "
    "concise final summary."
)
_AGENTIC_REPAIR_USER_PROMPT = (
    "The previous validation/diagnostic tool output shows the project is still "
    "broken. Your entire next response must be one valid tool call only: write, "
    "edit, or bash if a shell command is required to create directories, install "
    "dependencies, or run a targeted repair command. Do not output markdown, "
    "planning text, summaries, apologies, or any prose. Do not run the same "
    "diagnostic command again yet. Use write or edit now to fix the reported "
    "files, configuration, dependencies, or test failures. "
    "Keep the original user request in scope: if it named artifact categories "
    "such as source modules, tests, migrations, seed data, routes, controllers, "
    "or configuration, the file inventory must include those categories before "
    "you give a final answer. If the request asked for unit tests for each "
    "service, create tests that exercise each service layer, not only "
    "controllers, routes, or models. "
    "Fix the exact latest error before making unrelated changes. Do not expand scope "
    "with new features, entities, files, or frameworks unless the user request or "
    "the current validation error requires it. Read the latest tool output first; "
    "if a write/edit failed or an expected file is missing from the diagnostic "
    "inventory, create the missing parent directories or correct the path before "
    "continuing. If an edit failed because exact text was not found or produced no "
    "changes, or because oldText was not unique, use write with the complete "
    "corrected file content instead of guessing another oldText block. If validation reports that no tests were found, create test files "
    "using the project's test naming convention before running validation again. "
    "If validation already names the failing file, module, export, loader, or "
    "initialization error, do not use a read-only shell command such as cat, sed, "
    "ls, find, or grep as your next action unless the exact failing file is still "
    "unknown. If the latest tool output is source file contents after a failed "
    "validation, use write or edit to repair that file now. "
    "If the diagnostic inventory contains only manifests or root documentation "
    "while the requested task requires implementation files, create the missing "
    "source and test directory structure now. For module-not-found/import errors, "
    "calculate imports relative to the file being edited and the files that actually "
    "exist on disk; do not guess from the repository root. When a unit test lives "
    "inside a feature or module directory and is testing that feature's service, "
    "import the service/model from the same feature first. Do not import a "
    "sibling feature's service from that unit test unless the failing test is "
    "explicitly testing cross-feature integration. Count path segments when "
    "fixing imports: from src/features/<name>/file.ts to src/config use "
    "../../config, from src/features/<name>/<subdir>/file.ts to src/config use "
    "../../../config, from src/modules/<name>/file.ts to src/config use "
    "../../config, from src/modules/<name>/<subdir>/file.ts to src/config use "
    "../../../config, and from a feature-root file to that same feature's routes "
    "or services use ./routes or ./services. If a test file sits in the same "
    "directory as the source file it imports, use ./name, not ../name. From a test file inside a service "
    "directory to the service beside it, use ./serviceName, not ../../services. "
    "From src/modules/<name>/services/*.test.ts to the same module's models, use "
    "../models/modelName, not ../../models/modelName. "
    "Do not put ../src or ./src in a "
    "relative import from a file already under src; imports are relative to the "
    "current file, not to the project root. If the user asked for unit tests and "
    "you created integration tests that fail from real database or cross-module "
    "setup, remove or replace those integration tests with isolated unit tests "
    "for the requested service layer. For HTTP route tests, do not call router "
    "registration methods such as router.get(), router.post(), router.put(), or "
    "router.delete() as if they execute requests; mount the router on a test app "
    "and use the framework's request helper, or test handlers as plain functions. "
    "If a route registration error says a callback function is required but an "
    "object was provided, change the route module to pass actual handler "
    "functions, not controller/service objects. If validation says a "
    "package cannot be found and the manifest already lists that dependency, run "
    "the package manager install command instead of editing tests to mock the "
    "missing dependency. For named import/export "
    "errors, only import symbols confirmed to exist in the dependency and remove or "
    "replace any invalid symbol usages. If validation says a package module does "
    "not export a named symbol, do not move that symbol to another package unless "
    "the latest tool output confirms that package exports it. If the missing "
    "symbol is optional validation or query-helper code, prefer deleting that "
    "symbol usage or replacing it with plain local checks over importing it from "
    "an unrelated framework package. When the same "
    "missing export appears in multiple files, fix every importer using that "
    "package consistently instead of changing one file at a time. Do not leave "
    "duplicate class fields, "
    "duplicate functions, or duplicate declarations after edits. For missing runtime "
    "packages, update the project manifest or install the package before retrying. "
    "For unit tests, do not require real external services such as databases, "
    "network APIs, queues, or servers unless the user explicitly asked for an "
    "integration test. If validation shows connection refused, model not "
    "initialized, static data-model methods that require registration, or "
    "similar external setup failures, change the test setup to mock the boundary "
    "or initialize and register the data models in a local disposable test "
    "instance before tests run. For service unit tests, call the service methods "
    "under test; do not replace the service test with direct ORM model CRUD calls "
    "unless that is the service API. For sequelize-typescript tests that use "
    "static model methods, register every model on the local test Sequelize "
    "instance with models: [Model] or sequelize.addModels([Model]) before the "
    "first static call. For sequelize-typescript model tests, every "
    "declared field used as a persisted column, including createdAt and "
    "updatedAt, must have the appropriate sequelize-typescript decorator such "
    "as @Column, @CreatedAt, or @UpdatedAt. Do not import production database singletons into "
    "unit tests when those singletons connect to external services or can be "
    "closed across tests; instead construct an isolated in-memory or local test "
    "instance inside the test file and close it only after all tests that use it. "
    "For sequelize-typescript unit or model tests, prefer creating the local "
    "Sequelize instance directly in the test file and adding only the models under "
    "test; do not import the production config/database singleton. "
    "If validation says a connection manager, pool, client, or local test "
    "instance was already closed, stop reusing the closed instance and create a "
    "fresh instance per test or close it in suite cleanup after the final test. If "
    "validation raises ReferenceError for a variable that is not defined, do "
    "not assume globals; import it from an existing module or define it in the "
    "failing file's setup. If validation shows mocked methods are undefined, "
    "mock cleanup APIs do not exist, or response/request test doubles are "
    "missing methods, replace unsupported mock-runner calls with simple "
    "hand-written fakes or functions supported by the current test runner. When "
    "the same test-double failure appears in multiple test files, fix every "
    "affected test file in one pass. Do not leave imports pointing at files "
    "that are absent from the inventory. If validation says a sibling "
    "feature/module file cannot be found, either create the complete missing "
    "sibling feature slice if it is part of the current requested project, or "
    "remove the optional association/import and make the current feature "
    "self-contained. If validation says a model imports an optional sibling "
    "association that is absent, remove the optional association/import when the "
    "user did not explicitly request that sibling domain; do not create a partial "
    "sibling model only to satisfy the import. If validation says a config module such as config/database "
    "cannot be found and the inventory has no matching config file, create the "
    "missing config file instead of repeatedly changing relative import depth. "
    "Do not keep tests blocked by an import to a feature directory that has no "
    "model, service, route, migration, seed, and tests of its own. "
    "Only after changing files may you run validation again."
    " If a module loader says no default export is defined or an export does "
    "not satisfy a filename, fix the exporting module or explicit registration "
    "that the loader uses; do not keep changing unrelated tests. If a loader "
    "discovers files by filename and expects either a default export or a named "
    "export matching the filename stem, make every discovered file follow that "
    "same convention consistently. If validation "
    "says an export named symbol was not found and suggests importing default, "
    "inspect both the exporting file and all importers, then make the export "
    "style consistent in one file change instead of toggling named and default "
    "imports across retries. If validation "
    "reports a value cannot be accessed before initialization, inspect circular "
    "imports and move shared setup or lazy initialization to a module that does "
    "not import the dependent feature back. If a model/ORM loader reports that a "
    "file has no default export or no export matching its filename, rewrite that "
    "model file so it exports a concrete class in the loader's expected style and "
    "update related imports consistently."
)
_AGENTIC_FINAL_USER_PROMPT = (
    "The requested work is complete and the latest validation passed. "
    "Do not call any tools. Answer with a concise final summary only."
)
_AGENTIC_REPEATED_TOOL_PROMPT = (
    "You repeated a validation or diagnostic tool call without fixing files. "
    "That is a loop. Use the existing tool output. Your next response must be "
    "a write or edit tool call that changes the project files needed to fix the "
    "failure. Do not call bash again until after a file change. If the repeated "
    "tool calls are changing the same file after an import/export, loader, or "
    "initialization error, stop toggling that file. Either write the complete "
    "file once with a consistent export style, or change the related loader, "
    "registration, config, or importer that is actually enforcing the convention."
)
_AGENTIC_DESTRUCTIVE_COMMAND_REPAIR_PROMPT = (
    "Your previous shell tool call would delete source or generated project "
    "artifacts that are still needed for the requested implementation. Do not "
    "delete or reset the project tree. Your next response must be one valid tool "
    "call only: write or edit the specific missing or broken files, or run a "
    "non-destructive command only if it is required to create directories, "
    "install dependencies, or validate after a file change."
)
_AGENTIC_REPEATED_PATH_REPAIR_PROMPT = (
    "You repeatedly changed the same path after validation or diagnostic output "
    "still showed the project was incomplete or broken. That is a loop. Use the "
    "latest diagnostic output and file inventory. Your next response must be one "
    "valid tool call only. If the latest validation error still names that same "
    "path, use write with the complete corrected file content; do not use edit "
    "on that path again. If required implementation or test files are missing "
    "from the inventory, create those missing files now. "
    "If the latest stack trace points at compiler, transpiler, test runner, or "
    "runtime configuration instead of the edited source logic, change the "
    "relevant config, manifest, or dependency setup rather than editing that "
    "same source file again. "
    "If the same import/export mismatch repeats, stop alternating named and "
    "default exports. Inspect the runtime loader convention and every importer, "
    "then rewrite the exporting file and affected imports consistently. If a "
    "loader accepts either a default export or filename-matching named export "
    "but the same file still fails after multiple rewrites, change the loader "
    "registration/config to register explicit model classes or update all "
    "importers instead of rewriting that same model file again. If an "
    "edit failed because oldText was missing or edits overlapped, use write with "
    "the complete corrected file content on the next attempt. "
    "If the latest error names a different file, edit that file. If you cannot "
    "identify the next file to change, run a targeted validation command that "
    "prints the exact failing file and error."
)
_AGENTIC_MISSING_ARTIFACT_PROMPT = (
    "The original request named artifact categories that are still absent from "
    "the file inventory or from the paths you created. Your next response must "
    "be one valid tool call only. "
    "Inspect the original request and the current file inventory, then create the "
    "missing requested artifact files. Do not rewrite or duplicate a category "
    "that is already present while another requested category is still absent; "
    "for example, if seed files exist but migration files do not, create the "
    "migration files next instead of editing seed files again. If the request asked for unit tests for "
    "each service, create service-layer tests for every service, not only "
    "controller, route, or model tests. If the request asked for a vertical or "
    "feature-sliced architecture and the inventory has root-level service, model, "
    "route, controller, migration, or seed files, regroup them under feature or "
    "domain directories so each feature owns its model/service/route/test and "
    "related data setup instead of leaving shared root folders as the primary "
    "structure. Add missing categories to the existing feature/domain slices "
    "first; do not invent a new feature, domain, entity, or resource only to "
    "satisfy a missing category unless the original request named it or no "
    "feature slice exists yet. For a REST/API request with no route files, create "
    "route/router files and wire them to the existing feature service before "
    "adding another model or test. Do not give a final answer yet. After "
    "creating the missing artifacts, run validation again."
)


def _agentic_missing_artifact_prompt(messages: list) -> str:
    missing = sorted(_agentic_requested_artifacts_missing(messages))
    if not missing:
        return _AGENTIC_MISSING_ARTIFACT_PROMPT
    missing_text = ", ".join(missing)
    return (
        _AGENTIC_MISSING_ARTIFACT_PROMPT
        + "\nMissing requested artifact categories: "
        + missing_text
        + ". Create files for one or more of these categories next, and do not "
        "spend the next tool call on categories that are already present."
    )
_AGENTIC_DIAGNOSTIC_COMMAND = (
    "printf 'AGENTIC_DIAGNOSTIC\\n'; "
    "pwd; "
    "printf 'FILES\\n'; "
    "find . -maxdepth 5 -path './node_modules' -prune -o -path './.git' -prune -o "
    "-path './dist' -prune -o -path './build' -prune -o "
    "-type f -print | sort | sed 's#^./##' | head -200; "
    "printf 'MANIFESTS\\n'; "
    "ls package.json pyproject.toml setup.py requirements.txt go.mod Cargo.toml "
    "pom.xml build.gradle 2>/dev/null || true; "
    "printf 'VALIDATION\\n'; status=0; "
    "if [ -f package.json ]; then "
    "if [ ! -d node_modules ]; then echo 'DEPENDENCIES_MISSING: node_modules not found; run the package manager install command before editing imports or tests'; fi; "
    "if command -v bun >/dev/null 2>&1; then echo 'RUNNING_VALIDATION: bun test'; bun test || status=$?; "
    "elif command -v npm >/dev/null 2>&1; then echo 'RUNNING_VALIDATION: npm test'; npm test || status=$?; "
    "else echo 'VALIDATION_FAILED: no js package manager available'; status=1; fi; "
    "elif [ -f pyproject.toml ] || [ -f setup.py ] || [ -f requirements.txt ]; then "
    "if command -v pytest >/dev/null 2>&1; then echo 'RUNNING_VALIDATION: pytest'; pytest || status=$?; "
    "elif command -v python >/dev/null 2>&1; then echo 'RUNNING_VALIDATION: python -m pytest'; python -m pytest || status=$?; "
    "else echo 'VALIDATION_FAILED: no python test runner available'; status=1; fi; "
    "elif [ -f go.mod ]; then echo 'RUNNING_VALIDATION: go test ./...'; go test ./... || status=$?; "
    "elif [ -f Cargo.toml ]; then echo 'RUNNING_VALIDATION: cargo test'; cargo test || status=$?; "
    "elif [ -f pom.xml ]; then echo 'RUNNING_VALIDATION: mvn test'; mvn test || status=$?; "
    "elif [ -f build.gradle ]; then echo 'RUNNING_VALIDATION: ./gradlew test'; ./gradlew test || status=$?; "
    "else echo 'VALIDATION_FAILED: no recognized project manifest'; status=1; "
    "fi; "
    "if [ \"$status\" -eq 0 ]; then echo 'VALIDATION_PASSED'; "
    "else echo \"VALIDATION_FAILED: validation command exited with status $status\"; "
    "echo 'NEXT_ACTION: use write or edit to fix the latest validation error before any more diagnostic or final answer'; "
    "echo 'NEXT_ACTION: if required files are absent from FILES, create missing parent directories and files first'; "
    "echo 'NEXT_ACTION: if validation says no tests were found, create correctly named test files before rerunning validation'; "
    "fi; exit \"$status\""
)
_AGENTIC_DEPENDENCY_INSTALL_COMMAND = (
    "status=0; "
    "if [ -f package.json ]; then "
    "if command -v bun >/dev/null 2>&1; then bun install || status=$?; "
    "elif command -v npm >/dev/null 2>&1; then npm install || status=$?; "
    "else echo 'VALIDATION_FAILED: no js package manager available'; status=1; fi; "
    "elif [ -f requirements.txt ]; then python -m pip install -r requirements.txt || status=$?; "
    "elif [ -f pyproject.toml ] || [ -f setup.py ]; then python -m pip install -e . || status=$?; "
    "else echo 'VALIDATION_FAILED: no recognized dependency manifest'; status=1; fi; "
    "if [ \"$status\" -eq 0 ]; then echo 'DEPENDENCIES_INSTALLED'; "
    "else echo \"VALIDATION_FAILED: dependency install exited with status $status\"; fi"
)
_AGENTIC_EXTRA_JS_DEPENDENCY_INSTALL_COMMAND = (
    "status=0; "
    "if [ -f package.json ]; then "
    "if command -v bun >/dev/null 2>&1; then bun add {package} || status=$?; "
    "elif command -v npm >/dev/null 2>&1; then npm install {package} || status=$?; "
    "else echo 'VALIDATION_FAILED: no js package manager available'; status=1; fi; "
    "else echo 'VALIDATION_FAILED: package.json not found for js dependency install'; status=1; fi; "
    "if [ \"$status\" -eq 0 ]; then echo 'DEPENDENCIES_INSTALLED'; "
    "else echo \"VALIDATION_FAILED: dependency install exited with status $status\"; fi"
)
def _message_content_text(message) -> str:
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", "")
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _message_tool_call_text(message) -> str:
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not tool_calls:
        return ""

    parts: list[str] = []
    for tool_call in tool_calls:
        function = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        if function is None:
            continue
        name = (
            function.get("name")
            if isinstance(function, dict)
            else getattr(function, "name", None)
        )
        arguments = (
            function.get("arguments")
            if isinstance(function, dict)
            else getattr(function, "arguments", None)
        )
        if name:
            parts.append(str(name))
        if arguments:
            parts.append(str(arguments))
    return "\n".join(parts)


def _agentic_message_role(message) -> str | None:
    return (
        message.get("role")
        if isinstance(message, dict)
        else getattr(message, "role", None)
    )


def _agentic_function_payload(function) -> tuple[str, object]:
    if function is None:
        return "", None
    if isinstance(function, dict):
        return str(function.get("name") or ""), function.get("arguments")
    return str(getattr(function, "name", "") or ""), getattr(function, "arguments", None)


def _agentic_tool_call_command_text(message) -> str:
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not tool_calls:
        return ""

    commands: list[str] = []
    for tool_call in tool_calls:
        function = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        name, arguments = _agentic_function_payload(function)
        if name not in {"bash", "shell", "exec", "run_command"}:
            continue
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                command = decoded.get("command") or decoded.get("cmd")
                if isinstance(command, str):
                    commands.append(command)
            else:
                commands.append(arguments)
        elif isinstance(arguments, dict):
            command = arguments.get("command") or arguments.get("cmd")
            if isinstance(command, str):
                commands.append(command)
    return "\n".join(commands)


def _agentic_tool_result_has_successful_validation_command(
    messages: list, tool_index: int
) -> bool:
    tool_text = _message_content_text(messages[tool_index]).lower()
    has_success_evidence = bool(
        re.search(r"command exited with code\s+0\b", tool_text)
        or (
            re.search(r"\b\d+\s+pass\b", tool_text)
            and re.search(r"\b0\s+fail\b", tool_text)
        )
        or re.search(r"\b0\s+fail\b", tool_text)
    )
    if not has_success_evidence:
        return False
    for previous in reversed(messages[:tool_index]):
        if _agentic_message_role(previous) != "assistant":
            continue
        command_text = _agentic_tool_call_command_text(previous)
        return bool(_AGENTIC_VALIDATION_COMMAND_RE.search(command_text))
    return False


def _agentic_tool_result_has_failed_validation_command(
    messages: list, tool_index: int
) -> bool:
    tool_text = _message_content_text(messages[tool_index]).lower()
    has_failure_evidence = (
        "validation_failed" in tool_text
        or re.search(r"command exited with code\s+(?!0\b)\d+", tool_text)
        or re.search(r"\b[1-9]\d*\s+fail\b", tool_text)
        or "error:" in tool_text
        or "referenceerror" in tool_text
        or "couldn't find" in tool_text
        or "cannot find" in tool_text
        or "is not defined" in tool_text
        or "no test files found" in tool_text
        or "no tests found" in tool_text
    )
    if not has_failure_evidence:
        return False
    for previous in reversed(messages[:tool_index]):
        if _agentic_message_role(previous) != "assistant":
            continue
        command_text = _agentic_tool_call_command_text(previous)
        return bool(_AGENTIC_VALIDATION_COMMAND_RE.search(command_text))
    return False


def _agentic_message_text(message) -> str:
    return "\n".join(
        part
        for part in (_message_content_text(message), _message_tool_call_text(message))
        if part
    )


def _agentic_completion_needs_verification(messages: list) -> bool:
    transcript = "\n".join(
        _agentic_message_text(message)
        for message in messages
        if (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        not in {"system", "developer"}
    ).lower()
    return any(keyword in transcript for keyword in _AGENTIC_COMPLETION_KEYWORDS)


def _agentic_verification_present(messages: list) -> bool:
    return _agentic_validation_evidence_present(messages)


_AGENTIC_REQUESTED_ARTIFACT_TERMS = {
    "migration": ("migration", "migrations"),
    "seeder": ("seed", "seeder", "seeders", "seeds"),
    "model": ("model", "models"),
    "service": ("service", "services"),
    "controller": ("controller", "controllers"),
    "route": ("api", "endpoint", "endpoints", "rest", "route", "routes", "router"),
    "middleware": ("middleware", "middlewares"),
    "test": ("test", "tests", "spec", "specs"),
    "vertical_slice": (
        "feature slice",
        "feature sliced",
        "vertical slice",
        "vertically sliced",
    ),
}

_AGENTIC_REQUESTED_ARTIFACT_PATTERNS = {
    "vertical_slice": (r"\bvertical\s+sliced\b",),
}


def _agentic_requested_artifact_terms(messages: list) -> set[str]:
    text = "\n".join(
        _message_content_text(message)
        for message in messages
        if _agentic_message_role(message) == "user"
    ).lower()
    requested: set[str] = set()
    for term, variants in _AGENTIC_REQUESTED_ARTIFACT_TERMS.items():
        exact_match = any(
            re.search(rf"\b{re.escape(variant)}\b", text) for variant in variants
        )
        pattern_match = any(
            re.search(pattern, text)
            for pattern in _AGENTIC_REQUESTED_ARTIFACT_PATTERNS.get(term, ())
        )
        if exact_match or pattern_match:
            requested.add(term)
    return requested


def _agentic_user_request_text(messages: list) -> str:
    return "\n".join(
        _message_content_text(message)
        for message in messages
        if _agentic_message_role(message) == "user"
    ).lower()


def _agentic_created_artifact_paths(messages: list) -> list[str]:
    paths: list[str] = []
    for message in messages:
        signature = _assistant_tool_call_path_signature(message)
        if signature:
            _, path = signature
            paths.append(path.lower())

        if _agentic_message_role(message) != "tool":
            continue
        content = _message_content_text(message)
        for match in re.finditer(
            r"(?:to|in)\s+([^\s]+(?:\.[A-Za-z0-9_./-]+)?)",
            content,
        ):
            path = match.group(1).strip().rstrip(".:,;")
            if "/" in path or "." in path:
                paths.append(path.lower())
    return paths


def _agentic_requested_artifacts_missing(messages: list) -> set[str]:
    requested = _agentic_requested_artifact_terms(messages)
    if not requested:
        return set()
    paths = _agentic_created_artifact_paths(messages)
    user_request = _agentic_user_request_text(messages)
    missing: set[str] = set()
    for term in requested:
        if term == "vertical_slice":
            has_feature_owned_path = any(
                re.search(
                    r"(?:^|/)src/(?:modules|features|domains|slices)/[^/]+/",
                    path,
                )
                for path in paths
            )
            has_root_architecture_path = any(
                re.search(
                    r"(?:^|/)src/(?:models|services|routes|controllers|repositories)/",
                    path,
                )
                for path in paths
            )
            if not has_feature_owned_path or has_root_architecture_path:
                missing.add(term)
            continue
        variants = _AGENTIC_REQUESTED_ARTIFACT_TERMS[term]
        if not any(any(variant in path for variant in variants) for path in paths):
            missing.add(term)
    if "service" in requested and "test" in requested:
        service_names = {
            re.sub(r"(?:service|\.service)$", "", Path(path).stem).lower()
            for path in paths
            if "service" in path and "test" not in path and "spec" not in path
        }
        tested_names = {
            re.sub(r"(?:service|\.service|test|spec)$", "", Path(path).stem).lower()
            for path in paths
            if ("test" in path or "spec" in path) and "service" in path
        }
        if service_names and not service_names.issubset(tested_names):
            missing.add("test")
        if (
            re.search(r"\btests?\s+for\s+each\s+service\b", user_request)
            and not tested_names
        ):
            missing.add("test")
    return missing


def _agentic_task_terminal_ready(messages: list) -> bool:
    return (
        _agentic_completion_needs_verification(messages)
        and _agentic_verification_present(messages)
        and not _agentic_failed_validation_present(messages)
        and not _agentic_requested_artifacts_missing(messages)
    )


def _agentic_validation_evidence_present(messages: list) -> bool:
    success_markers = (
        "validation_passed",
        "validation passed",
        "all checks passed",
        "tests passed",
        "build passed",
        "lint passed",
    )
    latest_validation_success: bool | None = None
    for index, message in enumerate(messages):
        if _agentic_message_role(message) != "tool":
            continue
        tool_text = _message_content_text(message).lower()
        if _agentic_tool_result_has_failed_validation_command(messages, index):
            latest_validation_success = False
            continue
        if _agentic_tool_result_has_successful_validation_command(messages, index):
            latest_validation_success = True
            continue
        if any(marker in tool_text for marker in success_markers):
            latest_validation_success = True
    return latest_validation_success is True


def _agentic_failed_validation_present(messages: list) -> bool:
    explicit_failure_markers = (
        "validation_failed",
        "missing script",
        "no test script",
        "no test files found",
        "no tests found",
    )
    failure_markers = (
        "validation_failed",
        "command exited with code",
        "failed",
        " fail",
        "error:",
        "referenceerror",
        "couldn't find",
        "cannot find",
        "is not defined",
        "never used",
        "unknown option",
        "missing script",
        "no test files found",
        "no tests found",
        "no package.json",
        "no changes made to",
        "oldtext must be unique",
        "oldtext was not found",
        "found 2 occurrences",
        "found 3 occurrences",
        "found 4 occurrences",
        "found 5 occurrences",
    )
    success_markers = (
        "validation_passed",
        "validation passed",
        "all checks passed",
        "tests passed",
        "build passed",
        "lint passed",
    )
    for message in reversed(messages):
        if _agentic_message_role(message) != "tool":
            continue
        tool_text = _message_content_text(message).lower()
        if any(marker in tool_text for marker in explicit_failure_markers):
            return True
        if any(marker in tool_text for marker in failure_markers):
            return True
        if any(marker in tool_text for marker in success_markers):
            return False
    transcript = _agentic_transcript(messages).lower()
    return any(marker in transcript for marker in failure_markers)


def _agentic_transcript(messages: list) -> str:
    return "\n".join(
        _agentic_message_text(message)
        for message in messages
        if (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        not in {"system", "developer"}
    )


def _agentic_tool_result_count(messages: list) -> int:
    return sum(
        1
        for message in messages
        if (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        == "tool"
    )


def _agentic_tool_result_is_diagnostic(text: str) -> bool:
    return "agentic_diagnostic" in text


def _last_tool_result_is_agentic_diagnostic(messages: list) -> bool:
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "tool":
            continue
        return _agentic_tool_result_is_diagnostic(
            _message_content_text(message).lower()
        )
    return False


def _last_tool_result_missing_dependency(messages: list) -> str | None:
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "tool":
            continue
        text = _message_content_text(message)
        lower_text = text.lower()
        package_match = re.search(
            r"cannot find package\s+['\"]([^'\"]+)['\"]",
            lower_text,
        )
        if package_match:
            package_name = package_match.group(1)
            if package_name.startswith((".", "/")) or package_name.startswith("file:"):
                return None
            return package_name
        if "dependencies_missing" in lower_text:
            return ""
        package_hint = re.search(
            r"install\s+([@a-z0-9_.\-/]+)\s+(?:package|module|dependency)",
            lower_text,
        )
        if not package_hint:
            package_hint = re.search(
                r"install\s+(?:the\s+)?(?:package|module|dependency)"
                r"(?:\s+manually)?\s*:\s*([@a-z0-9_.\-/]+)",
                lower_text,
            )
        if package_hint:
            return package_hint.group(1)
        if "module_not_found" in lower_text:
            return ""
        module_match = re.search(
            r"cannot find module\s+['\"]([^'\"]+)['\"]", lower_text
        )
        if module_match:
            module_name = module_match.group(1)
            if module_name.startswith((".", "/")) or module_name.startswith("file:"):
                return None
            return module_name
        if re.search(r"install\s+(?:the\s+)?(?:package|module|dependency)", lower_text):
            return ""
        return None
    return None


def _last_tool_result_needs_dependency_install(messages: list) -> bool:
    return _last_tool_result_missing_dependency(messages) is not None


def _latest_tool_result_mentions_path(messages: list, path: str | None) -> bool:
    if not path:
        return False
    normalized_path = path.lower()
    basename = Path(path).name.lower()
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "tool":
            continue
        text = _message_content_text(message).lower()
        return normalized_path in text or bool(basename and basename in text)
    return False


def _agentic_tool_result_count_since_latest_failure(messages: list) -> int:
    count = 0
    seen_failure = False
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "tool":
            continue
        text = _message_content_text(message).lower()
        if _agentic_tool_result_is_diagnostic(text):
            if "validation_failed" in text:
                seen_failure = True
                break
            continue
        if any(
            marker in text
            for marker in (
                "validation_failed",
                "command exited with code",
                "failed",
                " fail",
                "error:",
                "typeerror",
                "referenceerror",
                "undefined is not an object",
                "couldn't find",
                "cannot find",
                "is not defined",
                "never used",
                "unknown option",
                "missing script",
                "no test files found",
                "no tests found",
                "no package.json",
            )
        ):
            seen_failure = True
            break
        count += 1
    return count if seen_failure else 0


def _agentic_max_same_path_tools_since_latest_failure(messages: list) -> int:
    counts: dict[tuple[str, str], int] = {}
    seen_failure = False
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role == "tool":
            text = _message_content_text(message).lower()
            if _agentic_tool_result_is_diagnostic(text):
                if "validation_failed" in text:
                    seen_failure = True
                    break
                continue
            if any(
                marker in text
                for marker in (
                    "validation_failed",
                    "command exited with code",
                    "failed",
                    " fail",
                    "error:",
                    "typeerror",
                    "referenceerror",
                    "undefined is not an object",
                    "couldn't find",
                    "cannot find",
                    "is not defined",
                    "never used",
                    "unknown option",
                    "missing script",
                    "no test files found",
                    "no tests found",
                    "no package.json",
                    "no changes made",
                    "replacement produced identical content",
                )
            ):
                seen_failure = True
                continue
            continue
        signature = _assistant_tool_call_path_signature(message)
        if signature:
            counts[signature] = counts.get(signature, 0) + 1
    return max(counts.values(), default=0) if seen_failure else 0


def _agentic_max_same_command_tools_since_latest_failure(messages: list) -> int:
    counts: dict[str, int] = {}
    seen_failure = False
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role == "tool":
            text = _message_content_text(message).lower()
            if _agentic_tool_result_is_diagnostic(text):
                if "validation_failed" in text:
                    seen_failure = True
                    break
                continue
            if any(
                marker in text
                for marker in (
                    "validation_failed",
                    "failed",
                    " fail",
                    "error:",
                    "typeerror",
                    "referenceerror",
                    "undefined is not an object",
                    "couldn't find",
                    "cannot find",
                    "is not defined",
                    "never used",
                    "unknown option",
                    "missing script",
                    "no test files found",
                    "no tests found",
                    "no package.json",
                )
            ):
                seen_failure = True
                continue
            continue
        if role != "assistant":
            continue
        command_text = _agentic_tool_call_command_text(message).strip()
        if command_text:
            counts[command_text] = counts.get(command_text, 0) + 1
    return max(counts.values(), default=0) if seen_failure else 0


def _agentic_max_same_path_tools_without_validation(messages: list) -> int:
    if _agentic_verification_present(messages) or _agentic_failed_validation_present(
        messages
    ):
        return 0
    counts: dict[tuple[str, str], int] = {}
    for message in messages:
        if _agentic_message_role(message) != "assistant":
            continue
        signature = _assistant_tool_call_path_signature(message)
        if signature:
            counts[signature] = counts.get(signature, 0) + 1
    return max(counts.values(), default=0)


def _last_tool_result_indicates_failure(messages: list) -> bool:
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "tool":
            continue
        text = _message_content_text(message).lower()
        return any(
            marker in text
            for marker in (
                "validation_failed",
                "failed",
                "error:",
                "typeerror",
                "model not initialized",
                "undefined is not an object",
                    "cannot find",
                    "not found",
                    "export named",
                    "no default export",
                    "does not satisfy filename",
                    "before initialization",
                    "no such file",
                "oldtext",
                "no changes made",
                "replacement produced identical content",
                "exit code 1",
                "exited with code 1",
            )
        )
    return False


def _tool_choice_requires_tool_call(tool_choice) -> bool:
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice not in {"none", "auto"}
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {None, "none", "auto"}:
            return False
        return True
    return False


def _tool_names(tools) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump(exclude_none=True)
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _agentic_diagnostic_tool_call() -> ToolCall:
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        function=FunctionCall(
            name="bash",
            arguments=json.dumps(
                {
                    "command": _AGENTIC_DIAGNOSTIC_COMMAND,
                    "timeout": 120,
                }
            ),
        ),
    )


def _agentic_dependency_install_tool_call(package_name: str | None = None) -> ToolCall:
    command = _AGENTIC_DEPENDENCY_INSTALL_COMMAND
    if package_name:
        command = _AGENTIC_EXTRA_JS_DEPENDENCY_INSTALL_COMMAND.format(
            package=shlex.quote(package_name)
        )
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        function=FunctionCall(
            name="bash",
            arguments=json.dumps(
                {
                    "command": command,
                    "timeout": 120,
                }
            ),
        ),
    )


def _is_repetitive_tool_text(text: str) -> bool:
    words = _REPETITION_WORD_RE.findall(text.lower())
    if len(words) < _TOOL_TEXT_REPETITION_MIN_WORDS:
        return False

    tail = words[-96:]
    most_common_count = Counter(tail).most_common(1)[0][1]
    if (
        most_common_count >= _TOOL_TEXT_REPETITION_MIN_COUNT
        and most_common_count / len(tail) >= _TOOL_TEXT_REPETITION_RATIO
    ):
        return True

    for size in (2, 3, 4):
        if len(tail) < size * 8:
            continue
        ngrams = [" ".join(tail[i : i + size]) for i in range(len(tail) - size + 1)]
        ngram_count = Counter(ngrams).most_common(1)[0][1]
        if ngram_count >= 8 and (ngram_count * size) / len(tail) >= 0.50:
            return True

    return False


def _buffered_stream_text(buffered_events: list[tuple]) -> str:
    parts = []
    for event, _ in buffered_events:
        parts.append(getattr(event, "content", None) or "")
        parts.append(getattr(event, "reasoning", None) or "")
    return "".join(parts)


def _tool_call_value(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _buffered_tool_calls_complete(buffered_events: list[tuple]) -> bool:
    """Return True when streamed tool-call chunks assemble to valid JSON."""
    calls: dict[int, dict[str, str | None]] = {}
    for event, _ in buffered_events:
        for tool_call in getattr(event, "tool_calls", None) or []:
            index = _tool_call_value(tool_call, "index")
            if not isinstance(index, int):
                index = len(calls)
            assembled = calls.setdefault(
                index,
                {"id": None, "name": None, "arguments": ""},
            )

            call_id = _tool_call_value(tool_call, "id")
            if call_id:
                assembled["id"] = str(call_id)

            function = _tool_call_value(tool_call, "function", {}) or {}
            name = _tool_call_value(function, "name")
            if name:
                assembled["name"] = str(name)

            arguments = _tool_call_value(function, "arguments")
            if arguments is not None:
                assembled["arguments"] = (assembled["arguments"] or "") + str(arguments)

    if not calls:
        return False

    for assembled in calls.values():
        if not assembled.get("name"):
            return False
        arguments = (assembled.get("arguments") or "").strip()
        if not arguments:
            return False
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return False
        if not isinstance(parsed, dict):
            return False

    return True


def _assembled_tool_call_signatures(
    buffered_events: list[tuple],
) -> list[tuple[str, str]]:
    calls: dict[int, dict[str, str | None]] = {}
    for event, _ in buffered_events:
        for tool_call in getattr(event, "tool_calls", None) or []:
            index = _tool_call_value(tool_call, "index")
            if not isinstance(index, int):
                index = len(calls)
            assembled = calls.setdefault(index, {"name": None, "arguments": ""})

            function = _tool_call_value(tool_call, "function", {}) or {}
            name = _tool_call_value(function, "name")
            if name:
                assembled["name"] = str(name)

            arguments = _tool_call_value(function, "arguments")
            if arguments is not None:
                assembled["arguments"] = (assembled["arguments"] or "") + str(arguments)

    signatures: list[tuple[str, str]] = []
    for assembled in calls.values():
        signature = _tool_call_signature(
            {
                "function": {
                    "name": assembled.get("name"),
                    "arguments": assembled.get("arguments"),
                }
            }
        )
        if signature:
            signatures.append(signature)
    return signatures


def _assembled_tool_call_path_signatures(
    buffered_events: list[tuple],
) -> list[tuple[str, str]]:
    calls: dict[int, dict[str, str | None]] = {}
    for event, _ in buffered_events:
        for tool_call in getattr(event, "tool_calls", None) or []:
            index = _tool_call_value(tool_call, "index")
            if not isinstance(index, int):
                index = len(calls)
            assembled = calls.setdefault(index, {"name": None, "arguments": ""})

            function = _tool_call_value(tool_call, "function", {}) or {}
            name = _tool_call_value(function, "name")
            if name:
                assembled["name"] = str(name)

            arguments = _tool_call_value(function, "arguments")
            if arguments is not None:
                assembled["arguments"] = (assembled["arguments"] or "") + str(arguments)

    signatures: list[tuple[str, str]] = []
    for assembled in calls.values():
        signature = _tool_call_path_signature(
            {
                "function": {
                    "name": assembled.get("name"),
                    "arguments": assembled.get("arguments"),
                }
            }
        )
        if signature:
            signatures.append(signature)
    return signatures


def _assembled_tool_call_functions(
    buffered_events: list[tuple],
) -> list[tuple[str, object]]:
    calls: dict[int, dict[str, object]] = {}
    for event, _ in buffered_events:
        for tool_call in getattr(event, "tool_calls", None) or []:
            index = _tool_call_value(tool_call, "index")
            if not isinstance(index, int):
                index = len(calls)
            assembled = calls.setdefault(index, {"name": None, "arguments": ""})

            function = _tool_call_value(tool_call, "function", {}) or {}
            name = _tool_call_value(function, "name")
            if name:
                assembled["name"] = str(name)

            arguments = _tool_call_value(function, "arguments")
            if arguments is not None:
                assembled["arguments"] = str(assembled["arguments"] or "") + str(
                    arguments
                )

    return [
        (str(assembled.get("name") or ""), assembled.get("arguments"))
        for assembled in calls.values()
        if assembled.get("name")
    ]


def _decoded_tool_arguments(arguments: object) -> object:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (TypeError, ValueError):
            return arguments
    return arguments


def _command_removes_artifact_tree(command: str) -> bool:
    lower_command = command.lower()
    if not re.search(
        r"(?:^|[;&|]\s*)rm\s+-[a-z]*r[a-z]*f?",
        lower_command,
    ):
        return False
    return bool(
        re.search(
            r"(^|[\s'\"])(?:\./)?src(?:/|[\s'\"]|$)|"
            r"/src(?:/|[\s'\"]|$)|"
            r"(^|[\s'\"])\.(?:[\s'\"]|$)",
            lower_command,
        )
    )


def _buffered_agentic_destructive_command(
    buffered_events: list[tuple],
    messages: list,
) -> bool:
    if not _agentic_created_artifact_paths(messages):
        return False
    for name, arguments in _assembled_tool_call_functions(buffered_events):
        if name not in {"bash", "shell", "exec", "run_command"}:
            continue
        decoded = _decoded_tool_arguments(arguments)
        command = ""
        if isinstance(decoded, dict):
            candidate = decoded.get("command") or decoded.get("cmd")
            if isinstance(candidate, str):
                command = candidate
        elif isinstance(decoded, str):
            command = decoded
        if command and _command_removes_artifact_tree(command):
            return True
    return False


def _partial_tool_path_signature(name: str | None, arguments: str) -> tuple[str, str] | None:
    if not name:
        return None
    match = _PARTIAL_TOOL_PATH_RE.search(arguments)
    if not match:
        return None
    try:
        path = json.loads(f'"{match.group(1)}"')
    except (json.JSONDecodeError, ValueError):
        path = match.group(1)
    if not isinstance(path, str) or not path.strip():
        return None
    return str(name), path.strip()


def _assembled_partial_tool_call_path_signatures(
    buffered_events: list[tuple],
) -> list[tuple[str, str]]:
    calls: dict[int, dict[str, str | None]] = {}
    for event, _ in buffered_events:
        for tool_call in getattr(event, "tool_calls", None) or []:
            index = _tool_call_value(tool_call, "index")
            if not isinstance(index, int):
                index = len(calls)
            assembled = calls.setdefault(index, {"name": None, "arguments": ""})

            function = _tool_call_value(tool_call, "function", {}) or {}
            name = _tool_call_value(function, "name")
            if name:
                assembled["name"] = str(name)

            arguments = _tool_call_value(function, "arguments")
            if arguments is not None:
                assembled["arguments"] = (assembled["arguments"] or "") + str(arguments)

    signatures: list[tuple[str, str]] = []
    for assembled in calls.values():
        signature = _partial_tool_path_signature(
            assembled.get("name"),
            assembled.get("arguments") or "",
        )
        if signature:
            signatures.append(signature)
    return signatures


def _buffered_tool_call_argument_chars(buffered_events: list[tuple]) -> int:
    total = 0
    for event, _ in buffered_events:
        for tool_call in getattr(event, "tool_calls", None) or []:
            function = _tool_call_value(tool_call, "function", {}) or {}
            arguments = _tool_call_value(function, "arguments")
            if arguments is not None:
                total += len(str(arguments))
    return total


def _stream_tool_call_repeats_recent(event, recent_signature) -> bool:
    if not recent_signature:
        return False
    for tool_call in getattr(event, "tool_calls", None) or []:
        if _tool_call_signature(tool_call) == recent_signature:
            return True
    return False


def _stream_tool_call_repeats_recent_path(event, recent_signature) -> bool:
    if not recent_signature:
        return False
    for tool_call in getattr(event, "tool_calls", None) or []:
        if _tool_call_path_signature(tool_call) == recent_signature:
            return True
    return False


def _buffered_tool_call_repeats_recent(
    buffered_events: list[tuple],
    recent_signature,
) -> bool:
    return bool(
        recent_signature
        and recent_signature in _assembled_tool_call_signatures(buffered_events)
    )


def _buffered_tool_call_repeats_recent_path(
    buffered_events: list[tuple],
    recent_signature,
) -> bool:
    return bool(
        recent_signature
        and (
            recent_signature in _assembled_tool_call_path_signatures(buffered_events)
            or recent_signature
            in _assembled_partial_tool_call_path_signatures(buffered_events)
        )
    )


async def _iterate_with_idle_timeout(async_iter, timeout: float):
    iterator = async_iter.__aiter__()
    while True:
        task = asyncio.create_task(iterator.__anext__())
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if task not in done:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (
                    asyncio.CancelledError,
                    RuntimeError,
                    StopAsyncIteration,
                    TimeoutError,
                ):
                    pass
                if hasattr(iterator, "aclose"):
                    try:
                        await asyncio.wait_for(iterator.aclose(), timeout=1.0)
                    except (asyncio.CancelledError, RuntimeError, TimeoutError):
                        pass
                raise TimeoutError
            yield task.result()
        except StopAsyncIteration:
            return
        except TimeoutError:
            if hasattr(iterator, "aclose"):
                try:
                    await asyncio.wait_for(iterator.aclose(), timeout=1.0)
                except (asyncio.CancelledError, RuntimeError, TimeoutError):
                    pass
            raise


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    _validate_model_name(request.model)
    engine = get_engine(request.model)

    # Validate messages is non-empty
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty",
        )

    # Validate message roles
    _valid_roles = {"system", "user", "assistant", "tool", "developer"}
    for msg in request.messages:
        if msg.role not in _valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be one of: {', '.join(sorted(_valid_roles))}",
            )

    # Validate n parameter (only n=1 supported)
    if request.n is not None and request.n > 1:
        raise HTTPException(
            status_code=400,
            detail="n > 1 is not supported. Rapid-MLX generates one completion per request.",
        )

    # Validate max_tokens (must be positive)
    if request.max_tokens is not None and request.max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at least 1",
        )

    # Validate temperature range (OpenAI spec: 0-2)
    if request.temperature is not None and (
        request.temperature < 0 or request.temperature > 2
    ):
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    # Validate top_logprobs range (OpenAI spec: 0-20)
    if request.top_logprobs is not None and (
        request.top_logprobs < 0 or request.top_logprobs > 20
    ):
        raise HTTPException(
            status_code=400,
            detail="top_logprobs must be between 0 and 20",
        )

    # --- Detailed request logging ---
    n_msgs = len(request.messages)
    msg_roles = [m.role for m in request.messages]
    total_chars = 0
    last_user_preview = ""
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    has_tools = bool(request.tools)
    n_tools = len(request.tools) if request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/chat/completions stream={request.stream} "
        f"model={request.model!r} max_tokens={request.max_tokens} "
        f"temp={request.temperature} msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.debug(f"[REQUEST] last user message preview: {last_user_preview!r}")

    cfg = get_config()
    if cfg.agentic_guard:
        for idx, message in enumerate(request.messages):
            role = getattr(message, "role", None)
            if role != "tool":
                continue
            content = getattr(message, "content", "")
            tool_call_id = getattr(message, "tool_call_id", None)
            logger.info(
                "[AGENTIC-TOOL-RESULT] index=%d tool_call_id=%r content=%s",
                idx,
                tool_call_id,
                content if isinstance(content, str) else str(content),
            )

    # Save original messages (clean dicts) for cloud routing BEFORE
    # local mutations (extract_multimodal_content, developer→system, suffix injection).
    if cfg.cloud_router:
        _cloud_original_messages = [
            (
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump")
                else {k: v for k, v in dict(msg).items() if v is not None}
            )
            for msg in request.messages
        ]
    else:
        _cloud_original_messages = None

    # For MLLM models, keep original messages with embedded images
    if engine.is_mllm:
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos = [], []
        logger.debug(f"MLLM: Processing {len(messages)} messages")
    else:
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

    has_media = bool(images or videos)
    if engine.is_mllm and not has_media:
        for msg in request.messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else (item.get("type", "") if isinstance(item, dict) else "")
                    )
                    if item_type in ("image_url", "image", "video", "video_url"):
                        has_media = True
                        break
            if has_media:
                break

    # Normalize "developer" role to "system"
    for i, m in enumerate(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "developer":
            if isinstance(m, dict):
                messages[i]["role"] = "system"
            else:
                m.role = "system"

    agentic_terminal_ready = bool(
        cfg.agentic_guard and request.tools and _agentic_task_terminal_ready(messages)
    )
    if agentic_terminal_ready:
        logger.info(
            "[agentic-guard] disabling tools after completed task and successful "
            "validation so the agent can finalize"
        )
        request.tools = None
        request.tool_choice = None
        messages = list(messages) + [{"role": "user", "content": _AGENTIC_FINAL_USER_PROMPT}]

    # Auto-inject system prompt suffix for tool use and/or reasoning control
    _inject_suffix = None
    if request.tools and cfg.tool_call_parser:
        _inject_suffix = _TOOL_USE_SYSTEM_SUFFIX
    elif cfg.reasoning_parser_name == "minimax":
        _inject_suffix = (
            "\n\nDo NOT think out loud or show your reasoning process. "
            "Give direct answers only — no preamble like 'The user asks...' or "
            "'We should respond...' or 'Let me think...'. Be concise."
        )
    agentic_verification_required = bool(
        cfg.agentic_guard
        and request.tools
        and _agentic_completion_needs_verification(messages)
    )
    if agentic_verification_required:
        _inject_suffix = (_inject_suffix or "") + _AGENTIC_TOOL_USE_SYSTEM_SUFFIX

    if _inject_suffix:
        has_system = any(
            (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
            == "system"
            for m in messages
        )
        if has_system:
            for i, m in enumerate(messages):
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                if role == "system":
                    if isinstance(m, dict):
                        messages[i] = {**m, "content": m["content"] + _inject_suffix}
                    else:
                        messages[i]["content"] = m["content"] + _inject_suffix
                    break
        else:
            system_msg = {"role": "system", "content": _inject_suffix.strip()}
            messages = [system_msg] + list(messages)

    messages, added_tool_continuation_prompt = _append_tool_continuation_prompt(
        list(messages),
        bool(request.tools),
    )

    # Auto-pin system prompt prefix cache blocks
    if cfg.pin_system_prompt:
        _maybe_pin_system_prompt(messages)

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        try:
            json_instruction = build_json_system_prompt(response_format)
        except Exception as e:
            logger.warning(f"Failed to build JSON system prompt: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid response_format schema: {e}",
            )
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(request.max_tokens, request.enable_thinking),
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": request.stop,
    }

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # Add tools if provided
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    # Pass through enable_thinking if explicitly set by the client
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking
    elif (
        cfg.no_thinking
        or (
            added_tool_continuation_prompt
            and cfg.tool_call_parser == "qwen3_coder_xml"
        )
        or agentic_verification_required
    ):
        chat_kwargs["enable_thinking"] = False

    if (
        cfg.structured_cot or (cfg.structured_cot_tools and request.tools)
    ) and chat_kwargs.get("enable_thinking") is not False:
        chat_kwargs["structured_cot"] = True
        chat_kwargs["structured_cot_token_budget"] = cfg.structured_cot_token_budget

    # Cloud routing: offload large-context requests to cloud LLM
    if cfg.cloud_router and not engine.is_mllm and hasattr(engine, "build_prompt"):
        try:
            prompt = engine.build_prompt(messages, tools=request.tools)
            total_tokens, new_tokens = engine.model.estimate_new_tokens(prompt)
            if cfg.cloud_router.should_route_to_cloud(new_tokens):
                logger.info(
                    f"[CLOUD ROUTE] {new_tokens} new tokens (total {total_tokens}) "
                    f"> threshold {cfg.cloud_router.threshold}, "
                    f"routing to {cfg.cloud_router.cloud_model}"
                )
                cloud_messages = _cloud_original_messages
                cloud_kwargs = {
                    "temperature": chat_kwargs.get("temperature"),
                    "max_tokens": chat_kwargs.get("max_tokens"),
                    "top_p": chat_kwargs.get("top_p"),
                }
                if request.stop:
                    cloud_kwargs["stop"] = request.stop
                if request.tool_choice is not None:
                    cloud_kwargs["tool_choice"] = request.tool_choice
                if request.response_format:
                    rf = request.response_format
                    cloud_kwargs["response_format"] = (
                        rf.model_dump() if hasattr(rf, "model_dump") else rf
                    )
                if request.tools:
                    cloud_kwargs["tools"] = [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in request.tools
                    ]
                if request.stream:
                    return StreamingResponse(
                        _disconnect_guard(
                            cfg.cloud_router.stream_completion(
                                cloud_messages,
                                model_name=cfg.model_name or "cloud",
                                **cloud_kwargs,
                            ),
                            raw_request,
                            timeout=request.timeout or cfg.default_timeout,
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    result = await _wait_with_disconnect(
                        cfg.cloud_router.completion(cloud_messages, **cloud_kwargs),
                        raw_request,
                        timeout=request.timeout or cfg.default_timeout,
                    )
                    if result is None:
                        return Response(status_code=499, content="Client disconnected")
                    return Response(
                        content=json.dumps(result),
                        media_type="application/json",
                    )
            else:
                logger.info(
                    f"[LOCAL] {new_tokens} new tokens (total {total_tokens}) "
                    f"<= threshold {cfg.cloud_router.threshold}, using local inference"
                )
        except Exception as e:
            logger.warning(
                f"[CLOUD ROUTE] Error during routing check: {e}, falling back to local"
            )

    if request.stream:
        # Validate chat template eagerly so template errors return 400
        if hasattr(engine, "build_prompt") and not engine.is_mllm:
            try:
                engine.build_prompt(
                    messages,
                    tools=chat_kwargs.get("tools"),
                    enable_thinking=chat_kwargs.get("enable_thinking"),
                )
            except Exception as e:
                err_msg = str(e)
                err_type = type(e).__name__
                if (
                    "TemplateError" in err_type
                    or "template" in err_msg.lower()
                    or ("user" in err_msg.lower() and "found" in err_msg.lower())
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Chat template error: {err_msg}",
                    )
                raise
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(
                    engine,
                    messages,
                    request,
                    tool_continuation_retry=added_tool_continuation_prompt,
                    **chat_kwargs,
                ),
                raw_request,
                timeout=request.timeout or cfg.default_timeout,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or cfg.default_timeout

    # Disable GC during generation to avoid latency spikes
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    # Determine if we need per-token logprobs
    want_logprobs = request.logprobs and request.top_logprobs
    top_k_logprobs = request.top_logprobs or 0
    token_logprobs_list: list[TokenLogProb] = []

    # Check if we should use guided generation for JSON schema
    use_guided = False
    json_schema = None
    if response_format and not request.tools:
        json_schema = extract_json_schema_for_guided(response_format)
        if json_schema and hasattr(engine, "supports_guided_generation"):
            use_guided = engine.supports_guided_generation
            if use_guided:
                logger.info("Using guided generation for JSON schema enforcement")

    try:
        if want_logprobs and not use_guided:
            output = None
            async for chunk in engine.stream_chat(messages=messages, **chat_kwargs):
                output = chunk
                if chunk.logprobs is not None and chunk.new_text:
                    token_id = chunk.tokens[-1] if chunk.tokens else 0
                    token_logprobs_list.append(
                        _extract_token_logprob(
                            chunk.logprobs, token_id, engine.tokenizer, top_k_logprobs
                        )
                    )
            if output is None:
                return Response(status_code=499)
        elif use_guided and json_schema:
            try:
                output = await _wait_with_disconnect(
                    engine.generate_with_schema(
                        messages=messages,
                        json_schema=json_schema,
                        **chat_kwargs,
                    ),
                    raw_request,
                    timeout=timeout,
                )
            except Exception as guided_err:
                logger.warning(
                    f"Guided generation failed, falling back to standard: {guided_err}"
                )
                logger.debug(f"Problematic schema: {json_schema}")
                output = await _wait_with_disconnect(
                    engine.chat(messages=messages, **chat_kwargs),
                    raw_request,
                    timeout=timeout,
                )
        else:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                raw_request,
                timeout=timeout,
            )
    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e)
        err_type = type(e).__name__
        if (
            "TemplateError" in err_type
            or "template" in err_msg.lower()
            or ("user" in err_msg.lower() and "found" in err_msg.lower())
        ):
            raise HTTPException(
                status_code=400, detail=f"Chat template error: {err_msg}"
            )
        raise
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()

    if output is None:
        return Response(status_code=499)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls from output using configured parser. In non-streaming
    # local-agent mode, retry premature text-only answers for tool workflows.
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, request)
    if agentic_terminal_ready and tool_calls:
        logger.info(
            "[agentic-guard] ignoring parsed tool calls after terminal-ready "
            "finalization"
        )
        cleaned_text = output.text
        tool_calls = []
    if (
        agentic_verification_required
        and not _agentic_verification_present(messages)
    ):
        retry_messages = list(messages)

        for retry_attempt in range(6):
            if tool_calls:
                break
            logger.info(
                "[agentic-guard] retrying text-only non-stream response "
                "without verification evidence (attempt %d/6)",
                retry_attempt + 1,
            )
            retry_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        _AGENTIC_REPAIR_USER_PROMPT
                        if _agentic_failed_validation_present(messages)
                        else "You are not finished. Use the next required tool now."
                    ),
                }
            ]
            retry_kwargs = {
                **chat_kwargs,
                "enable_thinking": False,
                "structured_cot": False,
                "max_tokens": min(int(chat_kwargs.get("max_tokens") or 512), 512),
            }
            retry_output = await _wait_with_disconnect(
                engine.chat(messages=retry_messages, **retry_kwargs),
                raw_request,
                timeout=timeout,
            )
            if retry_output is None:
                return Response(status_code=499)
            output = retry_output
            cleaned_text, tool_calls = _parse_tool_calls_with_parser(
                output.text, request
            )

    # Validate tool call parameter values against schemas. If the parser extracted
    # an incomplete or malformed tool call, retry before returning it to clients.
    tool_param_errors = (
        _validate_tool_call_params(tool_calls, request.tools)
        if tool_calls and request.tools
        else []
    )
    if tool_param_errors:
        retry_messages = list(messages)
        for retry_attempt in range(2):
            logger.info(
                "[tool-call] retrying invalid tool-call arguments "
                "(attempt %d/2): %s",
                retry_attempt + 1,
                "; ".join(tool_param_errors[:3]),
            )
            retry_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        _TOOL_CALL_JSON_RETRY_PROMPT
                        + "\nValidation errors:\n- "
                        + "\n- ".join(tool_param_errors[:8])
                    ),
                }
            ]
            retry_kwargs = {
                **chat_kwargs,
                "enable_thinking": False,
                "structured_cot": False,
            }
            retry_output = await _wait_with_disconnect(
                engine.chat(messages=retry_messages, **retry_kwargs),
                raw_request,
                timeout=timeout,
            )
            if retry_output is None:
                return Response(status_code=499)
            output = retry_output
            cleaned_text, tool_calls = _parse_tool_calls_with_parser(
                output.text, request
            )
            tool_param_errors = (
                _validate_tool_call_params(tool_calls, request.tools)
                if tool_calls and request.tools
                else []
            )
            if not tool_param_errors:
                break

    # Extract reasoning content FIRST.
    # Note: extract_reasoning() is stateless (pure regex on full text),
    # so using the singleton is safe here unlike the streaming variant.
    reasoning_text = None
    if cfg.reasoning_parser:
        text_to_parse = cleaned_text or output.text
        reasoning_text, cleaned_text = cfg.reasoning_parser.extract_reasoning(
            text_to_parse
        )

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        try:
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, response_format
            )
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
            if not is_valid:
                logger.warning(f"JSON validation failed: {error}")
        except Exception as e:
            logger.warning(f"JSON output parsing failed: {e}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    # Clean and strip thinking tags from content
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)
        if response_format and final_content:
            final_content = extract_json_from_response(final_content)

    # Build logprobs for response if requested
    choice_logprobs = None
    if want_logprobs and token_logprobs_list:
        choice_logprobs = ChoiceLogProbs(content=token_logprobs_list)

    chat_response = ChatCompletionResponse(
        model=_resolve_model_name(request.model),
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    reasoning_content=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
                logprobs=choice_logprobs,
            )
        ],
        usage=_build_usage(output, reasoning_text),
    )
    return Response(
        content=chat_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


async def stream_chat_completion(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    tool_continuation_retry: bool = False,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response.

    Uses StreamingPostProcessor for reasoning/tool/sanitization pipeline.
    SSE formatting stays inline for performance (fast path bypasses Pydantic).
    """
    from ..service.postprocessor import StreamingPostProcessor

    cfg = get_config()
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Check if we should include usage in the final chunk
        include_usage = request.stream_options and request.stream_options.include_usage

        # Logprobs configuration
        want_logprobs = request.logprobs and request.top_logprobs
        top_k_logprobs = request.top_logprobs or 0

        def _build_chunk_logprobs(output: GenerationOutput) -> ChoiceLogProbs | None:
            """Build ChoiceLogProbs for a streaming chunk if logprobs requested."""
            if not want_logprobs or output.logprobs is None:
                return None
            token_id = output.tokens[-1] if output.tokens else 0
            token_lp = _extract_token_logprob(
                output.logprobs, token_id, engine.tokenizer, top_k_logprobs
            )
            return ChoiceLogProbs(content=[token_lp])

        # Pre-compute SSE template parts that don't change per-token.
        _sse_created = int(time.time())
        _model_escaped = json.dumps(_resolve_model_name(request.model))
        _sse_prefix = (
            f'data: {{"id":"{response_id}","object":"chat.completion.chunk",'
            f'"created":{_sse_created},"model":{_model_escaped},'
            f'"choices":[{{"index":0,"delta":{{'
        )
        _sse_suffix = "}}]}\n\n"

        def _fast_sse_chunk(text: str, field: str = "content") -> str:
            """Build SSE chunk JSON directly, bypassing Pydantic serialization."""
            escaped = json.dumps(text)
            return f'{_sse_prefix}"{field}":{escaped}{_sse_suffix}'

        # First chunk with role
        _first_sse = f'{_sse_prefix}"role":"assistant"{_sse_suffix}'
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"[SSE-ROLE] {_first_sse.strip()[:200]}")
        yield _first_sse

        # Track token counts for usage reporting
        prompt_tokens = 0
        completion_tokens = 0

        def _format_stream_event(event, output: GenerationOutput) -> list[str]:
            if event.type == "content":
                if not want_logprobs:
                    _sse = _fast_sse_chunk(event.content, "content")
                    return [_sse] if _sse else []
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=event.content,
                            ),
                            logprobs=_build_chunk_logprobs(output),
                        )
                    ],
                )
                return [f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"]

            if event.type == "reasoning":
                return [_fast_sse_chunk(event.reasoning, "reasoning_content")]

            if event.type == "tool_call":
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                tool_calls=event.tool_calls,
                            ),
                            finish_reason=event.finish_reason,
                        )
                    ],
                    usage=get_usage(output) if output.finished else None,
                )
                _tc_sse = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                logger.info(f"[SSE-TC] {_tc_sse.strip()[:300]}")
                return [_tc_sse]

            if event.type == "finish":
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=event.content,
                                reasoning_content=event.reasoning,
                            ),
                            finish_reason=event.finish_reason,
                            logprobs=_build_chunk_logprobs(output),
                        )
                    ],
                    usage=get_usage(output) if output.finished else None,
                )
                return [f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"]

            return []

        def _format_forced_tool_call(tool_calls: list[ToolCall], output) -> str:
            stream_tool_calls = []
            for index, tool_call in enumerate(tool_calls):
                tool_call_delta = tool_call.model_dump(exclude_none=True)
                tool_call_delta["index"] = index
                stream_tool_calls.append(tool_call_delta)
            chunk = ChatCompletionChunk(
                id=response_id,
                model=_resolve_model_name(request.model),
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            tool_calls=stream_tool_calls,
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=get_usage(output) if output else None,
            )
            return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        agentic_stream_guard = (
            cfg.agentic_guard
            and bool(request.tools)
            and _agentic_completion_needs_verification(messages)
            and (
                not _agentic_verification_present(messages)
                or bool(_agentic_requested_artifacts_missing(messages))
            )
        )
        agentic_missing_requested_artifacts = (
            agentic_stream_guard
            and _agentic_verification_present(messages)
            and bool(_agentic_requested_artifacts_missing(messages))
        )
        agentic_repair_mode = (
            agentic_stream_guard and _agentic_failed_validation_present(messages)
        )
        agentic_missing_requested_artifacts_without_validation = (
            agentic_stream_guard
            and not _agentic_verification_present(messages)
            and bool(_agentic_requested_artifacts_missing(messages))
            and _agentic_tool_result_count(messages)
            >= _AGENTIC_MAX_TOOL_RESULTS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
            and not _last_tool_result_is_agentic_diagnostic(messages)
        )
        agentic_tool_results_since_failure = (
            _agentic_tool_result_count_since_latest_failure(messages)
        )
        agentic_same_path_tools_since_failure = (
            _agentic_max_same_path_tools_since_latest_failure(messages)
        )
        agentic_same_command_tools_since_failure = (
            _agentic_max_same_command_tools_since_latest_failure(messages)
        )
        agentic_same_path_tools_without_validation = (
            _agentic_max_same_path_tools_without_validation(messages)
        )
        agentic_repeated_path_repair_mode = (
            agentic_stream_guard
            and max(
                agentic_same_path_tools_since_failure,
                agentic_same_command_tools_since_failure,
            )
            >= _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
            and not _last_tool_result_is_agentic_diagnostic(messages)
        )
        agentic_missing_dependency = (
            _last_tool_result_missing_dependency(messages)
            if agentic_stream_guard and "bash" in _tool_names(request.tools)
            else None
        )
        agentic_dependency_install_needed = agentic_missing_dependency is not None
        if agentic_dependency_install_needed:
            logger.info("[agentic-guard] forcing dependency install after tool result")
            yield _format_forced_tool_call(
                [_agentic_dependency_install_tool_call(agentic_missing_dependency)],
                None,
            )
            if include_usage:
                usage_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"
            return
        agentic_diagnostic_needed = (
            cfg.agentic_guard
            and bool(request.tools)
            and not _agentic_verification_present(messages)
            and not _last_tool_result_is_agentic_diagnostic(messages)
            and "bash" in _tool_names(request.tools)
            and (
                (
                    not _agentic_failed_validation_present(messages)
                    and _agentic_tool_result_count(messages)
                    >= _AGENTIC_MAX_TOOL_RESULTS_BEFORE_DIAGNOSTIC
                )
                or (
                    agentic_tool_results_since_failure
                    >= _AGENTIC_MAX_TOOL_RESULTS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
                )
                or (
                    max(
                        agentic_same_path_tools_since_failure,
                        agentic_same_command_tools_since_failure,
                    )
                    >= _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
                )
                or (
                    not _agentic_failed_validation_present(messages)
                    and agentic_same_path_tools_without_validation
                    >= _AGENTIC_MAX_SAME_PATH_TOOLS_AFTER_FAILURE_BEFORE_DIAGNOSTIC
                )
            )
        )
        if (
            agentic_diagnostic_needed
            and not agentic_missing_requested_artifacts_without_validation
        ):
            logger.info(
                "[agentic-guard] forcing diagnostic after %d tool results without "
                "validation evidence (%d since latest failure, %d same-path, "
                "%d same-command)",
                _agentic_tool_result_count(messages),
                agentic_tool_results_since_failure,
                agentic_same_path_tools_since_failure,
                agentic_same_command_tools_since_failure,
            )
            yield _format_forced_tool_call([_agentic_diagnostic_tool_call()], None)
            if include_usage:
                usage_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"
            return
        retry_attempts = 0
        active_messages = (
            list(messages)
            + [
                {
                    "role": "user",
                    "content": (
                        _AGENTIC_REPEATED_PATH_REPAIR_PROMPT
                        if agentic_repeated_path_repair_mode
                        else _agentic_missing_artifact_prompt(messages)
                        if (
                            agentic_missing_requested_artifacts
                            or agentic_missing_requested_artifacts_without_validation
                        )
                        else _AGENTIC_REPAIR_USER_PROMPT
                    ),
                }
            ]
            if (
                agentic_repair_mode
                or agentic_repeated_path_repair_mode
                or agentic_missing_requested_artifacts
                or agentic_missing_requested_artifacts_without_validation
            )
            else messages
        )
        active_kwargs = (
            {
                **kwargs,
                "enable_thinking": False,
                "structured_cot": False,
                "max_tokens": min(
                    int(kwargs.get("max_tokens") or _AGENTIC_RETRY_MAX_TOKENS),
                    _AGENTIC_RETRY_MAX_TOKENS,
                ),
            }
            if (
                agentic_repair_mode
                or agentic_repeated_path_repair_mode
                or agentic_missing_requested_artifacts
                or agentic_missing_requested_artifacts_without_validation
            )
            else kwargs
        )
        tools_disabled_for_retry = False
        repeated_tool_retry_active = False
        recent_tool_call_signature = _last_assistant_tool_call_signature(messages)
        recent_tool_call_path_signature = _last_assistant_tool_call_path_signature(
            messages
        )
        latest_result_mentions_recent_path = (
            bool(recent_tool_call_path_signature)
            and _latest_tool_result_mentions_path(
                messages,
                recent_tool_call_path_signature[1],
            )
        )
        last_message = active_messages[-1] if active_messages else {}
        last_content = (
            last_message.get("content")
            if isinstance(last_message, dict)
            else getattr(last_message, "content", None)
        )
        repeated_tool_continuation = (
            tool_continuation_retry
            and last_content == _TOOL_CONTINUATION_REPEATED_TOOL_PROMPT
        )
        agentic_task_complete = (
            cfg.agentic_guard
            and bool(request.tools)
            and _agentic_completion_needs_verification(messages)
            and _agentic_verification_present(messages)
            and not _agentic_requested_artifacts_missing(messages)
        )
        structured_agentic_continuation = (
            (
                tool_continuation_retry
                or agentic_stream_guard
            )
            and (cfg.structured_cot_tools or agentic_stream_guard)
            and bool(request.tools)
            and not repeated_tool_continuation
            and not agentic_task_complete
        )
        tool_retries_enabled = (
            bool(request.tools)
            and (
                _tool_choice_requires_tool_call(request.tool_choice)
                or structured_agentic_continuation
            )
            and not repeated_tool_continuation
        )
        max_tool_continuation_retries = (
            2
            if agentic_stream_guard
            else 6
            if structured_agentic_continuation
            else (2 if tool_retries_enabled else 0)
        )
        max_repeated_tool_retries = (
            2 if tool_continuation_retry or agentic_missing_requested_artifacts else 0
        )
        retry_prompt = (
            _TOOL_CONTINUATION_RETRY_PROMPT
            if tool_continuation_retry
            else _TOOL_CALL_REQUIRED_RETRY_PROMPT
        )
        if agentic_stream_guard:
            text_before_tool_call_max_chars = _AGENTIC_TEXT_BEFORE_TOOL_CALL_MAX_CHARS
        else:
            text_before_tool_call_max_chars = _TOOL_TEXT_BEFORE_TOOL_CALL_MAX_CHARS
        configured_timeout = request.timeout or cfg.default_timeout
        stream_idle_timeout = min(configured_timeout, _STREAM_IDLE_TIMEOUT_SECONDS)

        while True:
            agentic_task_active = (
                cfg.agentic_guard
                and bool(request.tools)
                and _agentic_completion_needs_verification(messages)
                and not agentic_task_complete
            )
            active_tool_retries_enabled = (
                tool_retries_enabled
                and not tools_disabled_for_retry
                and not repeated_tool_retry_active
            )
            repeat_tool_detection_enabled = (
                tool_continuation_retry
                and not cfg.agentic_guard
                and not agentic_repair_mode
                and not _last_tool_result_indicates_failure(messages)
                and bool(
                    recent_tool_call_signature or recent_tool_call_path_signature
                )
                and not tools_disabled_for_retry
            )
            agentic_repeated_path_detection_enabled = (
                agentic_repeated_path_repair_mode
                and bool(recent_tool_call_path_signature)
                and not tools_disabled_for_retry
            )
            agentic_repeated_inspection_detection_enabled = (
                agentic_missing_requested_artifacts
                and bool(recent_tool_call_signature)
                and not tools_disabled_for_retry
            )
            tool_call_buffering_enabled = (
                active_tool_retries_enabled
                or repeat_tool_detection_enabled
                or agentic_repeated_path_detection_enabled
                or agentic_repeated_inspection_detection_enabled
            )
            tool_call_buffering_active = tool_call_buffering_enabled
            # Initialize post-processor
            processor = StreamingPostProcessor(
                cfg,
                tools_requested=bool(request.tools) and not tools_disabled_for_retry,
                json_mode=bool(
                    request.response_format
                    and getattr(request.response_format, "type", "text") != "text"
                ),
                request_dict=request.model_dump(),
            )
            processor.set_thinking_model(request.model)
            processor.reset()

            buffered_events: list[tuple] = []
            buffered_tool_call_events: list[tuple] = []
            deferred_finish: tuple | None = None
            emitted_tool_call = False
            last_output = None
            retry_reason = None
            last_progress_time = time.perf_counter()
            last_progress_tokens = 0

            # Stream content — PostProcessor handles reasoning/tool/sanitize
            try:
                async for output in _iterate_with_idle_timeout(
                    engine.stream_chat(messages=active_messages, **active_kwargs),
                    stream_idle_timeout,
                ):
                    last_output = output
                    if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                        prompt_tokens = output.prompt_tokens
                    if hasattr(output, "completion_tokens") and output.completion_tokens:
                        completion_tokens = output.completion_tokens
                    output_tokens = int(getattr(output, "completion_tokens", 0) or 0)
                    output_text = getattr(output, "new_text", None) or ""
                    if (
                        output_text
                        or output_tokens > last_progress_tokens
                        or getattr(output, "finished", False)
                    ):
                        last_progress_time = time.perf_counter()
                        last_progress_tokens = max(last_progress_tokens, output_tokens)
                    elif time.perf_counter() - last_progress_time > stream_idle_timeout:
                        retry_reason = "stream no-progress timeout"
                        logger.warning(
                            "[tool-continuation] stream yielded no text/token "
                            "progress for %.1fs; aborting current stream",
                            stream_idle_timeout,
                        )
                        break

                    retry_window = (
                        active_tool_retries_enabled
                        and not emitted_tool_call
                    )

                    for event in processor.process_chunk(output):
                        if retry_window and event.type in ("content", "reasoning"):
                            buffered_events.append((event, output))
                            buffered_text = _buffered_stream_text(buffered_events)
                            if (
                                len(buffered_text)
                                > text_before_tool_call_max_chars
                            ):
                                retry_reason = "too much text before tool call"
                                logger.info(
                                    "[tool-continuation] buffered text exceeded %d "
                                    "chars before tool call; aborting current stream",
                                    text_before_tool_call_max_chars,
                                )
                                break
                            if (
                                not agentic_stream_guard
                                and _is_repetitive_tool_text(buffered_text)
                            ):
                                retry_reason = "repetitive text before tool call"
                                logger.info(
                                    "[tool-continuation] detected repetitive text "
                                    "before tool call; aborting current stream"
                                )
                                break
                            continue

                        if (
                            retry_window
                            and event.type == "finish"
                            and event.finish_reason == "stop"
                        ):
                            deferred_finish = (event, output)
                            continue

                        if (
                            tool_call_buffering_active
                            and event.type == "finish"
                            and buffered_tool_call_events
                        ):
                            deferred_finish = (event, output)
                            continue

                        if tool_call_buffering_active and event.type == "tool_call":
                            buffered_tool_call_events.append((event, output))
                            if (
                                (
                                    repeat_tool_detection_enabled
                                    or agentic_repeated_path_detection_enabled
                                )
                                and (
                                    agentic_repeated_path_detection_enabled
                                    or not latest_result_mentions_recent_path
                                )
                                and _buffered_tool_call_repeats_recent_path(
                                    buffered_tool_call_events,
                                    recent_tool_call_path_signature,
                                )
                            ):
                                retry_reason = "repeated tool call"
                                logger.info(
                                    "[tool-continuation] detected repeated tool path "
                                    "from partial arguments; aborting current stream"
                                )
                                break
                            if (
                                _buffered_tool_call_argument_chars(
                                    buffered_tool_call_events
                                )
                                > _TOOL_CALL_REPEAT_BUFFER_MAX_ARGUMENT_CHARS
                            ):
                                logger.info(
                                    "[tool-continuation] tool-call buffer exceeded "
                                    "%d argument chars; streaming current tool call",
                                    _TOOL_CALL_REPEAT_BUFFER_MAX_ARGUMENT_CHARS,
                                )
                                for buffered_event, buffered_output in buffered_events:
                                    for _sse in _format_stream_event(
                                        buffered_event, buffered_output
                                    ):
                                        yield _sse
                                buffered_events.clear()

                                emitted_tool_call = True
                                for tool_event, tool_output in buffered_tool_call_events:
                                    for _sse in _format_stream_event(
                                        tool_event, tool_output
                                    ):
                                        yield _sse
                                buffered_tool_call_events.clear()
                                deferred_finish = None
                                tool_call_buffering_active = False
                            continue

                        if event.type == "tool_call":
                            emitted_tool_call = True
                            for buffered_event, buffered_output in buffered_events:
                                for _sse in _format_stream_event(
                                    buffered_event, buffered_output
                                ):
                                    yield _sse
                            buffered_events.clear()

                        for _sse in _format_stream_event(event, output):
                            yield _sse

                    if retry_reason:
                        break
            except TimeoutError:
                retry_reason = "stream idle timeout"
                logger.warning(
                    "[tool-continuation] stream produced no chunk for %.1fs; "
                    "aborting current stream",
                    stream_idle_timeout,
                )

            # Fallback tool call detection
            if not retry_reason:
                for event in processor.finalize():
                    if event.type == "tool_call":
                        if tool_call_buffering_active:
                            buffered_tool_call_events.append((event, last_output))
                            continue

                        emitted_tool_call = True
                        for buffered_event, buffered_output in buffered_events:
                            for _sse in _format_stream_event(
                                buffered_event, buffered_output
                            ):
                                yield _sse
                        buffered_events.clear()

                        tool_chunk = ChatCompletionChunk(
                            id=response_id,
                            model=_resolve_model_name(request.model),
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        tool_calls=event.tool_calls,
                                    ),
                                    finish_reason="tool_calls",
                                )
                            ],
                        )
                        _fb_sse = (
                            f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
                        )
                        logger.info(f"[SSE-FALLBACK-TC] {_fb_sse.strip()[:300]}")
                        yield _fb_sse

            if not retry_reason and buffered_tool_call_events:
                if _buffered_tool_calls_complete(buffered_tool_call_events):
                    if (
                        agentic_stream_guard
                        and _buffered_agentic_destructive_command(
                            buffered_tool_call_events,
                            messages,
                        )
                    ):
                        retry_reason = "destructive shell command"
                        logger.info(
                            "[agentic-guard] blocked destructive shell command "
                            "against existing project artifacts"
                        )
                        buffered_tool_call_events.clear()
                        deferred_finish = None
                    elif (
                        (
                            repeat_tool_detection_enabled
                            or agentic_repeated_inspection_detection_enabled
                        )
                        and _buffered_tool_call_repeats_recent(
                            buffered_tool_call_events,
                            recent_tool_call_signature,
                        )
                    ) or (
                        (
                            repeat_tool_detection_enabled
                            or agentic_repeated_path_detection_enabled
                        )
                        and _buffered_tool_call_repeats_recent_path(
                            buffered_tool_call_events,
                            recent_tool_call_path_signature,
                        )
                    ):
                        retry_reason = "repeated tool call"
                    else:
                        emitted_tool_call = True
                        for buffered_event, buffered_output in buffered_events:
                            for _sse in _format_stream_event(
                                buffered_event, buffered_output
                            ):
                                yield _sse
                        buffered_events.clear()

                        for tool_event, tool_output in buffered_tool_call_events:
                            if tool_output is None:
                                continue
                            for _sse in _format_stream_event(tool_event, tool_output):
                                yield _sse
                        buffered_tool_call_events.clear()

                        if deferred_finish:
                            event, output = deferred_finish
                            for _sse in _format_stream_event(event, output):
                                yield _sse
                            deferred_finish = None
                else:
                    retry_reason = "incomplete tool call JSON"

            if not retry_reason and deferred_finish and not emitted_tool_call:
                retry_reason = "text-only stop"
            elif not retry_reason and buffered_events and not emitted_tool_call:
                retry_reason = "stream exhausted without tool call"

            can_retry_repeated_tool = (
                retry_reason == "repeated tool call"
                and retry_attempts < max_repeated_tool_retries
            )
            can_retry_tool_continuation = (
                retry_reason != "repeated tool call"
                and retry_attempts < max_tool_continuation_retries
            )
            if retry_reason and (
                can_retry_tool_continuation or can_retry_repeated_tool
            ):
                retry_attempts += 1
                retry_limit = (
                    max_repeated_tool_retries
                    if can_retry_repeated_tool
                    else max_tool_continuation_retries
                )
                logger.info(
                    "[tool-continuation] retrying after %s following "
                    "tool result (attempt %d/%d)",
                    retry_reason,
                    retry_attempts,
                    retry_limit,
                )
                selected_retry_prompt = (
                    _TOOL_CALL_JSON_RETRY_PROMPT
                    if retry_reason == "incomplete tool call JSON"
                    else (
                        _AGENTIC_DESTRUCTIVE_COMMAND_REPAIR_PROMPT
                        if agentic_stream_guard
                        and retry_reason == "destructive shell command"
                        else
                        (
                            _AGENTIC_REPEATED_TOOL_PROMPT
                            if agentic_stream_guard
                            else _TOOL_CONTINUATION_REPEATED_TOOL_PROMPT
                        )
                        if retry_reason == "repeated tool call"
                        else (
                            _agentic_missing_artifact_prompt(messages)
                            if agentic_stream_guard
                            and bool(_agentic_requested_artifacts_missing(messages))
                            else _AGENTIC_REPAIR_USER_PROMPT
                            if agentic_stream_guard
                            and _agentic_failed_validation_present(messages)
                            else retry_prompt
                        )
                    )
                )
                active_messages = list(messages) + [
                    {"role": "user", "content": selected_retry_prompt}
                ]
                active_kwargs = {
                    **kwargs,
                    "enable_thinking": False,
                    "structured_cot": False,
                    "max_tokens": min(
                        int(
                            kwargs.get("max_tokens")
                            or (
                                _AGENTIC_NO_TOOL_RETRY_MAX_TOKENS
                                if retry_reason
                                == "stream exhausted without tool call"
                                else _AGENTIC_RETRY_MAX_TOKENS
                            )
                        ),
                        (
                            _AGENTIC_NO_TOOL_RETRY_MAX_TOKENS
                            if retry_reason == "stream exhausted without tool call"
                            else _AGENTIC_RETRY_MAX_TOKENS
                        ),
                    )
                    if agentic_stream_guard
                    else min(int(kwargs.get("max_tokens") or 512), 512),
                }
                tools_disabled_for_retry = False
                repeated_tool_retry_active = (
                    retry_reason == "repeated tool call" and not agentic_stream_guard
                )
                continue

            if (
                retry_reason
                and agentic_stream_guard
                and "bash" in _tool_names(request.tools)
            ):
                if buffered_events or buffered_tool_call_events:
                    logger.warning(
                        "[tool-continuation] suppressing buffered %s after retry "
                        "budget exhausted",
                        retry_reason,
                    )
                    buffered_events.clear()
                    buffered_tool_call_events.clear()
                    deferred_finish = None
                else:
                    logger.warning(
                        "[tool-continuation] forcing diagnostic after %s retry "
                        "budget exhausted with no buffered output",
                        retry_reason,
                    )
                yield _format_forced_tool_call(
                    [_agentic_diagnostic_tool_call()],
                    last_output,
                )
                break

            if retry_reason and (buffered_events or buffered_tool_call_events):
                logger.warning(
                    "[tool-continuation] suppressing buffered %s after retry "
                    "budget exhausted",
                    retry_reason,
                )
                buffered_events.clear()
                buffered_tool_call_events.clear()
                deferred_finish = None
                fallback_text = (
                    "I could not produce a valid tool call after retries."
                )
                _fallback_sse = _fast_sse_chunk(fallback_text, "content")
                if _fallback_sse:
                    yield _fallback_sse
                finish_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(),
                            finish_reason=(
                                last_output.finish_reason if last_output else "stop"
                            )
                            or "stop",
                        )
                    ],
                    usage=get_usage(last_output) if last_output else None,
                )
                yield f"data: {finish_chunk.model_dump_json(exclude_none=True)}\n\n"

            for buffered_event, buffered_output in buffered_events:
                for _sse in _format_stream_event(buffered_event, buffered_output):
                    yield _sse
            if deferred_finish:
                event, output = deferred_finish
                for _sse in _format_stream_event(event, output):
                    yield _sse

            break

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Send final chunk with usage if requested
        if include_usage:
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                model=_resolve_model_name(request.model),
                choices=[],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()
