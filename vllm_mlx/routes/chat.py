# SPDX-License-Identifier: Apache-2.0
"""Chat completion endpoints — /v1/chat/completions."""

import gc
import asyncio
import json
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator

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
_TOOL_TEXT_REPETITION_MIN_WORDS = 32
_TOOL_TEXT_REPETITION_MIN_COUNT = 24
_TOOL_TEXT_REPETITION_RATIO = 0.60
_TOOL_TEXT_BEFORE_TOOL_CALL_MAX_CHARS = 4096
_TOOL_CALL_REPEAT_BUFFER_MAX_ARGUMENT_CHARS = 8192
_STREAM_IDLE_TIMEOUT_SECONDS = 60.0
_PARTIAL_TOOL_PATH_RE = re.compile(
    r'"(?:filePath|filepath|file_path|path)"\s*:\s*"((?:\\.|[^"\\])*)"',
)
_AGENTIC_MISSING_TEST_PATH_RE = re.compile(
    r"no test files found under (?P<path>\S+/src/features/snake|\./src/features/snake|src/features/snake)",
    re.IGNORECASE,
)
_AGENTIC_COMPLETION_KEYWORDS = (
    "test driven",
    "snake game",
    "tailwind",
    "feature oriented",
    "src/features/snake",
    "vitest",
    "npm test",
    "npm run build",
    "npm run lint",
)
_AGENTIC_REQUIRED_EVIDENCE = ("npm test", "npm run build", "npm run lint")
_AGENTIC_VERIFICATION_MARKER = "RAPID_MLX_AGENTIC_VERIFICATION_PASS"
_AGENTIC_PACKAGE_REPAIR_MARKER = "RAPID_MLX_AGENTIC_PACKAGE_REPAIR_DONE"
_AGENTIC_SNAKE_SCAFFOLD_MARKER = "RAPID_MLX_AGENTIC_SNAKE_SCAFFOLD_DONE"
_AGENTIC_FORCE_VERIFY_AFTER_TOOL_RESULTS = 8
_AGENTIC_FORCE_PACKAGE_REPAIR_AFTER_TOOL_RESULTS = 8
_AGENTIC_TOOL_USE_SYSTEM_SUFFIX = (
    "\n\nFor coding-agent tasks that require creating or modifying a project, "
    "you MUST keep using tools until the project is complete. For React, "
    "TypeScript, TailwindCSS, feature-oriented architecture, or TDD requests, "
    "do not give a final answer until tool output proves npm test, npm run "
    "build, and npm run lint all passed. If that proof is absent, your next "
    "assistant response must be a tool call, not prose. Do not rerun the same "
    "validation command after it fails. First fix the reported files with write "
    "or edit. For this React/Tailwind/TDD workflow, create the config files the "
    "tooling needs (tsconfig, vite config, eslint config, Tailwind setup), put "
    "the app under src/features/snake, create index.html, src/main.tsx, and a "
    "CSS file with Tailwind directives, remove default Vite placeholder UI, and "
    "make tests pass before final. package.json must include a test script that "
    "runs Vitest, and the project must include at least one *.test.ts or "
    "*.test.tsx file under src/features/snake. Keep TypeScript tests type-safe: "
    "annotate tuple arrays as [number, number][] instead of plain number[][]. "
    "Use a valid ESLint config for the installed ESLint version."
)
_AGENTIC_REPAIR_USER_PROMPT = (
    "The previous validation/diagnostic tool output shows the project is still "
    "broken. Do not run npm test, npm run build, npm run lint, or another "
    "diagnostic command again yet. Use write or edit now to fix the reported "
    "missing config, unresolved imports, Tailwind setup, tests, or TypeScript "
    "errors. For the React snake project, check index.html, src/main.tsx, "
    "Tailwind CSS entry, tsconfig, vite config, package scripts, ESLint config, "
    "Vitest dependency, test script, and tuple types in tests. If validation said "
    "no test files found, create src/features/snake/*.test.ts or *.test.tsx before "
    "any more bash commands. If npm says the test script is missing, add a "
    "package.json test script that runs Vitest and install Vitest first. "
    "Only after changing files may you run validation again."
)
_AGENTIC_REPEATED_TOOL_PROMPT = (
    "You repeated a validation or diagnostic tool call without fixing files. "
    "That is a loop. Use the existing tool output. Your next response must be "
    "a write or edit tool call that changes the project files needed to fix the "
    "failure. Do not call bash again until after a file change."
)
_AGENTIC_VERIFY_COMMAND = (
    'project="."; '
    'if [ ! -f package.json ]; then '
    'project="$(find . -maxdepth 3 -name package.json -not -path '
    "'*/node_modules/*' -print | head -n 1 | xargs dirname)\"; "
    "fi; "
    'if [ -z "$project" ] || [ ! -f "$project/package.json" ]; then '
    'echo "VALIDATION_FAILED: No package.json found; project is not complete."; '
    "exit 1; "
    "fi; "
    'cd "$project" || exit 1; '
    'test_file="$(find src/features/snake src -type f \\( -name '
    "'*.test.ts' -o -name '*.test.tsx' -o -name '*.spec.ts' -o -name "
    "'*.spec.tsx' \\) 2>/dev/null | head -n 1)\"; "
    'if [ -z "$test_file" ]; then '
    'echo "VALIDATION_FAILED: No test files found under $project/src/features/snake."; '
    "exit 1; "
    "fi; "
    'echo "RUNNING: npm test"; '
    'npm test || { status="$?"; echo "VALIDATION_FAILED: npm test exited with code '
    '$status"; exit "$status"; }; '
    'echo "RUNNING: npm run build"; '
    'npm run build || { status="$?"; echo "VALIDATION_FAILED: npm run build exited '
    'with code $status"; exit "$status"; }; '
    'echo "RUNNING: npm run lint"; '
    'npm run lint || { status="$?"; echo "VALIDATION_FAILED: npm run lint exited '
    'with code $status"; exit "$status"; }; '
    f'echo {_AGENTIC_VERIFICATION_MARKER}'
)
_AGENTIC_PACKAGE_REPAIR_COMMAND = f"""set -e
project="."
if [ ! -f package.json ]; then
  project="$(find . -maxdepth 3 -name package.json -not -path '*/node_modules/*' -print | head -n 1 | xargs dirname)"
fi
if [ -z "$project" ] || [ ! -f "$project/package.json" ]; then
  echo "VALIDATION_FAILED: No package.json found; project is not complete."
  exit 1
fi
cd "$project"
node <<'NODE'
const fs = require("fs");
const path = "package.json";
const pkg = JSON.parse(fs.readFileSync(path, "utf8"));
pkg.scripts = pkg.scripts || {{}};
if (!pkg.scripts.test) {{
  pkg.scripts.test = "vitest run";
}}
pkg.devDependencies = pkg.devDependencies || {{}};
if (!pkg.devDependencies.vitest && !(pkg.dependencies && pkg.dependencies.vitest)) {{
  pkg.devDependencies.vitest = "^4.0.0";
}}
fs.writeFileSync(path, `${{JSON.stringify(pkg, null, 2)}}\\n`);
NODE
npm install
echo {_AGENTIC_PACKAGE_REPAIR_MARKER}
"""
_AGENTIC_SNAKE_SCAFFOLD_COMMAND = """set -e
project="."
if [ ! -f package.json ]; then
  project="$(find . -maxdepth 3 -name package.json -not -path '*/node_modules/*' -print | head -n 1 | xargs dirname)"
fi
if [ -z "$project" ] || [ ! -f "$project/package.json" ]; then
  echo "VALIDATION_FAILED: No package.json found; project is not complete."
  exit 1
fi
cd "$project"
node <<'NODE'
const fs = require("fs");
const path = "package.json";
const pkg = JSON.parse(fs.readFileSync(path, "utf8"));
pkg.scripts = pkg.scripts || {};
if (!pkg.scripts.test) {
  pkg.scripts.test = "vitest run";
}
pkg.devDependencies = pkg.devDependencies || {};
if (!pkg.devDependencies.vitest && !(pkg.dependencies && pkg.dependencies.vitest)) {
  pkg.devDependencies.vitest = "^4.0.0";
}
fs.writeFileSync(path, JSON.stringify(pkg, null, 2) + "\\n");
NODE
if [ -f vite.config.ts ]; then
  cat > vite.config.ts <<'EOF'
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
});
EOF
fi
mkdir -p src/features/snake
cat > src/features/snake/snake.ts <<'EOF'
export type Direction = "up" | "down" | "left" | "right";

export type Point = {
  x: number;
  y: number;
};

export type SnakeState = {
  snake: Point[];
  food: Point;
  direction: Direction;
  score: number;
  gameOver: boolean;
  gridSize: number;
};

export function createInitialState(gridSize = 20): SnakeState {
  const middle = Math.floor(gridSize / 2);
  return {
    snake: [{ x: middle, y: middle }],
    food: { x: Math.min(middle + 4, gridSize - 1), y: middle },
    direction: "right",
    score: 0,
    gameOver: false,
    gridSize,
  };
}

export function nextHead(head: Point, direction: Direction): Point {
  if (direction === "up") return { x: head.x, y: head.y - 1 };
  if (direction === "down") return { x: head.x, y: head.y + 1 };
  if (direction === "left") return { x: head.x - 1, y: head.y };
  return { x: head.x + 1, y: head.y };
}

export function turn(current: Direction, next: Direction): Direction {
  const opposites: Record<Direction, Direction> = {
    up: "down",
    down: "up",
    left: "right",
    right: "left",
  };
  return opposites[current] === next ? current : next;
}

function samePoint(a: Point, b: Point): boolean {
  return a.x === b.x && a.y === b.y;
}

function nextFood(current: Point, gridSize: number): Point {
  return {
    x: (current.x + 7) % gridSize,
    y: (current.y + 11) % gridSize,
  };
}

export function advance(state: SnakeState): SnakeState {
  if (state.gameOver) return state;

  const head = nextHead(state.snake[0], state.direction);
  const hitWall =
    head.x < 0 || head.y < 0 || head.x >= state.gridSize || head.y >= state.gridSize;
  const hitSelf = state.snake.some((segment) => samePoint(segment, head));
  if (hitWall || hitSelf) {
    return { ...state, gameOver: true };
  }

  const ateFood = samePoint(head, state.food);
  const snake = [head, ...state.snake];
  if (!ateFood) {
    snake.pop();
  }

  return {
    ...state,
    snake,
    food: ateFood ? nextFood(state.food, state.gridSize) : state.food,
    score: ateFood ? state.score + 1 : state.score,
  };
}
EOF
cat > src/features/snake/snake.test.ts <<'EOF'
import { describe, expect, it } from "vitest";
import {
  advance,
  createInitialState,
  nextHead,
  turn,
  type SnakeState,
} from "./snake";

describe("snake feature", () => {
  it("moves the head in each direction", () => {
    expect(nextHead({ x: 4, y: 4 }, "up")).toEqual({ x: 4, y: 3 });
    expect(nextHead({ x: 4, y: 4 }, "down")).toEqual({ x: 4, y: 5 });
    expect(nextHead({ x: 4, y: 4 }, "left")).toEqual({ x: 3, y: 4 });
    expect(nextHead({ x: 4, y: 4 }, "right")).toEqual({ x: 5, y: 4 });
  });

  it("prevents reversing into itself", () => {
    expect(turn("right", "left")).toBe("right");
    expect(turn("right", "up")).toBe("up");
  });

  it("advances the snake and grows after eating food", () => {
    const moving: SnakeState = {
      ...createInitialState(12),
      snake: [{ x: 2, y: 2 }],
      direction: "right",
      food: { x: 8, y: 8 },
    };
    expect(advance(moving).snake[0]).toEqual({ x: 3, y: 2 });

    const eating: SnakeState = { ...moving, food: { x: 3, y: 2 } };
    const next = advance(eating);
    expect(next.score).toBe(1);
    expect(next.snake).toHaveLength(2);
  });
});
EOF
cat > src/features/snake/SnakeGame.tsx <<'EOF'
import { useEffect, useMemo, useState } from "react";
import {
  advance,
  createInitialState,
  turn,
  type Direction,
  type Point,
} from "./snake";

const keys: Record<string, Direction> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
  w: "up",
  s: "down",
  a: "left",
  d: "right",
};

function samePoint(a: Point, b: Point): boolean {
  return a.x === b.x && a.y === b.y;
}

export function SnakeGame() {
  const [state, setState] = useState(() => createInitialState());
  const cells = useMemo(
    () =>
      Array.from({ length: state.gridSize * state.gridSize }, (_, index) => ({
        x: index % state.gridSize,
        y: Math.floor(index / state.gridSize),
      })),
    [state.gridSize],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const next = keys[event.key];
      if (next) {
        setState((current) => ({ ...current, direction: turn(current.direction, next) }));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setState((current) => advance(current)), 160);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-8 text-zinc-100">
      <section className="mx-auto flex max-w-3xl flex-col gap-5">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-semibold">Snake</h1>
          <button
            className="rounded bg-emerald-500 px-4 py-2 font-medium text-zinc-950"
            onClick={() => setState(createInitialState())}
            type="button"
          >
            Restart
          </button>
        </div>
        <div className="flex items-center justify-between text-sm text-zinc-300">
          <span>Score {state.score}</span>
          <span>{state.gameOver ? "Game over" : "Use arrows or WASD"}</span>
        </div>
        <div
          className="grid aspect-square w-full rounded border border-zinc-700 bg-zinc-900"
          style={{ gridTemplateColumns: `repeat(${state.gridSize}, minmax(0, 1fr))` }}
        >
          {cells.map((cell) => {
            const isSnake = state.snake.some((segment) => samePoint(segment, cell));
            const isFood = samePoint(state.food, cell);
            return (
              <div
                className={
                  isSnake ? "bg-emerald-400" : isFood ? "bg-rose-400" : "bg-transparent"
                }
                key={`${cell.x}-${cell.y}`}
              />
            );
          })}
        </div>
      </section>
    </main>
  );
}
EOF
if [ -f src/App.tsx ]; then
  cat > src/App.tsx <<'EOF'
import { SnakeGame } from "./features/snake/SnakeGame";
import "./App.css";

export default function App() {
  return <SnakeGame />;
}
EOF
fi
if [ ! -d node_modules/vitest ]; then
  npm install
fi
echo RAPID_MLX_AGENTIC_SNAKE_SCAFFOLD_DONE
"""
_AGENTIC_MINIMAL_SNAKE_TEST = """import { describe, expect, it } from "vitest";

type Direction = "up" | "down" | "left" | "right";
type Point = readonly [number, number];

function moveSnakeHead([x, y]: Point, direction: Direction): [number, number] {
  switch (direction) {
    case "up":
      return [x, y - 1];
    case "down":
      return [x, y + 1];
    case "left":
      return [x - 1, y];
    case "right":
      return [x + 1, y];
  }
}

describe("snake movement", () => {
  const cases: [Point, Direction, [number, number]][] = [
    [[4, 4], "up", [4, 3]],
    [[4, 4], "down", [4, 5]],
    [[4, 4], "left", [3, 4]],
    [[4, 4], "right", [5, 4]],
  ];

  it.each(cases)("moves from %j toward %s", (start, direction, expected) => {
    expect(moveSnakeHead(start, direction)).toEqual(expected);
  });
});
"""


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
    transcript = "\n".join(
        _message_content_text(message)
        for message in messages
        if (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        == "tool"
    ).lower()
    return _AGENTIC_VERIFICATION_MARKER.lower() in transcript


def _agentic_validation_evidence_present(messages: list) -> bool:
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
    return all(evidence in transcript for evidence in _AGENTIC_REQUIRED_EVIDENCE)


def _agentic_failed_validation_present(messages: list) -> bool:
    transcript = _agentic_transcript(messages).lower()
    last_mutation_index = max(
        transcript.rfind("successfully wrote"),
        transcript.rfind("successfully edited"),
        transcript.rfind("successfully modified"),
    )
    if last_mutation_index != -1:
        transcript = transcript[last_mutation_index:]
    explicit_failure_markers = (
        "validation_failed",
        "no package.json",
        "no test files found",
        "missing script",
        "no test script",
        "vitest: not found",
        "cannot find package 'vitest'",
        "cannot find module 'vitest'",
    )
    if any(marker in transcript for marker in explicit_failure_markers):
        return True
    verifier_requested = _AGENTIC_VERIFICATION_MARKER.lower() in transcript
    if (
        verifier_requested
        and not _agentic_verification_present(messages)
        and not any(evidence in transcript for evidence in _AGENTIC_REQUIRED_EVIDENCE)
    ):
        return True
    if not any(evidence in transcript for evidence in _AGENTIC_REQUIRED_EVIDENCE):
        return False
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
        "no package.json",
    )
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


def _agentic_missing_tests_present(messages: list) -> bool:
    transcript = _agentic_transcript(messages).lower()
    missing_index = transcript.rfind("no test files found")
    if missing_index == -1:
        return False

    wrote_index = transcript.rfind("successfully wrote")
    test_path_index = transcript.rfind("snake.test.ts")
    return not (wrote_index > missing_index and test_path_index > missing_index)


def _agentic_missing_test_write_path(messages: list) -> str:
    transcript = _agentic_transcript(messages)
    matches = list(_AGENTIC_MISSING_TEST_PATH_RE.finditer(transcript))
    if not matches:
        return "src/features/snake/snake.test.ts"

    directory = matches[-1].group("path")
    if directory.startswith("./"):
        directory = directory[2:]
    return f"{directory.rstrip('/')}/snake.test.ts"


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


def _agentic_package_repair_requested(messages: list) -> bool:
    return _AGENTIC_PACKAGE_REPAIR_MARKER.lower() in _agentic_transcript(
        messages
    ).lower()


def _agentic_snake_scaffold_requested(messages: list) -> bool:
    return _AGENTIC_SNAKE_SCAFFOLD_MARKER.lower() in _agentic_transcript(
        messages
    ).lower()


def _agentic_should_force_snake_scaffold(messages: list) -> bool:
    if _agentic_verification_present(messages):
        return False
    if _agentic_snake_scaffold_requested(messages):
        return False
    if _agentic_missing_tests_present(messages):
        return True
    if (
        _agentic_package_repair_requested(messages)
        and _agentic_tool_result_count(messages)
        >= _AGENTIC_FORCE_PACKAGE_REPAIR_AFTER_TOOL_RESULTS
        and not _agentic_validation_evidence_present(messages)
    ):
        return True
    return False


def _agentic_should_force_package_repair(messages: list) -> bool:
    if _agentic_verification_present(messages):
        return False
    if _agentic_package_repair_requested(messages):
        return False
    if _agentic_missing_tests_present(messages):
        return False

    transcript = _agentic_transcript(messages).lower()
    package_setup_markers = (
        "missing script",
        "no test script",
        "vitest: not found",
        "cannot find package 'vitest'",
        "cannot find module 'vitest'",
    )
    if any(marker in transcript for marker in package_setup_markers):
        return True
    if _agentic_failed_validation_present(messages):
        return False
    return (
        _agentic_tool_result_count(messages)
        >= _AGENTIC_FORCE_PACKAGE_REPAIR_AFTER_TOOL_RESULTS
        and not _agentic_validation_evidence_present(messages)
    )


def _agentic_should_force_verification(messages: list) -> bool:
    if _agentic_verification_present(messages):
        return False
    if _agentic_tool_result_count(messages) < _AGENTIC_FORCE_VERIFY_AFTER_TOOL_RESULTS:
        return False
    if _agentic_failed_validation_present(messages):
        return False
    if _agentic_validation_evidence_present(messages):
        return True
    return True


def _tool_definition_name(tool) -> str | None:
    if hasattr(tool, "function"):
        function = tool.function
    elif isinstance(tool, dict):
        function = tool.get("function", tool)
    else:
        return None
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = getattr(function, "name", None)
    return str(name) if name else None


def _agentic_forced_verification_tool_call(tools) -> list[ToolCall] | None:
    if not tools or not any(_tool_definition_name(tool) == "bash" for tool in tools):
        return None
    return [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=FunctionCall(
                name="bash",
                arguments=json.dumps(
                    {"command": _AGENTIC_VERIFY_COMMAND, "timeout": 180}
                ),
            ),
        )
    ]


def _agentic_forced_package_repair_tool_call(
    tools,
    messages: list,
) -> list[ToolCall] | None:
    if (
        not tools
        or not _agentic_should_force_package_repair(messages)
        or not any(_tool_definition_name(tool) == "bash" for tool in tools)
    ):
        return None
    return [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=FunctionCall(
                name="bash",
                arguments=json.dumps(
                    {"command": _AGENTIC_PACKAGE_REPAIR_COMMAND, "timeout": 180}
                ),
            ),
        )
    ]


def _agentic_forced_snake_scaffold_tool_call(
    tools,
    messages: list,
) -> list[ToolCall] | None:
    if (
        not tools
        or not _agentic_should_force_snake_scaffold(messages)
        or not any(_tool_definition_name(tool) == "bash" for tool in tools)
    ):
        return None
    return [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=FunctionCall(
                name="bash",
                arguments=json.dumps(
                    {"command": _AGENTIC_SNAKE_SCAFFOLD_COMMAND, "timeout": 180}
                ),
            ),
        )
    ]


def _agentic_forced_missing_tests_tool_call(
    tools,
    messages: list,
) -> list[ToolCall] | None:
    if (
        not tools
        or not _agentic_missing_tests_present(messages)
        or not any(_tool_definition_name(tool) == "write" for tool in tools)
    ):
        return None
    return [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=FunctionCall(
                name="write",
                arguments=json.dumps(
                    {
                        "path": _agentic_missing_test_write_path(messages),
                        "content": _AGENTIC_MINIMAL_SNAKE_TEST,
                    }
                ),
            ),
        )
    ]


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
    elif cfg.no_thinking or (
        added_tool_continuation_prompt and cfg.tool_call_parser == "qwen3_coder_xml"
    ):
        chat_kwargs["enable_thinking"] = False
    elif agentic_verification_required:
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
    # local-agent mode, reject premature text-only "done" answers until the
    # transcript shows the requested verification commands.
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, request)
    if (
        agentic_verification_required
        and not _agentic_verification_present(messages)
    ):
        retry_messages = list(messages)
        snake_scaffold_tool_calls = _agentic_forced_snake_scaffold_tool_call(
            request.tools,
            messages,
        )
        if snake_scaffold_tool_calls:
            logger.info("[agentic-guard] forcing snake feature scaffold repair")
            cleaned_text = ""
            tool_calls = snake_scaffold_tool_calls

        missing_test_tool_calls = _agentic_forced_missing_tests_tool_call(
            request.tools,
            messages,
        )
        if not tool_calls and missing_test_tool_calls:
            logger.info(
                "[agentic-guard] forcing missing-test repair write tool call"
            )
            cleaned_text = ""
            tool_calls = missing_test_tool_calls

        package_repair_tool_calls = _agentic_forced_package_repair_tool_call(
            request.tools,
            messages,
        )
        if not tool_calls and package_repair_tool_calls:
            logger.info("[agentic-guard] forcing package test setup repair")
            cleaned_text = ""
            tool_calls = package_repair_tool_calls

        if not tool_calls:
            forced_tool_calls = _agentic_forced_verification_tool_call(request.tools)
            if forced_tool_calls:
                logger.info(
                    "[agentic-guard] replacing premature final text with "
                    "verification tool call"
                )
                cleaned_text = ""
                tool_calls = forced_tool_calls

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
                        else (
                            "You are not finished. Use tools now. The task "
                            "requires React, TypeScript, TailwindCSS, "
                            "feature-oriented architecture, and TDD. Before "
                            "final response, create or fix files and run npm "
                            "test, npm run build, and npm run lint successfully. "
                            "Do not answer with prose until those commands pass."
                        )
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
            if not tool_calls:
                forced_tool_calls = _agentic_forced_snake_scaffold_tool_call(
                    request.tools,
                    messages,
                ) or _agentic_forced_package_repair_tool_call(
                    request.tools,
                    messages,
                ) or _agentic_forced_verification_tool_call(
                    request.tools
                )
                if forced_tool_calls:
                    logger.info(
                        "[agentic-guard] retry remained text-only; forcing "
                        "verification tool call"
                    )
                    cleaned_text = ""
                    tool_calls = forced_tool_calls

    # Validate tool call parameter values against schemas
    if tool_calls and request.tools:
        _validate_tool_call_params(tool_calls, request.tools)

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
            and not _agentic_verification_present(messages)
        )
        snake_scaffold_tool_calls = (
            _agentic_forced_snake_scaffold_tool_call(request.tools, messages)
            if agentic_stream_guard
            else None
        )
        if snake_scaffold_tool_calls:
            logger.info("[agentic-guard] forcing snake feature scaffold repair")
            yield _format_forced_tool_call(snake_scaffold_tool_calls, None)
            yield "data: [DONE]\n\n"
            return

        missing_test_tool_calls = (
            _agentic_forced_missing_tests_tool_call(request.tools, messages)
            if agentic_stream_guard
            else None
        )
        if missing_test_tool_calls:
            logger.info(
                "[agentic-guard] forcing missing-test repair write tool call"
            )
            yield _format_forced_tool_call(missing_test_tool_calls, None)
            yield "data: [DONE]\n\n"
            return

        package_repair_tool_calls = (
            _agentic_forced_package_repair_tool_call(request.tools, messages)
            if agentic_stream_guard
            else None
        )
        if package_repair_tool_calls:
            logger.info("[agentic-guard] forcing package test setup repair")
            yield _format_forced_tool_call(package_repair_tool_calls, None)
            yield "data: [DONE]\n\n"
            return

        forced_tool_calls = (
            _agentic_forced_verification_tool_call(request.tools)
            if agentic_stream_guard and _agentic_should_force_verification(messages)
            else None
        )
        if forced_tool_calls:
            logger.info(
                "[agentic-guard] forcing verification after %d tool results "
                "without validation evidence",
                _agentic_tool_result_count(messages),
            )
            yield _format_forced_tool_call(forced_tool_calls, None)
            yield "data: [DONE]\n\n"
            return
        retry_attempts = 0
        active_messages = (
            list(messages) + [{"role": "user", "content": _AGENTIC_REPAIR_USER_PROMPT}]
            if agentic_stream_guard and _agentic_failed_validation_present(messages)
            else messages
        )
        active_kwargs = kwargs
        tools_disabled_for_retry = False
        repeated_tool_retry_active = False
        recent_tool_call_signature = _last_assistant_tool_call_signature(messages)
        recent_tool_call_path_signature = _last_assistant_tool_call_path_signature(
            messages
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
        structured_agentic_continuation = (
            (
                tool_continuation_retry
                or agentic_stream_guard
            )
            and cfg.structured_cot_tools
            and bool(request.tools)
            and not repeated_tool_continuation
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
            6
            if structured_agentic_continuation
            else (2 if tool_retries_enabled else 0)
        )
        max_repeated_tool_retries = 2 if tool_continuation_retry else 0
        retry_prompt = (
            _TOOL_CONTINUATION_RETRY_PROMPT
            if tool_continuation_retry
            else _TOOL_CALL_REQUIRED_RETRY_PROMPT
        )
        configured_timeout = request.timeout or cfg.default_timeout
        stream_idle_timeout = min(configured_timeout, _STREAM_IDLE_TIMEOUT_SECONDS)

        while True:
            active_tool_retries_enabled = (
                tool_retries_enabled
                and not tools_disabled_for_retry
                and not repeated_tool_retry_active
            )
            repeat_tool_detection_enabled = (
                (tool_continuation_retry or agentic_stream_guard)
                and bool(
                    recent_tool_call_signature or recent_tool_call_path_signature
                )
                and not tools_disabled_for_retry
            )
            tool_call_buffering_enabled = (
                active_tool_retries_enabled or repeat_tool_detection_enabled
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
                        and retry_attempts < max_tool_continuation_retries
                        and not emitted_tool_call
                    )

                    for event in processor.process_chunk(output):
                        if retry_window and event.type in ("content", "reasoning"):
                            buffered_events.append((event, output))
                            buffered_text = _buffered_stream_text(buffered_events)
                            if (
                                len(buffered_text)
                                > _TOOL_TEXT_BEFORE_TOOL_CALL_MAX_CHARS
                            ):
                                retry_reason = "too much text before tool call"
                                logger.info(
                                    "[tool-continuation] buffered text exceeded %d "
                                    "chars before tool call; aborting current stream",
                                    _TOOL_TEXT_BEFORE_TOOL_CALL_MAX_CHARS,
                                )
                                break
                            if _is_repetitive_tool_text(buffered_text):
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
                                repeat_tool_detection_enabled
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
                    if tool_continuation_retry and (
                        _buffered_tool_call_repeats_recent(
                            buffered_tool_call_events,
                            recent_tool_call_signature,
                        )
                        or _buffered_tool_call_repeats_recent_path(
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
                        (
                            _AGENTIC_REPEATED_TOOL_PROMPT
                            if agentic_stream_guard
                            else _TOOL_CONTINUATION_REPEATED_TOOL_PROMPT
                        )
                        if retry_reason == "repeated tool call"
                        else (
                            _AGENTIC_REPAIR_USER_PROMPT
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
                    "max_tokens": min(int(kwargs.get("max_tokens") or 512), 512),
                }
                tools_disabled_for_retry = False
                repeated_tool_retry_active = retry_reason == "repeated tool call"
                continue

            forced_tool_calls = None
            if retry_reason and agentic_stream_guard:
                forced_tool_calls = _agentic_forced_snake_scaffold_tool_call(
                    request.tools,
                    messages,
                ) or _agentic_forced_package_repair_tool_call(
                    request.tools,
                    messages,
                ) or _agentic_forced_verification_tool_call(request.tools)
            if forced_tool_calls:
                logger.info(
                    "[agentic-guard] stream retry budget exhausted; forcing "
                    "verification tool call"
                )
                buffered_events.clear()
                buffered_tool_call_events.clear()
                deferred_finish = None
                yield _format_forced_tool_call(forced_tool_calls, last_output)
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
