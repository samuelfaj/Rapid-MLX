#!/usr/bin/env bash
set -u

if [ "$#" -lt 3 ]; then
  echo "usage: $0 PROFILE SERVER_ROOT MODEL FLAGS..." >&2
  exit 2
fi

PROFILE="$1"
SERVER_ROOT="$2"
MODEL="$3"
shift 3

BASE="${BENCH6_BASE:-/tmp/rapid-mlx-bench6}"
RUN_DIR="$BASE/$PROFILE"
SERVER_LOG="$BASE/$PROFILE.server.log"
OPENCODE_LOG="$BASE/$PROFILE.opencode.log"
RESULT="$BASE/$PROFILE.result.json"
PROMPT="create a REST api using express and bun and typescript and sequelize-typescript. It must be vertical sliced. You should create models, seeders and migrations. You must create unit tests for each service."
OPENCODE_TIMEOUT="${OPENCODE_TIMEOUT:-600}"
VALIDATION_TIMEOUT="${VALIDATION_TIMEOUT:-180}"
PORT="${BENCH6_PORT:-8010}"

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/AGENTS.md" <<'EOF'
Use the exact user request. Keep scope minimal but complete. Create User and Product vertical slices only. Use express, bun, typescript, sequelize-typescript. Create models, migrations, seeders, services, controllers, routes, app/server. Use bun test only; do not use jest, bun-jest, or reassign imported bindings. Unit tests should test services with simple in-memory fakes or pure repository injection. Add package scripts: test, build, start. Run bun install and bun test. Stop after tests pass or after one clear failing test report.
When using Bun test hooks, import them explicitly from bun:test, for example `import { describe, it, expect, beforeEach } from "bun:test"`. Do not keep working after tests pass.
Do not use the task tool or delegate to subagents; create and edit files directly in this workspace.
EOF

(
  cd "$SERVER_ROOT" && uv run rapid-mlx serve "$MODEL" \
    --served-model-name local --port "$PORT" --default-temperature 0 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder_xml \
    --max-tokens 4096 --timeout 300 "$@"
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

kill_tree() {
  local signal="$1"
  local pid="$2"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$signal" "$child"
  done
  kill "-$signal" "$pid" 2>/dev/null || true
}

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill_tree INT "$SERVER_PID"
    sleep 5
    kill_tree TERM "$SERVER_PID"
    sleep 2
    kill_tree KILL "$SERVER_PID"
  fi
}
trap cleanup EXIT

run_with_timeout() {
  local seconds="$1"
  shift
  "$@" &
  local child_pid=$!
  local start_ts
  start_ts=$(date +%s)
  while kill -0 "$child_pid" 2>/dev/null; do
    local now_ts
    now_ts=$(date +%s)
    if [ "$((now_ts - start_ts))" -ge "$seconds" ]; then
      kill -TERM "$child_pid" 2>/dev/null || true
      sleep 5
      kill -KILL "$child_pid" 2>/dev/null || true
      wait "$child_pid" 2>/dev/null || true
      return 124
    fi
    sleep 1
  done
  wait "$child_pid"
}

READY=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 2
done

START=$(date +%s)
if [ "$READY" != "1" ]; then
  OPENCODE_EXIT=999
  TIMEOUT=true
else
  (
    cd "$RUN_DIR" && run_with_timeout "$OPENCODE_TIMEOUT" opencode run \
      --model local/local \
      --format json \
      --dangerously-skip-permissions \
      --dir "$RUN_DIR" \
      "$PROMPT"
  ) > "$OPENCODE_LOG" 2>&1
  OPENCODE_EXIT=$?
  if [ "$OPENCODE_EXIT" = "124" ]; then
    TIMEOUT=true
  else
    TIMEOUT=false
  fi
fi
END=$(date +%s)

PROJECT_ROOT="$RUN_DIR"
HAS_PACKAGE=false
HAS_TEST_SCRIPT=false
HAS_TESTS=false
VALID_INSTALL=999
VALID_TEST=999

if [ -f "$PROJECT_ROOT/package.json" ]; then
  HAS_PACKAGE=true
  if jq -e ".scripts.test" "$PROJECT_ROOT/package.json" >/dev/null 2>&1; then
    HAS_TEST_SCRIPT=true
  fi
  (cd "$PROJECT_ROOT" && run_with_timeout "$VALIDATION_TIMEOUT" bun install) > "$BASE/$PROFILE.validation-install.log" 2>&1
  VALID_INSTALL=$?
  if [ "$HAS_TEST_SCRIPT" = true ]; then
    (cd "$PROJECT_ROOT" && run_with_timeout "$VALIDATION_TIMEOUT" bun test) > "$BASE/$PROFILE.validation-test.log" 2>&1
    VALID_TEST=$?
  fi
fi

if find "$PROJECT_ROOT" -name "*.test.ts" -o -name "*.spec.ts" | grep -q .; then
  HAS_TESTS=true
fi
PROJECT_FILES_COUNT=$(find "$PROJECT_ROOT" -type f \( -name "*.ts" -o -name "*.json" \) | wc -l | tr -d " ")

python3 - "$PROFILE" "$OPENCODE_EXIT" "$TIMEOUT" "$START" "$END" "$PROJECT_FILES_COUNT" "$HAS_PACKAGE" "$HAS_TEST_SCRIPT" "$HAS_TESTS" "$VALID_INSTALL" "$VALID_TEST" "$RESULT" "$SERVER_LOG" <<'PY'
import json
import re
import statistics
import sys

(
    profile,
    opencode_exit,
    timeout,
    start,
    end,
    files,
    has_pkg,
    has_test_script,
    has_tests,
    valid_install,
    valid_test,
    result,
    server_log,
) = sys.argv[1:]
text = open(server_log, errors="replace").read() if server_log else ""
prompts = [int(m.group(1)) for m in re.finditer(r"prompt_tokens=(\d+)", text)]
ttft = [float(m.group(1)) for m in re.finditer(r"first token after ([0-9.]+)s", text)]
tps = [float(m.group(1)) for m in re.finditer(r"Chat completion \(stream\): .*?\(([0-9.]+) tok/s\)", text)]
finish = {}
for match in re.finditer(r"finish_reason[=:]\s*['\"]?([a-zA-Z_]+)", text):
    finish[match.group(1)] = finish.get(match.group(1), 0) + 1

out = {
    "profile": profile,
    "opencode_exit": int(opencode_exit),
    "timeout": timeout == "true",
    "wall_seconds": int(end) - int(start),
    "project_files_count": int(files),
    "has_package_json": has_pkg == "true",
    "has_test_script": has_test_script == "true",
    "has_tests": has_tests == "true",
    "validation_install": int(valid_install),
    "validation_test": int(valid_test),
    "validation_ok": int(valid_install) == 0 and int(valid_test) == 0,
    "requests_count": len(tps),
    "cache_hits": len(re.findall(r"cache_fetch.* HIT", text)),
    "sse_timeout_seen": "SSE read timed out" in text,
    "max_prompt_tokens": max(prompts) if prompts else None,
    "median_prompt_tokens": statistics.median(prompts) if prompts else None,
    "median_ttft": statistics.median(ttft) if ttft else None,
    "median_effective_tps": statistics.median(tps) if tps else None,
    "finish_reasons": finish,
    "result_path": result,
}
open(result, "w").write(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
PY

cleanup
