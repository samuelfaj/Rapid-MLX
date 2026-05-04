# Benchmark 4: Agentic Speculative Policy Auto

## Goal

Implement `--agentic-speculative-policy auto` so the server can choose when to use or avoid speculative paths during tool-heavy agentic coding. The target was the same 35B opencode workload, with equal final quality and faster wall-clock time than the non-prefix-cache baseline.

## Implementation

The policy records per-request agentic context and exposes it through request metrics. Chat requests with tools are classified into phases: `initial_scaffold`, `repair`, `validation`, `finalization`, `tool_json`, or `long_text_or_code`.

For DFlash runs, the policy uses phase, prompt length, recent speculative acceptance, cooldown state, max tokens, and greedy mode to choose between target fallback and DDTree/n-gram speculative decode. Repair, validation, finalization, short tool JSON, very long prefill, low recent acceptance, or cooldown force target fallback. Long initial scaffold or long text/code can use DDTree/n-gram when configured.

For the default best server profile without a drafter, the flag still activates phase propagation and metrics while the established winning path remains prefix cache plus tool-result compaction.

## Exact Workload

Script:

```bash
scripts/run_opencode_bench3_once.sh PROFILE MODEL [FLAGS...]
```

Prompt:

```text
create a REST aopencode using express and bun and typescript and sequelize-typescript. It must be vertical sliced. You should create models, seeders and migrations. You must create unit tests for each service.
```

AGENTS instructions used by both final runs:

```text
Use the exact user request. Keep scope minimal but complete. Create User and Product vertical slices only. Use express, bun, typescript, sequelize-typescript. Create models, migrations, seeders, services, controllers, routes, app/server. Use bun test only; do not use jest, bun-jest, or reassign imported bindings. Unit tests should test services with simple in-memory fakes or pure repository injection. Add package scripts: test, build, start. Run bun install and bun test. Stop after tests pass or after one clear failing test report.
When using Bun test hooks, import them explicitly from bun:test, for example `import { describe, it, expect, beforeEach } from "bun:test"`. Do not keep working after tests pass.
Do not use the task tool or delegate to subagents; create and edit files directly in this workspace.
```

Common server flags:

```bash
uv run rapid-mlx serve "$MODEL" \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --max-tokens 4096 \
  --timeout 300
```

`--no-thinking` was not used.

## Final Commands

Baseline:

```bash
scripts/run_opencode_bench3_once.sh \
  qwen36_35b_benchmark4_baseline_no_prefix_direct_600s \
  /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --disable-prefix-cache
```

Auto policy:

```bash
scripts/run_opencode_bench3_once.sh \
  qwen36_35b_benchmark4_agentic_policy_auto_direct_600s \
  /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --agentic-speculative-policy auto
```

## Results

| Profile | Wall seconds | Validation | Requests | Cache hits | Median prompt tokens | Median TTFT | Median effective TPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, no prefix cache | 458 | pass | 44 | 0 | 22965 | 7.1s | 16.15 |
| Agentic policy auto | 121 | pass | 16 | 15 | 15033 | 1.5s | 71.5 |

Quality matched: both runs exited opencode with code `0`, created `package.json`, had a `test` script, created tests, completed `bun install`, and passed `bun test`.

Speed:

- Wall-clock speedup: `458 / 121 = 3.79x`.
- Median effective TPS speedup: `71.5 / 16.15 = 4.43x`.
- Median TTFT improved from `7.1s` to `1.5s`.

Artifacts:

- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_baseline_no_prefix_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_agentic_policy_auto_direct_600s.result.json`

## Notes

An earlier auto-policy run was discarded because the harness allowed opencode to delegate to the `task` tool. The final comparison forbids subagents for both profiles, so both runs exercise the same direct-edit workload.
