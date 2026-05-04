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

## Five-Run Reliability Check

After the first result, the benchmark was repeated five times per profile with
fresh profile names. Each run used a newly emptied workspace directory under
`/tmp/rapid-mlx-bench3`. The benchmark script cleanup was tightened to kill the
server process tree, preventing stale servers from staying on port `8010`
between retries.

| Profile | r1 | r2 | r3 | r4 | r5 | Validation rate | Wall median, all | Wall median, valid only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no prefix cache | 84s fail | 394s pass | 272s pass | 143s fail | 216s fail | 2/5 | 216s | 333s |
| Agentic policy auto | 76s fail | 195s pass | 505s fail | 442s fail | 198s pass | 2/5 | 198s | 196.5s |

Five-run summary:

- Baseline valid runs: `2/5`, valid wall median `333s`, valid median TTFT `5.9s`, valid median effective TPS `30.8`.
- Auto valid runs: `2/5`, valid wall median `196.5s`, valid median TTFT `2.62s`, valid median effective TPS `58.27`.
- Valid-output wall speedup: `333 / 196.5 = 1.69x`.
- All-run wall median speedup: `216 / 198 = 1.09x`.

Reliability conclusion: `--agentic-speculative-policy auto` remained faster on
the successful outputs, but it did not improve pass rate in this five-run set.
Both profiles passed validation `40%` of the time. The remaining quality
variance appears to come from the agent's generated app/test choices rather than
from server timeout behavior; no retry timed out in the corrected five-run set.

Artifacts:

- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_baseline_no_prefix_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_agentic_policy_auto_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_baseline_r1_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_baseline_r2_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_baseline_r3_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_baseline_r4_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_baseline_r5_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_auto_r1_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_auto_r2_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_auto_r3_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_auto_r4_direct_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_benchmark4_reliable_auto_r5_direct_600s.result.json`

## Notes

An earlier auto-policy run was discarded because the harness allowed opencode to delegate to the `task` tool. The final comparison forbids subagents for both profiles, so both runs exercise the same direct-edit workload.
