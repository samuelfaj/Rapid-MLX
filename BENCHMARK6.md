# BENCHMARK6

## Scope

Prompt:

```text
create a REST api using express and bun and typescript and sequelize-typescript. It must be vertical sliced. You should create models, seeders and migrations. You must create unit tests for each service.
```

Harness: `scripts/run_opencode_bench6_once.sh`.

Benchmark date: 2026-05-04.

Note: request text conflicts on model choice. It asks for same 35B-A3B base, but the final target command uses `Qwen3.6-27B-UD-Q4_K_XL-mlx` with `Qwen3.6-27B-DFlash`. These runs use the final target 27B base for both raw and optimized profiles.

## Commands

Raw profile:

```bash
BENCH6_BASE=/tmp/rapid-mlx-bench6 OPENCODE_TIMEOUT=900 VALIDATION_TIMEOUT=240 \
  scripts/run_opencode_bench6_once.sh raw-27b-N \
  /Users/samuelfajreldines/dev/rapid-mlx \
  /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx
```

Optimized profile:

```bash
BENCH6_BASE=/tmp/rapid-mlx-bench6 OPENCODE_TIMEOUT=900 VALIDATION_TIMEOUT=240 \
  scripts/run_opencode_bench6_once.sh ddtree-auto-targetcache10-27b-N \
  /Users/samuelfajreldines/Rapid-MLX-ddtree \
  /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --agentic-speculative-policy auto \
  --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-27B-DFlash \
  --dflash-ddtree-budget 4
```

## Results

| Profile | Runs | Timeout | Validation OK | Avg wall s | Avg median tok/s | Avg cache hits |
|---|---:|---:|---:|---:|---:|---:|
| raw-27b | 3 | 3/3 | 3/3 | 905.0 | 6.20 | 0 |
| ddtree-auto-targetcache10-27b | 3 | 0/3 | 3/3 | 758.0 | 15.07 | 10689 |

Raw runs:

| Run | wall s | median tok/s | requests | max prompt tokens | validation |
|---|---:|---:|---:|---:|---|
| raw-27b-1 | 905 | 7.10 | 20 | 26320 | pass |
| raw-27b-2 | 905 | 6.30 | 19 | 25143 | pass |
| raw-27b-3 | 905 | 5.20 | 19 | 23823 | pass |

Optimized runs:

| Run | wall s | median tok/s | requests | cache hits | max prompt tokens | validation |
|---|---:|---:|---:|---:|---:|---|
| ddtree-auto-targetcache10-27b-1 | 732 | 14.00 | 53 | 10400 | 29174 | pass |
| ddtree-auto-targetcache10-27b-2 | 708 | 16.20 | 41 | 10665 | 28511 | pass |
| ddtree-auto-targetcache10-27b-4 | 834 | 15.00 | 49 | 11002 | 30079 | pass |

## Findings

Optimized final runs passed validation 3/3 and avoided the 900 second opencode timeout 3/3. Average wall time improved from 905s to 758s, and median effective throughput improved from 6.20 tok/s to 15.07 tok/s.

The main win is target-prefix-cache inside target-only agentic phases. `--agentic-speculative-policy auto` still preserves quality by routing `tool_json`, `repair`, `validation`, and `finalization` to target decode. The target path now stores reusable prompt KV snapshots before decode mutates the cache, so repeated opencode tool/repair turns can prefill only the uncached suffix. Final optimized runs showed 10k+ target-prefix-cache hits each.

DDTree and n-gram remain decode-only. Speculative prefill is used only for uncached suffixes, and the auto policy treats `--drafter` as capability: without a drafter there is no DDTree/DFlash path; with a drafter, auto still chooses target-only for quality-critical phases.

## Code Validation

Unit/regression slice after final edits:

```text
103 passed, 3 deselected
```

Covered behavior:

- DDTree prefix-state cache defaults on when DDTree is enabled, with env override.
- Target-prefix-cache defaults on when DDTree is enabled, uses a bounded LRU, and avoids exact cached logprobs for tool calls.
- Auto policy treats `temperature=None` and `temperature=0` as greedy.
- Auto policy can route large uncached scaffold/long-code phases to DDTree when speculative prefill is enabled.
- Non-stream chat now passes prefix boundaries like stream chat.
- TUI token/s uses active request decode/effective metrics instead of stale last DDTree generation TPS.

## Status

Accepted for this benchmark: final optimized profile is faster than raw, avoids timeout, preserves validation quality, and records real target-prefix-cache hits.
