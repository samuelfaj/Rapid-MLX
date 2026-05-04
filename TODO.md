# Rapid DDTree / n-gram TODO

Goal: make `--agentic-speculative-policy auto` choose DDTree, n-gram, or target-only only when that path is expected to win wall-clock time for the current agentic phase.

Context:

- Target-only with prefix cache is often fastest for `pi`/Codex-style tool loops because most requests are prefill/cache dominated.
- DDTree helps mainly when decode is long enough, prompt is inside the sweet spot, and draft acceptance is high.
- n-gram helps only when the output repeats prompt text or obvious code/text spans; forcing it everywhere adds overhead and can hurt tool-call phases.
- The TUI now shows `-` when no speculative work actually happened. Keep that behavior.
- Current adaptive DDTree trigger uses recent target/cache TPS. That is useful but too weak; policy must learn from actual acceptance length and wall-time outcomes.

## Success Criteria

- [x] TUI truth: `path` shows `ddtree`, `ngram`, `ng+tree`, or `-` based on actual executed work, never planned/stale mode.
- [x] Policy truth: each completed request records enough metrics to explain why current path won or lost.
- [x] Agentic auto chooses target-only for tool JSON, repair, validation, finalization, and oversized prompt/prefill cases.
- [x] Agentic auto explores DDTree/ngram occasionally, but stops using them when measured acceptance or speed is poor.
- [x] DDTree budget is adaptive by phase/prompt bucket, not fixed at `4` forever.
- [x] Benchmarks report wall-clock, TTFT, prefill TPS, decode TPS, acceptance length, acceptance ratio, and chosen path per request.
- [x] README startup guidance matches measured behavior, not assumptions.

## Phase 1: Metrics Needed For Real Decisions

- [x] Add `acceptance_length` metric to completed request history.
  - Definition: `accepted_tokens / speculative_steps` when `speculative_steps > 0`.
  - Store alongside `acceptance_ratio`, `generation_tps`, `prompt_tokens`, `generated_tokens`, `agentic_phase`, `mode`, `tree_budget`, and n-gram counters.
  - Verify: unit test with DDTree request where accepted/proposed/steps produce expected AL.

- [x] Separate decode TPS from request effective TPS.
  - Decode TPS should measure generated tokens after first token.
  - Effective TPS should include prefill and total elapsed.
  - Policy should prefer wall-clock/effective TPS when comparing target vs DDTree, and use decode TPS only as supporting signal.
  - Verify: synthetic history test where high decode TPS but bad total elapsed does not mark DDTree as winner.

- [x] Record policy decision fields for every request.
  - Required fields: `phase`, `selected_path`, `reason`, `prompt_bucket`, `max_tokens_bucket`, `cached_tokens`, `uncached_tokens`, `target_recent_tps`, `ddtree_recent_al`, `ddtree_recent_effective_tps`, `cooldown_remaining`.
  - TUI can show compact subset; JSON metrics should keep full detail.
  - Verify: `/metrics` or internal stats exposes fields for target-only and DDTree paths.

- [x] Track n-gram metrics separately from DDTree metrics.
  - Required fields: `ngram_cycles`, `ngram_proposed_tokens`, `ngram_accepted_tokens`, `ngram_acceptance_ratio`, `ngram_disabled_cycles`, `ngram_tool_guard_cycles`.
  - Policy should not treat DDTree acceptance as n-gram acceptance.
  - Verify: test where DDTree works but n-gram has zero cycles keeps n-gram score unknown, not good.

## Phase 2: Learned Policy / Bandit

- [x] Replace fixed target TPS trigger with per-bucket scoring.
  - Bucket keys:
    - `agentic_phase`
    - prompt bucket: `0-4k`, `4-8k`, `8-16k`, `16-32k`, `32k+`
    - generated budget bucket: `0-256`, `257-512`, `513-1024`, `1025+`
    - greedy vs non-greedy
  - Candidate paths: `target-prefix-cache`, `ddtree`, `ddtree-ngram`, `ngram`.
  - Score primary: recent effective TPS or tokens per wall second.
  - Score secondary: acceptance length and TTFT.
  - Verify: unit test where same phase has different winners in different prompt buckets.

- [x] Add DDTree acceptance-length gates.
  - Proposed defaults:
    - `DFLASH_AGENTIC_DDTREE_AL_DISABLE=2.5`
    - `DFLASH_AGENTIC_DDTREE_AL_ENABLE=4.0`
    - `DFLASH_AGENTIC_DDTREE_AL_STRONG=6.0`
  - If recent AL `< disable`, enter cooldown for that bucket.
  - If recent AL `>= enable`, allow DDTree.
  - If recent AL `>= strong`, allow higher budget exploration.
  - Verify: tests for disable, enable, and strong paths.

- [x] Add per-bucket cooldown.
  - Current cooldown is too global for agent loops.
  - Cooldown should apply to `(phase, prompt_bucket, max_tokens_bucket, path)`.
  - Target-only should remain available always.
  - Verify: poor DDTree in `repair` or `32k+` bucket must not block DDTree exploration in `initial_scaffold` / `8-16k`.

- [x] Add deterministic exploration.
  - Proposed env:
    - `DFLASH_AGENTIC_POLICY_EXPLORE_EVERY=8`
    - `DFLASH_AGENTIC_POLICY_EXPLORE_MIN_OUTPUT_TOKENS=64`
  - Explore only when phase can benefit and prompt is inside configured max prompt.
  - Skip exploration for tool JSON, repair, validation, finalization.
  - Verify: sequence test where request 8 explores, request 9 returns to best known path if exploration lost.

- [x] Add path winner memory.
  - Keep small LRU/deque history per bucket.
  - Store last N outcomes per path.
  - Winner requires enough samples or a strong margin.
  - Suggested margin: `winner_effective_tps >= other_effective_tps * 1.10`.
  - Verify: noisy single outlier does not permanently switch path.

## Phase 3: Adaptive DDTree Budget

- [x] Sweep DDTree budgets automatically during exploration.
  - Candidate budgets: `2`, `4`, `6`, `8`, `12`, `16`.
  - Do not assume budget `4` is universally best.
  - Larger budget should only stay enabled if AL and wall-time improve.
  - Verify: tests where budget `8` wins one bucket and budget `4` wins another.

- [x] Keep conservative defaults.
  - Default startup can still use `--dflash-ddtree-budget 4` as base capability.
  - Auto policy may internally choose lower/higher explored budget when evidence says so.
  - Add env:
    - `DFLASH_AGENTIC_DDTREE_BUDGET_CANDIDATES=2,4,6,8,12,16`
    - `DFLASH_AGENTIC_DDTREE_BUDGET_EXPLORE=1`
  - Verify: invalid candidate env falls back safely to `4`.

- [x] Record budget in TUI/history.
  - Recent requests should show `block` and `budget` or equivalent compact field.
  - Avoid widening TUI too much; prefer replacing ambiguous column if needed.
  - Verify: TUI snapshot test.

## Phase 4: Better n-gram Gating

- [x] Use n-gram only when prompt/output pattern predicts reuse.
  - Signals:
    - `long_text_or_code` phase
    - generated text likely code/text, not tool call
    - prompt has repeated identifiers/imports/types
    - no active tool-call XML/JSON guard
  - Skip for repair/validation/tool JSON/finalization.
  - Verify: classifier tests for tool XML and code generation.

- [x] Add n-gram acceptance gates.
  - Proposed env:
    - `DFLASH_AGENTIC_NGRAM_MIN_ACCEPTANCE=0.30`
    - `DFLASH_AGENTIC_NGRAM_COOLDOWN=3`
  - If n-gram cycles happen but acceptance low, disable n-gram for bucket cooldown.
  - Verify: n-gram poor acceptance disables `ddtree-ngram` but still allows plain `ddtree` if DDTree AL is good.

- [x] Evaluate standalone n-gram vs DDTree+n-gram.
  - Candidate paths should be scored separately.
  - `ngram` may win prompt-copy workloads without drafter overhead.
  - `ddtree-ngram` may win only if both proposal sources help.
  - Verify: synthetic policy tests with path-specific scores.

## Phase 5: Runtime Performance Audit

- [x] Audit DDTree cache rollback/restore path.
  - Compare against Luce-style fast cache mutation/rollback.
  - Goal: reduce Python overhead and repeated cache copying.
  - Verify: microbenchmark before/after for same prompt and budget.

- [x] Reduce Python loop and synchronization overhead.
  - Search for per-token `mx.eval`, host transfers, tokenizer decode, or list conversions inside hot loops.
  - Move stable tree tensors to device once per cycle where possible.
  - Verify: profiler or timing counters show lower per-cycle overhead.

- [x] Audit top-K candidate expansion.
  - Luce data suggests too-large top-K can hurt.
  - Test top-K candidates like `8`, `16`, `32`.
  - Verify: budget/top-K matrix benchmark with AL and wall time.

- [x] Keep target-prefix-cache logging quiet.
  - `cache_fetch HIT` should remain debug-level, not TUI spam.
  - Verify: run server with TUI and repeated cache hits; screen does not flood logs.

## Phase 6: Benchmarks

- [x] Add agentic benchmark script.
  - Inputs:
    - model path
    - optional drafter path
    - port
    - prompt corpus
    - flags matrix
  - Outputs:
    - JSONL per request
    - summary table
    - recommended startup command
  - Verify: script runs one short local smoke without model by mocked metrics, and real model when paths exist.

- [x] Benchmark profiles.
  - Profile A: target-only + prefix cache.
  - Profile B: drafter + auto adaptive.
  - Profile C: forced DDTree budget `4`.
  - Profile D: adaptive budget sweep.
  - Profile E: n-gram enabled for long text only.
  - Verify: report includes winner by workload, not one global winner.

- [x] Benchmark workloads.
  - Initial scaffold: empty repo, create full REST API.
  - Long code generation: generate large file/module.
  - Tool loop: many short tool calls.
  - Repair: failing test then fix.
  - Validation/final answer.
  - Prompt sizes: `2k`, `4k`, `8k`, `16k`, `32k`.
  - Max tokens: `512`, `1024`, `2048`, `4096`.

- [x] Add regression thresholds.
  - Target-only must not become slower when no drafter is configured.
  - Auto+drafters must not force speculative decode in tool JSON/repair/validation.
  - TUI path must remain truthful.

## Phase 7: Docs / Startup Guidance

- [x] Update README after benchmark data.
  - Explain when to run without `--drafter`.
  - Explain when `--drafter + auto` helps.
  - Explain why `--dflash-ddtree-budget 4` is capability/default, not universal best.
  - Explain adaptive budget/env knobs.

- [x] Add recommended commands.
  - Conservative fastest for agent loops:
    - no drafter
    - prefix cache
    - `--agentic-speculative-policy auto`
  - Experimental adaptive:
    - drafter
    - `--agentic-speculative-policy auto`
    - DDTree budget candidates enabled
  - Benchmark command:
    - run matrix and print recommendation.

- [x] Document TUI columns.
  - `path = -`: no speculative work actually executed.
  - `ddtree`: DDTree proposed and verified tokens.
  - `ngram`: n-gram cycles accepted/proposed tokens.
  - `ng+tree`: both n-gram and DDTree contributed.
  - `acc/cyc`: accepted tokens per speculative cycle, not generic accuracy.

## Open Questions

- [x] Should adaptive policy live entirely inside `DFlashEngine`, or should phase/path scoring move into a separate policy module?
- [x] Should budget adaptation require explicit opt-in first, even if candidate list default is conservative?
- [x] Can standalone n-gram run without DFlash engine when no drafter is configured, or is that outside current architecture?
- [x] What minimum live benchmark size is enough before updating README recommendations?
- [x] Should TUI show one compact `winner` field or keep all details only in JSON metrics?

## Immediate Next Step

- [x] Implement Phase 1 first.
  - Add acceptance length to history.
  - Add effective-vs-decode TPS distinction.
  - Add unit tests.
  - No behavior change yet.
