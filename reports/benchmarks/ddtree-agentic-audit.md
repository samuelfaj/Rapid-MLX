# DDTree Agentic Audit

Scope: compare current DDTree/ngram hot path against the agentic policy TODO, with focus on choosing speculative work only when it should win wall-clock time.

## Findings

### Cache rollback / restore

The current DDTree path already has a tree-aware fast commit path in `vllm_mlx/speculative/ddtree/cache.py`. `tree_aware_path_commit` packs only the accepted tree path after the prefix and mutates attention KV in place. This is the preferred path when `_can_tree_aware_commit()` succeeds.

The fallback slow path still snapshots and restores cache state before replaying accepted tokens. Evidence: `snapshot_caches()` and `restore_caches()` are still used by the slow path in `vllm_mlx/speculative/ddtree/engine.py`. This is correct, but slower. Policy therefore should not assume DDTree wins solely because a drafter exists; it should require acceptance length and effective TPS evidence.

Action taken: policy now records `acceptance_length`, `effective_tps`, selected path, bucket, and budget per completed request. Low AL causes bucket cooldown, so slow rollback/poor acceptance does not keep hurting future agentic turns.

### Python loop / synchronization

Hot-path risks remain visible:

- `_build_tree_from_mlx_logits()` performs `mx.eval(top_token_ids, top_log_probs)` after top-k selection.
- DDTree verification converts `posterior_mx.tolist()` for path matching.
- Stop-text checks decode emitted tokens when streaming or stop markers are active.
- Extended prefix-cache capture deep-copies final cache state.

These costs are acceptable only when speculative acceptance is good. The policy now uses effective TPS and AL, not decode TPS alone, so Python overhead is included in the decision.

Action taken: target-prefix-cache hit logs stay debug-level, avoiding TUI spam during repeated cache hits.

### Top-K / budget expansion

Current DDTree tree construction uses `topk = min(tree_budget, vocab_size)` and builds nodes under the same `tree_budget`. Larger budget can improve AL, but also increases tree build and verify work. A fixed budget of `4` is therefore only a conservative capability default.

Action taken: policy now supports `DFLASH_AGENTIC_DDTREE_BUDGET_CANDIDATES`, defaults safely to the configured `--dflash-ddtree-budget`, and selects the best budget per phase/prompt/max-token bucket when AL and effective TPS support it.

## Current Decision Model

The policy buckets history by:

- agentic phase
- prompt size bucket
- max-token bucket
- greedy vs sampled request

Each bucket tracks target/cache, DDTree, and DDTree+n-gram outcomes. DDTree is cooled down when acceptance length is poor. N-gram is cooled down independently when n-gram acceptance is poor. Target/cache can become the measured winner when it beats DDTree effective TPS by the configured margin.

## Verification

- Focused tests cover TUI truth, AL history, effective TPS, bucket cooldown, target winner memory, deterministic exploration, adaptive budget selection, and n-gram acceptance gate.
- `scripts/agentic_speculative_bench.py --mock --profile all --workload all` covers benchmark matrix output without requiring a model.
- Live DFlash audit on port `8011` generated a complete REST API project and repaired tests to green, but the server later terminated with Metal OOM during repeated finalization/tool calls at roughly 28k-40k prompt tokens. Independent validation in `/tmp/rapid-pi-audit` passed: `bun test` reported 32 pass, 0 fail.
- Live target-only audit on port `8011` stayed stable longer and continued creating the project, but `pi` did not complete within the allotted run window. Independent validation in `/tmp/rapid-pi-audit-target` was only partial at interruption time: users/products files existed, but the full required orders slice was not complete.
- Result: implementation tasks are complete; whole-thread goal is not complete because the required `pi` run did not finish cleanly without errors.
