# DDTree / n-gram performance study

Date: 2026-04-30

## Goal

Find more throughput beyond the current best Rapid-MLX DDTree + n-gram setup for
Qwen3.6 27B dense:

```bash
uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-27B-DFlash \
  --dflash-ddtree-budget 4 \
  --dflash-no-adaptive \
  --dflash-fallback-mode ngram \
  --ngram-num-draft-tokens 4 \
  --ngram-size 2 \
  --ngram-min-matches 1 \
  --served-model-name qwen3.6-27b-crud-ngram \
  --port 8010 \
  --no-thinking \
  --default-temperature 0
```

## Specialist findings

- DDTree: biggest remaining code opportunities are CPU tree build, `.tolist()`
  syncs, posterior tree walk, hidden concat, and exact-tree-attention thresholds.
  These are real code projects, not simple flag wins.
- n-gram: current confidence is weak. `ngram-min-matches` means minimum draft
  length, not match frequency. Retrying n-gram after cooldown looked logical, but
  benchmark showed it made throughput worse.
- MLX memory/cache: `DDTREE_CLEAR_CACHE_INTERVAL`, draft sink/window, exact tree
  attention, and prefix cache are key knobs. Most trade memory vs throughput.
- opencode: must use `local/qwen3.6-27b-crud-ngram`; provider must not send
  sampling temperature, otherwise Rapid falls back to plain DFlash.
- metrics: use wall time and OpenAI `usage.completion_tokens` for request-level
  truth. TUI/engine TPS can be misleading on tool-call streams and concurrent
  polling.
- CLI/env: best low-risk sweep knobs are DDTree budget, clear-cache interval,
  draft KV window, exact tree attention, and n-gram size/draft length.

## Direct API sweep

Prompt:

```text
Create a complete CRUD REST API in Python FastAPI for users, projects, tasks,
comments, and tags. Use SQLAlchemy models, Pydantic schemas, routers,
dependency-injected DB sessions, create/list/get/update/delete endpoints, and
tests. Use consistent style and repeat the full pattern for each resource.
```

All direct API tests used `max_tokens=256`, `temperature=0`, non-streaming
`/v1/chat/completions`.

| Config | Result |
| --- | ---: |
| `budget=4`, `ngram-size=2`, `draft=4`, no adaptive | 27.48 tok/s wall, 28.27 engine tok/s |
| Same + `DDTREE_CLEAR_CACHE_INTERVAL=0` | 27.49 tok/s wall, 28.29 engine tok/s |
| `budget=6`, `ngram-size=2`, `draft=4`, no adaptive | 25.49 tok/s wall, 26.16 engine tok/s |
| Same + `DFLASH_DRAFT_WINDOW=512` | 27.43 tok/s wall, 28.22 engine tok/s |
| Patched n-gram cooldown retry | 24.65 tok/s wall, 25.29 engine tok/s |

Outcome: no tested knob beat the existing best config. The n-gram cooldown patch
was reverted because it increased n-gram attempts but reduced throughput.

## Earlier long-form API results

These were the more reliable 6400-token runs from the previous loop:

| Workload | Config | Result |
| --- | --- | ---: |
| TCP/UDP explanatory prose | Pure DDTree budget 4 no adaptive | 23.8 tok/s |
| TCP/UDP explanatory prose | n-gram size 3 draft 4 + DDTree | 22.6 tok/s |
| TCP/UDP explanatory prose | n-gram size 3 draft 8 + DDTree | 21.3 tok/s |
| CRUD repetitive coding | Pure DDTree budget 4 no adaptive | 29.8 tok/s |
| CRUD repetitive coding | n-gram size 2 draft 4 + DDTree | 35.2 tok/s |
| CRUD repetitive coding | n-gram size 2 draft 8 + DDTree | 34.3 tok/s |
| CRUD repetitive coding | n-gram size 1 draft 4 + DDTree | 34.0 tok/s |

The best measured gain remains `35.2 / 29.8 = 1.18x`, about 18% over pure
DDTree on repetitive CRUD generation.

## opencode validation

opencode config was adjusted outside this repo:

```json
"model": "local/qwen3.6-27b-crud-ngram",
"small_model": "local/qwen3.6-27b-crud-ngram",
"temperature": false
```

Without `temperature=false`, opencode sent a non-greedy request and Rapid used
plain `dflash` instead of `ddtree-ngram`.

opencode command:

```bash
opencode run \
  --model local/qwen3.6-27b-crud-ngram \
  --dangerously-skip-permissions \
  "Create a small FastAPI CRUD API for users and projects. Use SQLAlchemy models, Pydantic schemas, routers, create/list/get/update/delete endpoints, and pytest tests. Use consistent style and repeat the same pattern for both resources."
```

Result:

- Wall time: 966.02s
- Generated project completed.
- Tests: 16 passed.
- Server request count: 25 DDTree/n-gram requests.
- Generated tokens across recorded requests: 4925.
- Average engine `generation_tps`: 44.23 tok/s.
- Median engine `generation_tps`: 20.58 tok/s.
- Average end-to-end request TPS: 4.47 tok/s.
- Average n-gram acceptance: 0.166.
- Average DDTree fast-path ratio: 0.869.

opencode wall time is dominated by many tool calls, shell commands, package
installation, pytest, and long prefill. It is not comparable to raw decode TPS.

## Final recommendation

Use this as fastest validated config for repetitive coding / CRUD / boilerplate:

```bash
uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-27B-DFlash \
  --dflash-ddtree-budget 4 \
  --dflash-no-adaptive \
  --dflash-fallback-mode ngram \
  --ngram-num-draft-tokens 4 \
  --ngram-size 2 \
  --ngram-min-matches 1 \
  --served-model-name qwen3.6-27b-crud-ngram \
  --port 8010 \
  --no-thinking \
  --default-temperature 0
```

Use pure DDTree for prose or unknown workloads:

```bash
--dflash-ddtree-budget 4 --dflash-no-adaptive
```

Do not use the tested alternatives as defaults:

- `--dflash-ddtree-budget 6`: slower in this loop.
- `DFLASH_DRAFT_WINDOW=512`: no gain.
- `DDTREE_CLEAR_CACHE_INTERVAL=0`: no gain in this short loop.
- n-gram cooldown retry patch: slower, reverted.

Next real code improvements worth doing later:

- remove CPU syncs and `.tolist()` in DDTree loop;
- optimize/cachify tree build;
- make n-gram confidence frequency-aware, not only draft-length-based;
- make opencode/tool-call benchmark request-correlated in metrics;
- expose actual request mode clearly when a non-greedy request falls back to
  DFlash.
