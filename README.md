# Lightning-MLX

Lightning-MLX is a performance-focused fork of [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) for running local MLX models on Apple Silicon through an OpenAI-compatible API.

This fork keeps the original Rapid-MLX goal: serve local models with a simple `/v1/chat/completions` interface. The extra focus here is agentic coding performance: prefix cache reuse, tool-result compaction, tool-call parsing, DFlash/DDTree speculative decoding experiments, n-gram prompt lookup, and automatic policy control for tool-heavy workflows.

## Status

The current best validated profile for opencode-style coding agents is not a forced DFlash/DDTree profile. It is the target model with prefix cache enabled, tool-result compaction enabled by the server, deterministic decoding, the correct tool parser, and:

```bash
--agentic-speculative-policy auto
```

For the tested Qwen3.6 35B A3B 4-bit model, this profile was very fast because repeated tool turns were prefill-bound, not decode-bound. In that shape, avoiding speculative decode overhead matters more than trying to draft every token.

## Install

```bash
git clone https://github.com/samuelfaj/lightning-mlx.git
cd lightning-mlx
uv sync
uv pip install -e .
```

Both commands work:

```bash
lightning-mlx --help
rapid-mlx --help
```

`rapid-mlx` remains available as compatibility alias.

## Quick Start

```bash
lightning-mlx serve /path/to/model \
  --served-model-name local \
  --port 8010
```

Call it like an OpenAI-compatible server:

```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "Say hello from Lightning-MLX"}
    ]
  }'
```

Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8010/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Write a tiny haiku about MLX."}],
)

print(response.choices[0].message.content)
```

## Best Agentic Coding Config

Use this for opencode/Codex-style local coding workloads with many tool calls:

```bash
uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --agentic-speculative-policy auto \
  --max-tokens 4096 \
  --timeout 300
```

Equivalent installed command:

```bash
lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --agentic-speculative-policy auto \
  --max-tokens 4096 \
  --timeout 300
```

Why this is fast:

- Prefix cache is enabled by default.
- Repeated tool-turn prompts reuse cached KV state.
- Tool-result compaction reduces the amount of history that must be prefetched again.
- `--default-temperature 0` makes tool and code generation more deterministic.
- `--tool-call-parser qwen3_coder_xml` matches the Qwen3 Coder XML tool format.
- `--agentic-speculative-policy auto` adds phase classification and policy telemetry.
- Without `--drafter`, the server does not enter DFlash mode, so it avoids drafter overhead.

Important: this command does not use a draft model. It is fast because cache and compaction dominate this workload.

## What `--agentic-speculative-policy auto` Does

`--agentic-speculative-policy auto` is a server-side policy flag for agentic tool workflows. It is designed to prevent speculative features from hurting performance or tool-call reliability when the model is doing coding-agent loops.

It currently works in two modes, depending on whether DFlash is configured.

### Without `--drafter`

This is the best measured path for the 35B opencode benchmark.

When no `--drafter` is provided:

- The server uses the regular batched target engine.
- Prefix cache stays enabled unless `--disable-prefix-cache` is passed.
- Tool-result compaction is used by chat flow.
- The server classifies the current agentic phase.
- The server records metrics such as cache hits, prompt tokens, generated tokens, TTFT, effective TPS, and policy fields.
- DFlash/DDTree/n-gram decode is not activated because there is no draft model.

So `auto` does not magically load a drafter. It uses only what the server was configured with.

This is why the command below can be extremely fast:

```bash
uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --agentic-speculative-policy auto \
  --max-tokens 4096 \
  --timeout 300
```

The speed comes mostly from avoiding repeated full prefill, not from draft-token verification.

### With `--drafter`

When `--drafter` is provided, DFlash mode is enabled. Then `auto` becomes a gate around DFlash/DDTree/n-gram choices.

The policy classifies the request into one of these phases:

| Phase | Meaning | Policy bias |
| --- | --- | --- |
| `initial_scaffold` | Early coding turn, prompt still smaller, model may generate bulk code. | Speculation may help if output is long. |
| `long_text_or_code` | Long non-tool text/code generation. | Speculation may help if acceptance is good. |
| `tool_json` | Model likely needs to emit tool-call JSON/XML. | Prefer target fallback for correctness. |
| `repair` | After error, failed command, traceback, or failing test. | Prefer target fallback. |
| `validation` | Around test/build/diagnostic flow. | Prefer target fallback. |
| `finalization` | Tests passed or final answer likely. | Prefer target fallback. |

The DFlash auto policy uses:

- phase
- prompt token count
- remaining prefill size
- max token budget
- greedy mode, usually `temperature=0`
- recent acceptance ratio
- cooldown state
- whether DDTree budget is configured
- whether n-gram lookup is configured

Initial rule shape:

```text
if phase in {repair, validation, finalization, tool_json}:
    use target fallback
elif prompt_tokens > DFLASH_AGENTIC_POLICY_MAX_PREFILL:
    use target fallback
elif recent_acceptance < DFLASH_AGENTIC_POLICY_MIN_ACCEPTANCE:
    use target fallback for cooldown window
elif max_tokens > 512 and greedy and ddtree_budget > 0:
    use ddtree or ddtree-ngram
else:
    use target fallback
```

Environment knobs:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `DFLASH_AGENTIC_POLICY_MAX_PREFILL` | `8000` | Above this prompt size, auto avoids DFlash because prefill dominates. |
| `DFLASH_AGENTIC_POLICY_MIN_ACCEPTANCE` | `0.35` | Below this recent acceptance ratio, auto disables speculation temporarily. |
| `DFLASH_AGENTIC_POLICY_COOLDOWN` | `3` | Number of following requests to keep fallback after poor acceptance. |

## Why DFlash/DDTree/n-gram Did Not Win This Agentic Benchmark

DFlash, DDTree, and n-gram lookup optimize decode when draft tokens are accepted cheaply. That can help when the model is producing long continuous text or code.

The opencode benchmark was different:

- The agent repeatedly sent large prompts with growing tool history.
- Most time was spent in prefill, not pure decode.
- Tool turns often needed structured tool-call output, where speculative decode can be fragile.
- Repair and validation turns benefit more from correctness and cache reuse than from draft-token aggression.
- DFlash mode disables continuous batching and adds drafter/model coordination overhead.

So the best profile was:

- no `--drafter`
- prefix cache on
- tool compaction on
- deterministic decoding
- correct tool parser
- `--agentic-speculative-policy auto`

DFlash is still useful for other shapes. It just was not the winning profile for this specific long tool-history coding task.

## DFlash/DDTree Example

Use this only when testing decode-heavy workloads or DFlash-specific behavior:

```bash
DFLASH_DRAFT_SINK=64 DFLASH_DRAFT_WINDOW=1024 \
lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash \
  --dflash-ddtree-budget 4 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --agentic-speculative-policy auto \
  --max-tokens 4096 \
  --timeout 300
```

Optional experimental n-gram flags:

```bash
  --thinking-ngram \
  --ngram-num-draft-tokens 4 \
  --ngram-size 2 \
  --ngram-min-matches 1
```

Notes:

- `--drafter` triggers DFlash mode.
- DFlash mode is text-only.
- DFlash is mutually exclusive with `--enable-mtp` and `--mllm`.
- DFlash mode is single-request oriented; continuous batching is disabled there.
- `--dflash-ddtree-budget 4` is the tested DDTree budget suggestion for Qwen3.6 A3B.

## Benchmarks

Benchmark document:

```text
BENCHMARK4.md
```

Workload prompt:

```text
create a REST aopencode using express and bun and typescript and sequelize-typescript. It must be vertical sliced. You should create models, seeders and migrations. You must create unit tests for each service.
```

Single best validated comparison:

| Profile | Wall seconds | Validation | Requests | Cache hits | Median TTFT | Median effective TPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Baseline, no prefix cache | 458 | pass | 44 | 0 | 7.1s | 16.15 |
| Agentic policy auto | 121 | pass | 16 | 15 | 1.5s | 71.5 |

Five-run reliability check:

| Profile | Validation rate | Wall median, all | Wall median, valid only | Valid-run median TPS |
| --- | ---: | ---: | ---: | ---: |
| Baseline, no prefix cache | 2/5 | 216s | 333s | 30.8 tok/s |
| Agentic policy auto | 2/5 | 198s | 196.5s | 58.27 tok/s |

Conclusion:

- Auto was faster on successful outputs.
- Auto did not improve quality pass rate in the five-run set.
- Both profiles passed validation 40% of the time.
- Remaining quality variance appears to come from generated app/test choices, not server timeout behavior.

## Serve Arguments

The authoritative source is:

```bash
uv run rapid-mlx serve --help
```

### Core Server

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `model` | required | Model path or Hugging Face id to serve. |
| `--served-model-name` | model path if omitted | Model name exposed through the API. |
| `--host` | `0.0.0.0` | Host bind address. |
| `--port` | `8000` | HTTP port. |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`; default `INFO` | Python and uvicorn log level. |
| `--timeout` | `300` | Default request timeout in seconds. |
| `--api-key` | unset | Optional API key. If unset, auth is disabled. |
| `--cors-origins` | `*` behavior when unset | Allowed CORS origins. |
| `--rate-limit` | `0` | Requests per minute per client. `0` disables. |
| `--tui` | off | Run live monitor TUI with the server. |

### Generation Defaults

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--max-tokens` | `4096` | Default generation token limit. |
| `--default-temperature` | model default | Override request temperature when client omits it. |
| `--default-top-p` | model default | Override request top-p when client omits it. |
| `--stream-interval` | `1` | Tokens batched before streaming. Higher can improve throughput; `1` is smoother. |

### Batching And Prefill

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--max-num-seqs` | `256` | Max concurrent sequences. |
| `--prefill-batch-size` | `8` | Prefill batch size. |
| `--completion-batch-size` | `32` | Decode/completion batch size. |
| `--continuous-batching` | on | Enable continuous batching. |
| `--prefill-step-size` | `2048` | Prompt prefill chunk size. Larger can improve throughput but uses more memory. |
| `--chunked-prefill-tokens` | `0` | Max prefill tokens per scheduler step. `0` disables. Helps prevent starvation during long prefills. |

### Prefix Cache And Memory

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--enable-prefix-cache` | enabled | Enable KV prefix caching for repeated prompts. |
| `--disable-prefix-cache` | off | Disable prefix caching. Mostly useful for baselines. |
| `--prefix-cache-size` | `100` | Legacy max entry count when memory-aware cache is disabled. |
| `--cache-memory-mb` | auto | Explicit cache memory limit in MB. |
| `--cache-memory-percent` | `0.20` | Fraction of available RAM for cache auto-sizing. |
| `--no-memory-aware-cache` | off | Use legacy entry-count cache instead of memory-aware cache. |
| `--pin-system-prompt` | off | Pin system prompt cache blocks to reduce eviction under pressure. |
| `--gpu-memory-utilization` | `0.90` | Metal allocation/emergency cache clear threshold. Use cautiously. |

### KV Cache Compression

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--kv-cache-quantization` | off | Quantize stored KV caches to reduce memory. |
| `--kv-cache-quantization-bits` | `4` or `8`; default `8` | Bit width for KV quantization. |
| `--kv-cache-quantization-group-size` | `64` | Group size for KV quantization. |
| `--kv-cache-min-quantize-tokens` | `256` | Minimum cached token count before quantization applies. |
| `--kv-cache-turboquant` | off | Experimental V-cache TurboQuant compression. Mutually exclusive with `--kv-cache-quantization`. |
| `--kv-cache-turboquant-bits` | `3` or `4`; auto if unset | TurboQuant bit width. |
| `--kv-cache-turboquant-group-size` | `32` | TurboQuant group size. |

### Paged Cache

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--use-paged-cache` | off | Experimental paged KV cache. |
| `--paged-cache-block-size` | `64` | Tokens per cache block. |
| `--max-cache-blocks` | `1000` | Max paged cache blocks. |

### Tool Calling

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--enable-auto-tool-choice` | off | Enable model-controlled tool choice. Requires `--tool-call-parser`. |
| `--tool-call-parser` | `auto`, `mistral`, `qwen`, `qwen3_coder`, `qwen3_coder_xml`, `qwen3_xml`, `llama`, `hermes`, `deepseek`, `kimi`, `granite`, `nemotron`, `xlam`, `functionary`, `glm47`, `minimax`, `harmony`, `gpt-oss`, `gemma4` | Parser for model-specific tool-call syntax. |
| `--enable-tool-logits-bias` | off | Bias structural tool-call tokens. Currently supports minimax. |
| `--mcp-config` | unset | Path to MCP config for tool integration. |

Parser guidance:

- Qwen3 Coder XML: use `--tool-call-parser qwen3_coder_xml`.
- Generic Qwen: use `--tool-call-parser qwen`.
- If unsure, try `--tool-call-parser auto`, then inspect tool-call correctness.

### Reasoning And Agentic Controls

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--reasoning-parser` | `gemma4`, `qwen3`, `deepseek_r1`, `gpt_oss`, `harmony`, `minimax` | Extract reasoning tags into `reasoning_content`. |
| `--no-thinking` | off | Disable reasoning parser even when auto-detected. Thinking tokens appear as normal content. |
| `--structured-cot` | off | Constrain thinking to structured `<think>` format with decode-time masking. |
| `--structured-cot-tools` | off | Apply structured CoT only to tool-calling requests. |
| `--structured-cot-token-budget` | `256` | Approximate budget for structured CoT lines. |
| `--agentic-guard` | off | Benchmark-specific repair guard for tool workflows. |
| `--agentic-speculative-policy` | `off`, `auto`; default `off` | Automatically gate speculative paths for tool-heavy agentic workflows. |

For the 35B benchmark, `--no-thinking` was intentionally not used.

### Speculative Prefill

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--speculative-prefill` | off | Conservative draft-scored prompt compression before target prefill. Falls back to original prompt if safety checks fail. |
| `--speculative-prefill-draft-model` | unset | Optional small MLX/HF model used for token importance scoring. |
| `--speculative-prefill-ratio` | `0.85` | Target compressed prompt token ratio. |
| `--speculative-prefill-min-tokens` | `128` | Minimum prompt length before speculative prefill can apply. |

This is separate from DFlash. It tries to reduce prefill work by compressing the prompt, not by drafting decode tokens.

### MTP

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--enable-mtp` | off | Enable Multi-Token Prediction for models with built-in MTP heads. |
| `--mtp-num-draft-tokens` | `1` | Draft tokens per MTP step. |
| `--mtp-optimistic` | off | Skip MTP acceptance check. Faster but can produce wrong tokens; not recommended for code. |

MTP is mutually exclusive with DFlash.

### DFlash, DDTree, And n-gram

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--drafter` | unset | Path/HF id of a DFlash drafter checkpoint. Enables DFlash mode. |
| `--dflash-block-size` | drafter default | Override drafter block size. |
| `--dflash-no-adaptive` | off | Disable adaptive block sizing. |
| `--dflash-block-min` | `8` | Adaptive lower bound. |
| `--dflash-block-max` | `22` | Adaptive upper bound. |
| `--dflash-turboquant-bits` | unset | Optional TurboQuant bits for target model KV cache under DFlash. |
| `--dflash-ddtree-budget` | `0` | Enable DDTree verification with node budget. Use `4` for Qwen3.6 A3B experiments. |
| `--dflash-ddtree-block-size` | block size/drafter config | Override DDTree draft block size. |
| `--thinking-ngram` | off | Enable aggressive prompt-lookup n-gram only inside generated `<think>` blocks. |

Compatibility/experimental hidden flags accepted by the parser:

| Argument | Values / default | Meaning |
| --- | --- | --- |
| `--dflash-fallback-mode` | `ngram`, `ar`, `none` | Compatibility knob from earlier experiments. Current DDTree path does not rely on it. |
| `--dflash-disable-threshold` | float | Compatibility adaptive-disable threshold. |
| `--dflash-disable-window` | int | Compatibility adaptive-disable window. |
| `--dflash-disable-cooldown` | int | Compatibility adaptive-disable cooldown. |
| `--ngram-num-draft-tokens` | int | Hidden n-gram draft length knob. |
| `--ngram-size` | int | Hidden n-gram size knob. |
| `--ngram-min-matches` | int | Hidden n-gram match threshold. |
| `--thinking-ngram-num-draft-tokens` | int | Hidden thinking n-gram draft length. |
| `--thinking-ngram-size` | default `1` | Hidden thinking n-gram size. |
| `--thinking-ngram-min-matches` | default `1` | Hidden thinking n-gram match threshold. |

### Multimodal, Embeddings, Cloud

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--mllm` | off | Force multimodal model loading. Mutually exclusive with DFlash. |
| `--embedding-model` | unset | Preload embedding model. |
| `--cloud-model` | unset | Route large-context requests to a LiteLLM cloud model. |
| `--cloud-threshold` | `20000` | New-token threshold for cloud routing. |
| `--cloud-api-base` | unset | Custom OpenAI-compatible cloud API base. |
| `--cloud-api-key` | env/provider default | Cloud API key override. |

### GC

| Argument | Default / choices | Meaning |
| --- | --- | --- |
| `--gc-control` | enabled | Disable Python GC during generation to reduce latency spikes. |
| `--no-gc-control` | off | Allow normal Python GC during generation. |

### Deprecated Compatibility Flags

These are accepted so old scripts do not crash, but they should not be used for new configs:

| Argument | Replacement / note |
| --- | --- |
| `--simple-engine` | Deprecated; no effect. |
| `--kv-bits` | Use `--kv-cache-quantization-bits`. |
| `--kv-group-size` | Use `--kv-cache-quantization-group-size`. |
| `--draft-model` | Deprecated; use MTP or DFlash-specific `--drafter` depending on model. |
| `--num-draft-tokens` | Deprecated old draft knob. |
| `--specprefill` | Deprecated; use `--speculative-prefill`. |
| `--specprefill-threshold` | Deprecated. |
| `--specprefill-keep-pct` | Deprecated. |
| `--specprefill-draft-model` | Deprecated; use `--speculative-prefill-draft-model`. |

## Choosing A Profile

### Agentic coding with tools

Start here:

```bash
uv run rapid-mlx serve /path/to/model \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --agentic-speculative-policy auto \
  --max-tokens 4096 \
  --timeout 300
```

Then only change one thing at a time.

### Decode-heavy chat or code generation

Try DFlash only when the model will generate long continuous outputs and tool-call correctness is not the bottleneck:

```bash
uv run rapid-mlx serve /path/to/target \
  --drafter /path/to/drafter \
  --dflash-ddtree-budget 4 \
  --default-temperature 0 \
  --agentic-speculative-policy auto
```

### Memory pressure

Try in this order:

1. Reduce `--max-tokens`.
2. Lower `--cache-memory-percent`.
3. Use `--kv-cache-quantization`.
4. Test `--kv-cache-turboquant` only if you accept experimental behavior.
5. Avoid DFlash if the drafter pushes memory over the edge.

### Tool-call failures

Check in this order:

1. Correct `--tool-call-parser`.
2. Use `--default-temperature 0`.
3. Avoid `--mtp-optimistic`.
4. Avoid forced DFlash/DDTree on repair/validation-heavy loops.
5. Keep `--agentic-speculative-policy auto` instead of forcing speculative settings manually.

## Development

Run focused tests:

```bash
uv run pytest tests/test_agentic_policy.py tests/test_speculative_prefill.py
```

Run lint on touched files:

```bash
uv run ruff check vllm_mlx tests
```

Run the benchmark harness:

```bash
scripts/run_opencode_bench3_once.sh PROFILE /path/to/model FLAGS...
```

Benchmark outputs are written under:

```text
/tmp/rapid-mlx-bench3
```

## Credits

Lightning-MLX is forked from Rapid-MLX by Raul Lanchai and the Rapid-MLX/vllm-mlx contributors.

Original project:

```text
https://github.com/raullenchai/Rapid-MLX
```

License:

```text
Apache-2.0
```

See [LICENSE](LICENSE).
