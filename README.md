# Lightning-MLX

Fast local OpenAI-compatible inference for Apple Silicon.

Lightning-MLX is a performance-focused fork of [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). The project keeps Rapid-MLX's core goal: run local models on macOS with an OpenAI-compatible API. This fork tracks a more aggressive path around DFlash/DDTree speculative decoding, tool-call reliability, and agentic coding workloads.

## Credits

Lightning-MLX is forked from Rapid-MLX by Raul Lanchai and the Rapid-MLX/vllm-mlx contributors.

Original project:

https://github.com/raullenchai/Rapid-MLX

License:

Apache-2.0. See [LICENSE](LICENSE).

## What This Fork Focuses On

- Apple Silicon local inference through MLX.
- OpenAI-compatible `/v1/chat/completions` server.
- DFlash/DDTree speculative decoding paths.
- Structured tool calling for local coding agents.
- Streaming safeguards for repeated, partial, or stalled tool calls.
- Mac-first command name: `lightning-mlx`.

## Install From Source

```bash
git clone https://github.com/samuelfaj/lightning-mlx.git
cd lightning-mlx
uv sync
uv pip install -e .
```

After editable install, the main command is:

```bash
lightning-mlx --help
```

The legacy `rapid-mlx` command is still kept as an alias for compatibility.

## Serve A Model

```bash
lightning-mlx serve /path/to/model \
  --served-model-name local \
  --port 8010
```

Then call it with any OpenAI-compatible client:

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

## Best Agentic Coding Server Config

This is the fastest validated configuration for opencode-style agentic coding
with tools and finalization. It keeps prefix caching enabled, uses segmented
prefix cache and tool-result compaction from the server, and avoids forcing
DFlash/DDTree on long tool workflows.

```bash
lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --max-tokens 4096 \
  --timeout 300
```

For the replicated 27B run, use the same server shape plus short structured
tool reasoning:

```bash
lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --max-tokens 4096 \
  --timeout 300 \
  --structured-cot-tools \
  --structured-cot-token-budget 96
```

Do not add `--no-thinking` for these benchmarks. It was intentionally not used.

Validated result for the 35B opencode benchmark:

- Baseline without prefix cache: 472.24s, opencode exit 0, `bun test` passed.
- Best config above: 207.25s, opencode exit 0, `bun test` passed.
- End-to-end wall-clock speedup: 2.28x.

## DFlash Example For Decode-Heavy Runs

DFlash/DDTree/n-gram are available, but they were not the winning configuration
for long agentic coding with many tool results. In the opencode benchmark,
forcing DFlash/DDTree/n-gram timed out and was slower because the bottleneck was
repeated prefill over tool history, not pure decode.

```bash
DFLASH_DRAFT_SINK=64 DFLASH_DRAFT_WINDOW=1024 \
lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash \
  --dflash-ddtree-budget 4 \
  --dflash-no-adaptive \
  --dflash-fallback-mode ngram \
  --thinking-ngram \
  --ngram-num-draft-tokens 4 \
  --ngram-size 2 \
  --ngram-min-matches 1 \
  --served-model-name local \
  --port 8010 \
  --structured-cot \
  --structured-cot-tools \
  --default-temperature 0 \
  --tui
```

## Python Client Example

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

## Development

Run focused tests:

```bash
uv run pytest tests/test_chat_stream_tool_continuation.py -q
```

Run the full test suite when changing shared engine, parser, or route behavior:

```bash
uv run pytest
```

Format and lint policy follows the existing Python project configuration in [pyproject.toml](pyproject.toml).

## Repository Relationship

This repository is intended to remain visibly connected to the original Rapid-MLX project while using `main` for the performance fork line. Upstream history is preserved so original authorship and commit ancestry remain available.
