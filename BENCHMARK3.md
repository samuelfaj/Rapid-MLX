# BENCHMARK3

Status: meta 35B atingida. Replicacao 27B executada e validada.

Prompt usado exatamente no `opencode run`:

```text
create a REST aopencode using express and bun and typescript and sequelize-typescript. It must be vertical sliced. You should create models, seeders and migrations. You must create unit tests for each service.
```

Nunca foi usado `--no-thinking`.

Todos os runs abaixo usaram opencode contra Rapid-MLX OpenAI-compatible em `http://127.0.0.1:8010/v1`, `--enable-auto-tool-choice`, parser `qwen3_coder_xml`, `temperature=0`, `max_tokens=4096`. O workspace do benchmark incluiu um `AGENTS.md` local para manter o escopo minimo e verificavel: User/Product vertical slices, Bun test, sem Jest, seeders/migrations/models/services/controllers/routes.

## Resultado principal 35B

Modelo: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`.

| Run | Config | opencode | validacao | wall | mediana TPS efetivo | TTFT mediano | cache hits | SSE timeout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen36_35b_opencode_tool_compaction_baseline_no_prefix_600s` | baseline sem prefix-cache | exit 0 | `bun install` 0, `bun test` 0 | 472.24s | 18.96 | 6.178s | 0 | nao |
| `qwen36_35b_opencode_tool_compaction_prefix_only_600s` | prefix-cache segmentado + compaction de tool results | exit 0 | `bun install` 0, `bun test` 0 | 207.25s | 54.44 | 2.973s | 24 | nao |

Speedup 35B:

- Wall-clock: `472.24 / 207.25 = 2.28x`.
- TPS efetivo mediano: `54.44 / 18.96 = 2.87x`.
- Ambos os lados finalizaram opencode e passaram `bun test`; comparacao e valida.

## DDTree / DFlash / n-gram

| Run | Config | Resultado |
| --- | --- | --- |
| `qwen36_35b_opencode_true_dflash_no_agentic_fallback_900s` | `--drafter /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash`, DDTree budget 4, fallback n-gram, `DFLASH_AGENTIC_TARGET_FALLBACK=0` | DFlash/DDTree/n-gram ativaram fluxo especulativo, tool calls sairam, mas o run bateu timeout 900s e a mediana de TPS efetivo ficou ~11.3. Nao foi escolhido para o perfil vencedor. |

Conclusao: DDTree/DFlash/n-gram estao funcionais, mas neste fluxo agentic longo eles pioraram o gargalo real. O ganho vencedor veio de cache de prefixo segmentado, compaction de tool results e finalizacao correta do stream.

## Replicacao 27B

Modelo: `/Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx`.

| Run | Config | opencode | validacao | wall | mediana TPS efetivo | TTFT mediano | cache hits | SSE timeout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen36_27b_opencode_tool_compaction_baseline_no_prefix_600s` | baseline sem prefix-cache | timeout 600s | `bun install` 0, `bun test` 1 | 605s | 7.90 | 19.55s | 0 | nao |
| `qwen36_27b_opencode_prefix_structured_short_finalfix_1050s` | prefix-cache segmentado + compaction + `--structured-cot-tools --structured-cot-token-budget 96` | exit 0 | `bun install` 0, `bun test` 0 | 629s | 13.15 | 3.80s | 23 | nao |

O baseline 27B nao validou, entao nao ha speedup fim-a-fim justo com qualidade igual. Ainda assim, a replicacao mostra que o perfil otimizado finaliza opencode, passa `bun test`, elimina timeout de SSE e reduz TTFT mediano de 19.55s para 3.80s.

## Artefatos

Resultados JSON e logs:

- `/tmp/rapid-mlx-bench3/qwen36_35b_opencode_tool_compaction_baseline_no_prefix_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_opencode_tool_compaction_prefix_only_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_35b_opencode_true_dflash_no_agentic_fallback_900s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_27b_opencode_tool_compaction_baseline_no_prefix_600s.result.json`
- `/tmp/rapid-mlx-bench3/qwen36_27b_opencode_prefix_structured_short_finalfix_1050s.result.json`

Arquivos `.server.log`, `.opencode.log`, `.validation-install.log` e `.validation-test.log` correspondentes ficam no mesmo diretorio.
