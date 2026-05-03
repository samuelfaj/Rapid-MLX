# Benchmark 2: Qwen3.6 35B A3B 4bit e Qwen3.6 27B

Prompt fixo:

`create a REST api using express and bun and typescript and sequelize-typescript. It must be vertical sliced. You should create models, seeders and migrations. You must create unit tests for each service.`

Meta: achar configuracao que supere baseline em pelo menos 1.50x, mantendo criterio agentico do `BENCHMARK.md`: `pi` termina, projeto nasce em pasta limpa, validacao local passa.

## Resultado curto

- Melhor 35B validado ate agora: DFlash + DDTree budget 4 + fallback ngram + structured-cot tools + agentic guard + pin system prompt + speculative prefill. Validou, mas foi mais lento que baseline: 4m50.2s contra 3m18.2s.
- Melhor 27B por wall-clock bruto: target-only + `--no-thinking` + agentic guard + pin system prompt. Cortou timeout de 420.5s para 280.26s, speedup bruto 1.50x, mas ainda deu timeout e nao validou.
- Melhor 27B por throughput: DFlash otimizado, 22.16 tok/s contra baseline 20.93 tok/s, apenas 1.06x e tambem sem terminar.
- MTP 27B nao serviu neste setup: precisei expor temporariamente o sidecar como `model-mtp.safetensors`; o run caiu para ~1.17 tok/s e tambem timeout.

Conclusao atual: nenhum perfil novo atingiu a meta critica com qualidade valida. O unico perfil validado continua sendo o 35B otimizado do `BENCHMARK.md`, mas ele nao supera baseline em performance.

## Tabela

| modelo | perfil | pi finalizou | validou | timeout | wall | tok/s mediana | speedup wall bruto | criterio 1.50x valido |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B A3B 4bit | baseline | 1/1 | 0/1 | 0 | 3m18.2s | 113.11 | 1.00x | nao |
| Qwen3.6 35B A3B 4bit | DFlash/DDTree/spec-prefill/guard | 1/1 | 1/1 | 0 | 4m50.2s | 97.53 | 0.68x | nao |
| Qwen3.6 35B A3B 4bit | target-only/no-thinking/guard | 0/1 | 0/1 | abortado manualmente depois de >270s | >4m30s | n/a | <0.73x | nao |
| Qwen3.6 35B A3B 4bit | DFlash budget2/adaptive/spec-prefill/guard | 0/1 | 0/1 | 1 | 2m12.1s | 99.93 | 1.50x bruto | nao |
| Qwen3.6 35B A3B 4bit | DFlash validado + appended pi system prompt | 0/1 | 0/1 | 1 | 2m12.0s | 101.49 | 1.50x bruto | nao |
| Qwen3.6 35B A3B 4bit | DFlash warm-cache 2 runs | 1/2 | 0/2 | 1 | 10m00.1s / 7m35.2s | 94.46 / 95.90 | 0.44x bruto no melhor run | nao |
| Qwen3.6 35B A3B 4bit | target no-thinking + restricted tools + strict system | 0/1 | 0/1 | 1 | 2m12.5s | 111.96 | 1.50x bruto | nao |
| Qwen3.6 27B UD Q4_K_XL | baseline | 0/1 | 0/1 | 1 | 7m00.5s | 20.93 | 1.00x | nao |
| Qwen3.6 27B UD Q4_K_XL | DFlash/DDTree/spec-prefill/guard | 0/1 | 0/1 | 1 | 7m00.2s | 22.16 | 1.00x | nao |
| Qwen3.6 27B UD Q4_K_XL | no-thinking/guard/pin | 0/1 | 0/1 | 1 | 4m40.3s | 21.51 | 1.50x bruto | nao |
| Qwen3.6 27B UD Q4_K_XL | no-thinking/guard/pin/max_tokens=2048 | 0/1 | 0/1 | 1 | 5m00.3s | 20.69 | 1.40x bruto | nao |
| Qwen3.6 27B UD Q4_K_XL | MTP optimistic/guard/pin | 0/1 | 0/1 | 1 | 4m40.1s | 1.17 | 1.50x bruto | nao |
| Qwen3.6 27B UD Q4_K_XL | DFlash + appended pi system prompt | 0/1 | 0/1 | 1 | 4m40.5s | 1.23 | 1.50x bruto | nao |
| Qwen3.6 27B UD Q4_K_XL | no-thinking + strict appended system | 0/1 | 0/1 | 1 | 4m40.1s | 21.34 | 1.50x bruto | nao |
| Qwen3.6 27B UD Q4_K_XL | no-thinking + restricted tools + strict system | 0/1 | 0/1 | 1 | 4m40.3s | 22.53 | 1.50x bruto | nao |
| Qwen3.6 27B UD Q4_K_XL | no-thinking + custom system prompt | 0/1 | 0/1 | 1 | 4m40.3s | 22.45 | 1.50x bruto | nao |
| Qwen3.6 27B UD Q4_K_XL | structured-cot budget 256 + custom system | 0/1 | 0/1 | 1 | 4m40.2s | 22.80 | 1.50x bruto | nao |

## Melhor config 35B encontrada

```bash
uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit \
  --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash \
  --dflash-ddtree-budget 4 \
  --dflash-no-adaptive \
  --dflash-fallback-mode ngram \
  --thinking-ngram \
  --ngram-num-draft-tokens 4 \
  --ngram-size 2 \
  --ngram-min-matches 1 \
  --structured-cot-tools \
  --agentic-guard \
  --pin-system-prompt \
  --speculative-prefill \
  --speculative-prefill-draft-model /Users/samuelfajreldines/dev/models/Qwen3-1.7B-4bit-mlx \
  --speculative-prefill-ratio 0.85 \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --max-tokens 4096 \
  --timeout 300
```

Evidencia: `BENCHMARK.md` registra `pi` terminado, validacao 1/1, `bun test` 14/14 e `bun run build` pass em `/tmp/rapid-mlx-manual-pi6`.

## Melhor config 27B encontrada

Config mais rapida em wall-clock bruto, mas ainda sem sucesso completo:

```bash
uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --max-tokens 4096 \
  --timeout 300 \
  --no-thinking \
  --structured-cot-tools \
  --agentic-guard \
  --pin-system-prompt
```

Evidencia: `/tmp/rapid-mlx-bench2/qwen36_27b_no_thinking_guard_rest.result.json`. Wall 280.26s contra baseline 420.5s, mas `pi_finished=0/1`, `validation_success=0/1`.

Config 27B mais promissora por completude parcial:

```bash
uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-27B-DFlash \
  --dflash-ddtree-budget 4 \
  --dflash-no-adaptive \
  --dflash-fallback-mode ngram \
  --thinking-ngram \
  --ngram-num-draft-tokens 4 \
  --ngram-size 2 \
  --ngram-min-matches 1 \
  --structured-cot-tools \
  --agentic-guard \
  --pin-system-prompt \
  --speculative-prefill \
  --speculative-prefill-draft-model /Users/samuelfajreldines/dev/models/Qwen3-1.7B-4bit-mlx \
  --speculative-prefill-ratio 0.85 \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --max-tokens 4096 \
  --timeout 300
```

Evidencia: `/tmp/rapid-mlx-bench2/qwen36_27b_optimized_rest.result.json`. Gerou estrutura vertical, models, seeders, migrations e arquivos de teste, mas `pi` deu timeout e o projeto falhou em `bun test`/`tsc`.

## Falhas observadas

- 27B baseline gerou estrutura melhor que no-thinking, mas nao criou unit tests/scripts e deu timeout.
- 27B DFlash gerou arquivos de teste, mas `package.json` nao tinha scripts. `bun test` falhou por import invalido `Allow` em `sequelize-typescript`; `tsc --noEmit` falhou por tipos de `Sequelize`, seeders e services.
- 27B no-thinking reduziu tempo bruto, mas simplificou demais: nao criou migrations/seeders completas nem testes.
- 27B restricted/custom-system criou scripts `test` e `build`, mas ainda falhou validacao por uso incorreto de Bun mock API, import proibido `Allow`, ou testes sem inicializar Sequelize.
- 35B restricted/custom-system nao criou `package.json` antes do gate de 132s; a tentativa warm-cache terminou um segundo run, mas sem validacao.
- 27B MTP optimistic ficou muito mais lento no decode observado. O sidecar existe em `mtp-sidecar/model-mtp.safetensors`, mas o loader espera `model-mtp.safetensors` no root; symlink temporario foi criado para teste e removido depois.

## Artefatos

- 35B baseline/otimizado: `BENCHMARK.md`, `/tmp/rapid-mlx-bench`, `/tmp/rapid-mlx-manual-pi6`.
- 27B baseline: `/tmp/rapid-mlx-bench2/qwen36_27b_baseline_rest.result.json`.
- 27B DFlash: `/tmp/rapid-mlx-bench2/qwen36_27b_optimized_rest.result.json`.
- 27B no-thinking: `/tmp/rapid-mlx-bench2/qwen36_27b_no_thinking_guard_rest.result.json`.
- 27B no-thinking max 2048: `/tmp/rapid-mlx-bench2/qwen36_27b_no_thinking_guard_max2048_rest.result.json`.
- 27B MTP: `/tmp/rapid-mlx-bench2/qwen36_27b_mtp_guard_rest.result.json`.
- 27B DFlash + appended pi system prompt: `/tmp/rapid-mlx-bench2/qwen36_27b_dflash_agent_prompt_rest.result.json`.
- 35B DFlash budget2/adaptive: `/tmp/rapid-mlx-bench2/qwen36_35b_dflash_budget2_adaptive_rest.result.json`.
- 35B DFlash + appended pi system prompt: `/tmp/rapid-mlx-bench2/qwen36_35b_dflash_agent_prompt_rest.result.json`.
- 35B DFlash warm-cache: `/tmp/rapid-mlx-bench2/qwen36_35b_optimized_warm_cache_rest.result.json`.
- 35B no-thinking restricted strict: `/tmp/rapid-mlx-bench2/qwen36_35b_no_thinking_restricted_strict_rest.result.json`.
- 27B no-thinking strict system: `/tmp/rapid-mlx-bench2/qwen36_27b_no_thinking_strict_system_rest.result.json`.
- 27B no-thinking restricted strict: `/tmp/rapid-mlx-bench2/qwen36_27b_no_thinking_restricted_strict_rest.result.json`.
- 27B no-thinking custom system: `/tmp/rapid-mlx-bench2/qwen36_27b_no_thinking_custom_system_rest.result.json`.
- 27B structured budget256 custom system: `/tmp/rapid-mlx-bench2/qwen36_27b_structured_budget256_custom_system_rest.result.json`.

## Proximo caminho

Ainda falta encontrar perfil valido 1.50x. Proximas tentativas de maior chance:

1. Testar 35B DFlash validado com ajustes pequenos de DDTree/spec-prefill: budget 2, adaptive on, sem `thinking-ngram`, ratio 0.70/0.95.
2. Testar 27B com prompt de sistema fixando explicitamente imports validos de `sequelize-typescript` e scripts `test`/`build`, mas mantendo prompt de usuario igual.
3. Testar 27B target-only sem `--no-thinking`, mas com `--structured-cot-token-budget` baixo se suportado, para preservar qualidade e reduzir loops.
4. Se performance de decode for prioridade isolada, descartar MTP neste checkpoint local e focar em DFlash/ngram/guard.
