# Benchmark Rapid-MLX: agente local sem vs com otimizacao

1/ Mesmo prompt, mesma maquina, mesma porta, pasta vazia por run.

> Prompt: `Create the snake game using react, typescript, tailwindcss. Do not run npm run dev or vite dev. Use test driven development. Use react feature oriented architecture.`

2/ O teste nao mede so tokens/s. Mede comportamento agentico: criar projeto React + TypeScript + Tailwind, usar TDD, nao subir dev server, terminar sozinho, depois passar em validacao local.

## Resultado curto

- Qwen3.6 27B UD Q4_K_XL: speedup fim-a-fim = n/a; 0/3 baseline e 0/3 otimizado validaram.
- Qwen3.6 35B A3B 4bit: speedup fim-a-fim = n/a; 0/3 baseline e 0/3 otimizado validaram.

## Tabela

| modelo | perfil | runs | pi finalizou | validou | timeouts | mediana wall | mediana wall valida | tok/s mediana |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B A3B 4bit | baseline | 3 | 1/3 | 0/3 | 2 | 10m00.0s | n/a | 87.8 |
| Qwen3.6 35B A3B 4bit | optimized | 3 | 0/3 | 0/3 | 3 | 10m00.0s | n/a | n/a |
| Qwen3.6 27B UD Q4_K_XL | baseline | 3 | 0/3 | 0/3 | 3 | 10m00.0s | n/a | 4.62 |
| Qwen3.6 27B UD Q4_K_XL | optimized | 3 | 0/3 | 0/3 | 3 | 10m00.0s | n/a | n/a |

## Metodo

- Runs por perfil: `3`.
- Timeout do agente: `600s`.
- Timeout por comando de validacao: `180s`.
- Baseline: target model sem drafter, sem DDTree, sem ngram fallback, sem structured-cot tool guard.
- Otimizado: target model + drafter DFlash pareado, DDTree budget 4, adaptive off, fallback ngram, thinking ngram, structured-cot e structured-cot-tools.
- `pi` usa provider local OpenAI-compatible via `PI_CODING_AGENT_DIR`, `rapid-mlx` em `http://127.0.0.1:8010/v1`, `temperature=0`, `max_tokens=2048`.
- Validacao: `npm install --no-audit --no-fund`, `npm test -- --run` quando Vitest, `npm run build`, e `npm run lint` se existir.
- Tok/s e diagnostico de servidor, nao criterio de sucesso. Runs longos podem trocar entradas antigas do `/v1/requests`; `BENCHMARK.py` agora faz polling para novos reruns.

## Perfis

### qwen36_35b_baseline

- Target: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`.
- Drafter: `none`.
- Server command: `uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit --served-model-name local --port 8010 --default-temperature 0 --max-tokens 2048 --timeout 180`.

### qwen36_35b_optimized

- Target: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`.
- Drafter: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash`.
- Server command: `uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash --dflash-ddtree-budget 4 --dflash-no-adaptive --dflash-fallback-mode ngram --thinking-ngram --ngram-num-draft-tokens 4 --ngram-size 2 --ngram-min-matches 1 --structured-cot --structured-cot-tools --served-model-name local --port 8010 --default-temperature 0 --max-tokens 2048 --timeout 180`.

### qwen36_27b_baseline

- Target: `/Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx`.
- Drafter: `none`.
- Server command: `uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx --served-model-name local --port 8010 --default-temperature 0 --max-tokens 2048 --timeout 180`.

### qwen36_27b_optimized

- Target: `/Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx`.
- Drafter: `/Users/samuelfajreldines/dev/models/Qwen3.6-27B-DFlash`.
- Server command: `uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-27B-UD-Q4_K_XL-mlx --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-27B-DFlash --dflash-ddtree-budget 4 --dflash-no-adaptive --dflash-fallback-mode ngram --thinking-ngram --ngram-num-draft-tokens 4 --ngram-size 2 --ngram-min-matches 1 --structured-cot --structured-cot-tools --served-model-name local --port 8010 --default-temperature 0 --max-tokens 2048 --timeout 180`.

## Leitura para X

3/ Otimizacao que importa aqui nao e micro-benchmark isolado. E fim-a-fim: se agente para cedo, entra em loop, ou gera projeto quebrado, velocidade de decode nao salva run.

4/ Resultado bom = run termina, projeto nasce em pasta limpa, testes passam, build passa. Resultado ruim = timeout, erro de tool-call, pacote incompleto, ou validacao quebrada.

5/ Conclusao: nenhum perfil atingiu sucesso completo. Neste teste, as otimizacoes nao produziram ganho fim-a-fim confiavel; throughput parcial nao basta se agente nao termina e valida.

6/ Artefatos brutos: `/tmp/rapid-mlx-bench`. JSON completo: `/tmp/rapid-mlx-bench/results.json`.
