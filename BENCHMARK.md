# Benchmark Rapid-MLX: agente local sem vs com otimizacao

1/ Mesmo prompt, mesma maquina, mesma porta, pasta vazia por run.

> Prompt: `create a REST api using express and bun and typescript and sequelize-typescript. It must be vertical sliced. You
should create models, seeders and migrations. You must create unit tests for each service.`

2/ O teste nao mede so tokens/s. Mede comportamento agentico: criar o artefato pedido em pasta limpa, terminar sozinho, depois passar em validacao local.

## Resultado curto

- Qwen3.6 35B A3B 4bit: speedup fim-a-fim = n/a; 0/1 baseline e 0/1 otimizado validaram; throughput de decode 0.41x maior.

## Tabela

| modelo | perfil | runs | pi finalizou | validou | timeouts | mediana wall | mediana wall valida | tok/s mediana |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B A3B 4bit | baseline | 1 | 1/1 | 0/1 | 0 | 3m18.2s | n/a | 113.11 |
| Qwen3.6 35B A3B 4bit | optimized | 1 | 0/1 | 0/1 | 1 | 10m00.2s | n/a | 46.2 |

## Metodo

- Runs por perfil: `1`.
- Timeout do agente: `600s`.
- Timeout por comando de validacao: `180s`.
- Baseline: target model sem drafter, sem DDTree, sem ngram fallback, sem structured-cot tool guard.
- Otimizado: target model + drafter DFlash pareado, Speculative Prefill conservador com draft pequeno, DDTree budget 4, adaptive off, fallback ngram, thinking ngram, structured-cot e structured-cot-tools.
- `pi` usa provider local OpenAI-compatible via `PI_CODING_AGENT_DIR`, `rapid-mlx` em `http://127.0.0.1:8010/v1`, `temperature=0`, `max_tokens=4096`.
- Validacao: instala dependencias com package manager detectado, roda `test` quando existir ou for pedido, roda `build`/`lint` quando existirem.
- Tok/s e diagnostico de servidor, nao criterio de sucesso. Runs longos podem trocar entradas antigas do `/v1/requests`; `BENCHMARK.py` agora faz polling para novos reruns.

## Perfis

### qwen36_35b_baseline

- Target: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`.
- Drafter: `none`.
- Server command: `uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit --served-model-name local --port 8010 --default-temperature 0 --enable-auto-tool-choice --tool-call-parser qwen3_coder_xml --max-tokens 4096 --timeout 300`.

### qwen36_35b_optimized

- Target: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`.
- Drafter: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash`.
- Server command: `uv run rapid-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit --drafter /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash --dflash-ddtree-budget 4 --dflash-no-adaptive --dflash-fallback-mode ngram --thinking-ngram --ngram-num-draft-tokens 4 --ngram-size 2 --ngram-min-matches 1 --structured-cot-tools --agentic-guard --pin-system-prompt --speculative-prefill --speculative-prefill-draft-model /Users/samuelfajreldines/dev/models/Qwen3-1.7B-4bit-mlx --speculative-prefill-ratio 0.85 --served-model-name local --port 8010 --default-temperature 0 --enable-auto-tool-choice --tool-call-parser qwen3_coder_xml --max-tokens 4096 --timeout 300`.

## Leitura para X

3/ Otimizacao que importa aqui nao e micro-benchmark isolado. E fim-a-fim: se agente para cedo, entra em loop, ou gera projeto quebrado, velocidade de decode nao salva run.

4/ Resultado bom = run termina, projeto nasce em pasta limpa, testes passam, build passa. Resultado ruim = timeout, erro de tool-call, pacote incompleto, ou validacao quebrada.

5/ Conclusao: nenhum perfil otimizado atingiu sucesso completo. Neste teste, throughput parcial nao basta se agente nao termina e valida.

6/ Artefatos brutos: `/tmp/rapid-mlx-bench`. JSON completo: `/tmp/rapid-mlx-bench/results.json`.
