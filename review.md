# Review: oportunidades aprendidas com Lucebox DFlash

## Resumo

Rapid-MLX ja faz a arquitetura geral: DFlash + DDTree, motor dedicado, CLI com `--drafter`, tree verify em MLX/Metal, n-gram fallback e metricas. Nada de Lucebox parece plug-and-play, porque Lucebox e C++/CUDA/ggml/GGUF/NVIDIA, enquanto Rapid-MLX e Python/MLX/Metal/Apple Silicon.

Mesmo assim, Lucebox mostra oportunidades reais para Rapid-MLX: reduzir sync CPU, garantir chain pre-seed, baratear commit/rollback, reavaliar budget depois de otimizar verify, medir com benchmark limpo AR vs DDTree, e tratar n-gram como modo especifico de workload.

## O que Lucebox fez

- DFlash + DDTree para Qwen3.5-27B GGUF Q4_K_M com draft BF16.
- Custom ops CUDA/ggml para SSM/tree:
  - `ggml_ssm_conv_tree`
  - `ggml_gated_delta_net_tree`
  - `ggml_gated_delta_net_tree_persist`
- Tree verify com budget 22, rollback rapido, KV quantizado e target feature ring.
- Resultado reportado:
  - HumanEval: 37.78 -> 129.52 tok/s
  - Math500: 37.71 -> 110.51 tok/s
  - GSM8K: 37.65 -> 96.15 tok/s

## O que Rapid-MLX ja tem

- Motor DFlash em `vllm_mlx/engine/dflash.py`.
- DDTree MLX em `vllm_mlx/speculative/ddtree/engine.py`.
- Verify tree-aware e kernels Metal em:
  - `vllm_mlx/speculative/ddtree/verify.py`
  - `vllm_mlx/speculative/ddtree/kernels.py`
- CLI/docs com:
  - `--drafter`
  - `--dflash-ddtree-budget`
  - `--dflash-fallback-mode ngram`
- Bench local ja medido:
  - Prose TCP/UDP pure DDTree budget 4: 23.8 tok/s
  - CRUD pure DDTree budget 4: 29.8 tok/s
  - CRUD n-gram size 2 draft 4 + DDTree: 35.2 tok/s

## Oportunidades concretas

### 1. Garantir chain pre-seed

Lucebox aponta que chain pre-seed no `build_ddtree` foi critico. Sem isso, a arvore ficava "bushy" e rasa sob ruido de quantizacao, com acceptance length perto de 4. Com seed da chain top-1, acceptance length voltou perto de 9.

Acao possivel: auditar builder Rapid-MLX para confirmar se sempre preserva cadeia greedy top-1 antes de gastar budget em siblings. Se nao preservar, implementar e medir.

### 2. Tirar CPU sync do loop DDTree

Lucebox ganhou removendo roundtrip GPU->CPU->GPU em `target_feat`. Rapid-MLX ainda tem sinais de custo Python/CPU:

- `.tolist()` em prompt/posterior/DFS
- tree build via NumPy
- posterior tree walk em Python
- concat/extração de hidden states em Python

Acao possivel: reduzir `.tolist()`, mover top-k/tree build/walk para MLX/Metal quando viavel, ou cachear estruturas de arvore por shape/topology.

### 3. Medir se kernels Metal realmente removem custo

Rapid-MLX ja tem kernels Metal para conv1d tree e gated delta tree. Lucebox mostra que o grande ganho veio quando kernel persistente escreveu intermediarios diretamente e evitou copias.

Acao possivel: criar perfil por fase:

- draft
- tree_build
- tree_verify_attention
- tree_verify_linear
- commit
- hidden extraction

Objetivo: ver se gargalo atual e kernel, sync, commit ou Python.

### 4. Re-varrer budget depois de reduzir overhead

Hoje budget 6 ficou pior que budget 4 no Rapid-MLX. Lucebox mostra que budget maior so compensa quando verify/commit esta barato. Antes disso, budget maior apenas aumenta custo.

Acao possivel: primeiro otimizar overhead; depois repetir sweep budget 4/6/8/12/16/22 em prompts Luce-style.

### 5. Usar feature ring / sliding hidden cache

Lucebox usa target feature ring de 4096 slots para manter memoria fixa. Rapid-MLX tem caminhos que concatenam/guardam hidden chunks para prompt cache extendido.

Acao possivel: estudar ring buffer para features do drafter e evitar concat/copy em long context.

### 6. Separar benchmark decode puro de benchmark agente

Lucebox mede AR vs DFlash no mesmo harness, concurrency 1, greedy, `n_gen=256`. Rapid-MLX tem numeros de TUI/opencode/tool-calls que misturam prefill, polling, comandos, testes e overhead de agente.

Acao possivel: criar bench equivalente:

- mesmos prompts/datasets
- AR target-only
- DFlash chain
- DDTree
- DDTree + n-gram
- `temperature=0`
- non-streaming
- wall time com `usage.completion_tokens`

### 7. Tratar n-gram como especializacao

Rapid-MLX viu n-gram piorar prose e melhorar CRUD repetitivo. Lucebox nao usa n-gram como principal. Logo n-gram deve ser opcional por workload, nao default geral.

Acao possivel:

- pure DDTree default para prose, reasoning, tool planning
- n-gram first apenas para boilerplate, CRUD, JSON/XML, testes repetitivos
- detector/heuristica baseada em repeticao real, nao so draft length

### 8. Comparar Qwen3.5 e Qwen3.6 separadamente

Lucebox reporta bons ganhos em Qwen3.5-27B. Rapid-MLX testa muito Qwen3.6-27B/35B. Se Qwen3.6 draft tiver menor acceptance, o motor parece pior do que e.

Acao possivel: medir Qwen3.5 target/drafter e Qwen3.6 target/drafter separadamente, sem misturar conclusoes.

## Veredito

Nao existe ganho imediato que pareca apenas ativar flag. Mas existem oportunidades claras aprendidas com Lucebox:

1. confirmar/fortalecer chain pre-seed;
2. remover sync CPU no loop DDTree;
3. baratear commit/rollback e hidden feature update;
4. re-varrer budget apos otimizar overhead;
5. criar benchmark AR-vs-DDTree limpo;
6. usar n-gram so onde workload favorece;
7. comparar familias Qwen separadamente.

Meta realista para Rapid-MLX: nao reproduzir exatamente 129 tok/s CUDA, mas provar speedup AR -> DDTree no Apple Silicon e reduzir overhead ate DDTree vencer baseline de modo consistente.
