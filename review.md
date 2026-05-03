# REVIEW

Status: objetivo 35B concluido; 27B replicado com perfil otimizado validado.

## O que foi implementado

- Speculative prefill pode atuar no primeiro request com tools, mas e desativado depois de `role=tool` para preservar resultados de ferramenta.
- Tool results antigos e grandes agora sao compactados antes do chat template, mantendo os ultimos resultados e preservando falhas/diagnosticos.
- O prompt do chat agora calcula multiplos `prefix_boundaries` seguros a partir dos prefixes reais do template.
- `Request`, `EngineCore` e `Scheduler` carregam `prefix_boundaries`.
- Scheduler ganhou hook nativo para `mlx-lm 0.31+`, salvando cache prompt-only e caches de segmentos mesmo sem a antiga API interna `_process_prompts`.
- Stream de chat ganhou heartbeat SSE com delta vazio enquanto aguarda chunks.
- Retry de stream agora cobre `length` sem tool call.
- `structured-cot-tools` pode atuar desde o primeiro request com tools, mas agora para de forcar continuação quando ja existe evidencia de validacao e artefatos pedidos.
- Agentic guard diagnostica mais cedo e o modo de uma tool-call por resposta fica restrito a repair/missing, porque limitar desde o inicio degradou scaffold.
- Script `scripts/run_opencode_bench3_once.sh` registra runs opencode reproduziveis em `/tmp/rapid-mlx-bench3`.

## O que funcionou

O ganho decisivo veio de prefix-cache segmentado + compaction de tool results. No 35B, baseline sem prefix-cache levou 472.24s; o perfil otimizado levou 207.25s. Ambos finalizaram opencode e passaram `bun test`, entao o speedup fim-a-fim valido foi 2.28x.

O cache realmente entrou no caminho critico: no 35B otimizado houve 24 cache hits, TTFT mediano caiu de 6.178s para 2.973s, e TPS efetivo mediano subiu de 18.96 para 54.44.

O 27B tambem validou no perfil otimizado. O baseline 27B sem prefix-cache bateu timeout e falhou `bun test`; o perfil com prefix-cache, compaction e `structured-cot-tools` curto finalizou opencode em 629s e passou `bun test`.

## O que nao virou perfil vencedor

DFlash/DDTree/n-gram funcionaram tecnicamente quando a flag correta `--drafter` foi usada. A flag antiga `--draft-model` nao ativa DFlash.

No fluxo opencode 35B, DFlash/DDTree/n-gram forcados ficaram lentos: o run com `DFLASH_AGENTIC_TARGET_FALLBACK=0`, DDTree budget 4 e fallback n-gram bateu timeout de 900s e ficou em ~11.3 TPS efetivo mediano. Para este workload, o custo de draft/verify nao compensou o padrao de tool calls, contexto crescente e prefill dominante.

Limitar o agente a uma tool-call por resposta desde o inicio removeu alguns timeouts, mas deixou o scaffold lento e incompleto. A regra ficou limitada a modos de reparo/missing.

## Insights

Agentic coding longo nao e dominado por decode puro. O gargalo principal foi refazer prefill de um historico com muitas tools. Por isso DDTree/DFlash, que atacam decode, perderam para prefix-cache segmentado.

Tool-result compaction foi importante porque o historico cresce rapido. Compactar apenas resultados antigos bem-sucedidos preserva a informacao que o agente precisa para continuar e evita apagar erros que precisam de reparo.

Finalizacao precisa permitir texto final depois de validacao. Antes, `structured-cot-tools` podia transformar uma resposta final text-only em novo retry de tool call. A correcao foi considerar task complete quando ha evidencia de validacao e artefatos pedidos, mesmo sem `--agentic-guard`.

## Evidencia

- Benchmarks resumidos: `BENCHMARK3.md`.
- Artefatos: `/tmp/rapid-mlx-bench3/*.result.json`, `.server.log`, `.opencode.log`, `.validation-install.log`, `.validation-test.log`.
- Validacoes locais do repo:
  - `uv run ruff check vllm_mlx/routes/chat.py vllm_mlx/engine/batched.py vllm_mlx/scheduler.py vllm_mlx/engine_core.py vllm_mlx/request.py tests/test_speculative_prefill.py`
  - `uv run pytest tests/test_speculative_prefill.py`
  - `bash -n scripts/run_opencode_bench3_once.sh`
