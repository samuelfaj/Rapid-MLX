# Verdict — E12 sampler_gpu_path

## Investigation
- Per-token loop `.tolist()` at scheduler.py:320 — batch_size=1 agentic workloads, negligible cost
- Speculative decode `.tolist()` at scheduler.py:960-961 — 1-5 elements, negligible
- All CPU syncs are on small tensors in batch=1 agentic path

## Gates
- G1: no benchable change — SKIP
- G6: N/A (no code change)

## Decision
SKIP — low impact for agentic workloads. Optimization would benefit batch>1 only, which is not the agentic target. Preserving ≤50 LOC budget for higher-value changes.

## Next
E13 — mx_compile_coverage
