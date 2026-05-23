# Verdict — E4 prefix_cache_size_tune

## Snapshot
- Cache disabled (T0 default): 160.42 tok/s
- Cache=4096: 151.87 tok/s (−5.3%)
- Sweep: 4096=174.73, 8192=174.41, 16384=149.79, 32768=139.78
- Second-run never improved over first (no cache hit benefit for short prompts)

## Gates
- G1 (agentic-short Δ% M5): −5.3% — FAIL
- G2: N/A
- G3: SKIP
- G4: N/A
- G5: N/A
- G6: N/A

## Decision
REVERT — prefix cache doesn't help agentic_short/tool_heavy workloads. Cache overhead dominates for prompts <100 tokens.

## Reason
Short agentic prompts (21-84 tokens) get zero cache hit benefit. Second-run throughput is equal or worse than first-run. The cache management overhead costs 5-15% throughput. Current approach of `--disable-prefix-cache` in sweep script is correct for agentic workloads.

## Next
E5 — chunked_prefill_default_on
