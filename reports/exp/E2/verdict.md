# Verdict — E2 mtp_depth_sweep

## Snapshot
- Before (d=3, warm): 151.82 tok/s short / 145.36 tok/s long
- After (d=5): 167.51 tok/s short / 177.07 tok/s long
- Warm-vs-warm comparison: short +10.3%, long +21.8%

## Gates
- G1 (agentic-short Δ% M5): +10.3% — PASS (≥+5%)
- G2 (agentic-long Δ% M5): +21.8% — PASS (≥0%)
- G3 (M1/M3 regression): SKIP (hw unavailable)
- G4 (drift mean_abs): N/A (no numerics changed — purely config)
- G5 (acceptance Δpp): N/A
- G6 (LOC added): 1 LOC — PASS

## Decision
KEEP

## Reason
MTP draft depth 5 outperforms default 3 by +10.3% on short agentic turns and +21.8% on long artifact generation. Single-line change in bench preset. No numerics touched.

## Next
E3 — kv_turboquant_default_on
