# Verdict — E9 nax_runtime_detect

## Snapshot
- Code-only — pure observability. No bench impact.
- device=Apple M5 Max, macOS=26.3, metal4=True, m5_nax=True

## Gates
- G1: N/A (no code path changes for inference)
- G2: N/A
- G3: SKIP
- G4: N/A (no numerics)
- G5: N/A
- G6: +19 LOC — PASS

## Decision
KEEP

## Reason
Non-invasive capability detection. Logs Metal4/NAX status at server startup. Gated behind macOS ≥26.2 and LIGHTNING_DISABLE_METAL4 env var. Purely informational.

## Next
E10 — mlx_floor_bump (mlx>=0.30.0)
