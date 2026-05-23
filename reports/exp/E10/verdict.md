# Verdict — E10 mlx_floor_bump

## Snapshot
- Before (mlx>=0.29.0, running 0.31.2): 167.51 tok/s
- After (mlx>=0.30.0, still 0.31.2): 164.99 tok/s (−1.5%, within noise)

## Gates
- G1: −1.5% (noise) — PASS
- G2: N/A (no long-term impact expected)
- G3: SKIP
- G4: N/A (no numerics changed — same mlx version)
- G5: N/A
- G6: 1 LOC — PASS

## Decision
KEEP

## Reason
Floor bump from 0.29.0→0.30.0 formalizes minimum MLX version. No actual version change (still 0.31.2). Enables safe use of MLX 0.30+ features like NAX.

## Next
E11 — dflash_sweep
