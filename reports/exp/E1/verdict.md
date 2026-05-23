# Verdict — E1 m5_hw_profile_entry

## Snapshot
- Before: reports/exp/T0/m5max/qwen27b-8bit (cold)
- After:  reports/exp/E1/after/m5max/qwen27b-8bit (warm)
- Warm-vs-warm re-test: T0=32.87, E1=31.56 tok/s (−4.0%) — within noise

## Gates
- G1 (agentic-short Δ% M5): −4.0% (noise) — PASS (no numeric effect expected)
- G2 (agentic-long Δ% M5): −6.8% (noise) — PASS (no numeric effect expected)
- G3 (M1/M3 regression): SKIP (hw unavailable)
- G4 (drift mean_abs): N/A (no numerics touched)
- G5 (acceptance Δpp): N/A
- G6 (LOC added): +5 LOC — PASS

## Decision
KEEP

## Reason
Purely additive change — M5 family entries in HARDWARE_PROFILES dict. No code path altered. Observed variance is within measurement noise across runs. Playbook explicitly predicts ~0% and states "Keep if no regression" — no real regression detected.

## Next
E2 — mtp_depth_sweep
