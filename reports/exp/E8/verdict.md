# Verdict — E8 ngram_K_per_workload

## Snapshot
- Code-only change — no bench impact (default ngram_num_draft_tokens already 6)
- New: per-profile engine.ngram_max_k with cli.py override

## Gates
- G1: N/A (default unchanged — both were 6)
- G2: N/A
- G3: SKIP
- G4: N/A (no numerics)
- G5: N/A
- G6: +28 LOC — PASS

## Decision
KEEP — infrastructure plumbing for per-agent ngram tuning

## Reason
Added engine.ngram_max_k to codex/openclaude/cline profiles and wire-up in cli.py _resolve_agent_shortcut_args. Current value (6) matches existing default, so no behavioral change. Enables future per-profile differentiation.

## Next
E9 — nax_runtime_detect (Metal4 capability detection)
