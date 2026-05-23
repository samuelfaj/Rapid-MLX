# Verdict — E3 kv_turboquant_default_on

## Status
BLOCKED — turboquant is serve-only, drift gate can't be run via bench.

## Reason
--kv-cache-turboquant only exists on serve_parser, not bench_parser. The mandatory drift gate (G4) requires comparing logit outputs with/without turboquant, which can't be done via bench mode. Serve-mode testing would require a different evaluation harness.

## Next
E4 — prefix_cache_size_tune (testable via bench)
