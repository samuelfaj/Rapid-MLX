# Verdict — E13 mx_compile_coverage

## Investigation
- deepseek_v4.py: 98 funcs, 9 @mx.compile — but N4 "never touch" protection
- gemma4_text.py: mostly class methods with Python branches, not compilable
- mllm.py: I/O-heavy image/video processing, no mx ops
- No safe pure-mx candidates outside N4-protected code

## Decision
SKIP — no safe targets within ≤30 LOC budget. All pure mx ops are in deepseek_v4.py which is N4-protected.

## Next
E14 — sliding_window_default_long_ctx (Phase 3)
