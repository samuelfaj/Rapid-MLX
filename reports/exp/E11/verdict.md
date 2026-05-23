# Verdict — E11 dflash_sweep

## Status
BLOCKED — DFlash has no bench-parser flags. Only surfaced in TUI status panel. The sweep requires `--dflash-block-size` or similar, which doesn't exist in current code.

## Reason
DFlash drafter exists (`vllm_mlx/speculative/dflash_drafter.py`) with block_size=16 default, but no CLI flag to override it in bench mode. Can't run the adaptive bounds sweep without adding bench parser support first.

## Next
E12 — sampler_gpu_path
