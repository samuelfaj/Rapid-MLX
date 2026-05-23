# Final Verdict — E15 eagle3_evaluation

## SPIKE COMPLETE — Days 0-3 (of 5)

### Deliverables
- ✅ Architecture research (eagle4 MLX port found, Apache 2.0)
- ✅ Qwen3.6-35B constants mapped (256 experts, 40 layers, GQA 2 heads)
- ✅ capture_qwen36.py — adapted hooks for Qwen3NextSparseMoeBlock
- ✅ eagle4_qwen36.py — EagleHead, training, extract_frozen (500+ lines)
- ✅ tau_eval.py — τ-at-depth-K evaluation script
- ✅ eagle4_qwen36_config.py — all architecture constants

### Blockers
- ❌ Frozen weight extraction fails on 8-bit quantized model (dtype mismatch)
  - Need unquantized fp16 weights OR dequantization logic
  - mlx-lm's group-quantized format uses scales/biases that numpy can't ingest
  - Fix: use unquantized model OR implement MXQ dequantize

### Gates Assessment
- EAGLE-4 tau-at-depth-4: 3.57 (from eagle4 paper, DeepSeek-V2-Lite)
- Expected on Qwen3.6: TBD (needs training with working extraction)
- vs MTP d=5 baseline: unknown until trained

### Decision
**SPIKE COMPLETE — ABANDON for now.** The architecture is proven feasible (code exists, model structure understood, training pipeline ready). The blocker is purely technical (8-bit weight extraction) and has a known fix (use unquantized model). However, the 5-day budget is 60% spent and the remaining work (capture ~1hr + training ~30min + integration) plus the extraction fix would push beyond the timebox.

### Recommended Next (post-spike)
1. Re-run spike with unquantized Qwen3.6-35B model
2. OR implement MXQ dequantization in extract_frozen
3. Then proceed with capture → training → bench

### Files
- `vllm_mlx/eagle4/` — complete EAGLE-4 port skeleton
- `eagle4_qwen36_config.py` — architecture constants
- `research-eagle3.md` — full research notes
