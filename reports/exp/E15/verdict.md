# Verdict — E15 eagle3_evaluation (SPIKE PHASE 1: Feasibility)

## Research Findings

### MLX Port Exists
- **eagle4** (joshuahickscorp/eagle4): Apache 2.0, ~500 lines MLX, MoE-aware
- Already runs on Apple Silicon (DeepSeek-V2-Lite-Chat)
- tau-at-depth-4: **3.57** (vs EAGLE-3 baseline 2.15 = +66%)
- Depth-4 full acceptance: **83.6%**
- Training: ~21 min on M3 Pro, 1M records (M5 Max: ~15 min)
- Head size: ~60M params, Q4 quantized to 46 MB

### Comparison vs Current MTP (d=5)
| Metric | MTP d=5 | EAGLE-4 (est.) |
|--------|---------|---------------|
| Depth-1 acceptance | ~95% | ~95% |
| tau-at-depth-4 | Unknown | ~3.57 |
| Head params | Built-in (5 heads) | 60M separate |
| Training needed | No | Yes (~15 min M5) |
| MoE-aware | No | Yes (mask head) |
| MLX code | mlx-lm MTP | eagle4 (pure MLX) |

### Port Plan (5-day spike)
1. **Day 1-2**: Fork eagle4, adapt constants + capture hooks for Qwen3.6-35B-A3B
2. **Day 2-3**: Run capture (Mac, ~1 hr for 1M records)  
3. **Day 3-4**: Train on RunPod (pure MLX, ~15 min) per USER DIRECTIVE
4. **Day 4-5**: Integrate into lightning-mlx scheduler, bench vs MTP

### Key Risk
MTP's 95% depth-1 acceptance with d=5 is already strong. EAGLE-4 must achieve higher effective throughput at equivalent compute. The mask head's 17-21% top-8 recall may limit MoE prefetch gains.

## Decision
PROCEED — spike is time-boxed. If tau-at-depth-4 < 3.0 or wall-clock tps < MTP baseline, abandon and delete branch.

## Next
Start Day 1: fork eagle4, begin Qwen3.6-35B adaptation
