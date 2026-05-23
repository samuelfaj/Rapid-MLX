"""EAGLE-4 head port for Qwen3.6-35B-A3B (MoE) on MLX.

Adapted from joshuahickscorp/eagle4 (Apache 2.0).
Original target: DeepSeek-V2-Lite-Chat.
This port targets: Qwen3.6-35B-A3B (architecture: Qwen3_5MoeForCausalLM).

Key architectural differences from DeepSeek-V2-Lite:
- Experts: 256 (was 64) — mask head grows from 1664→10240 outputs
- Layers: 40 (was 27) — capture hooks on different layer indices
- Vocab: 248320 (was 102400)
- GQA: 2 KV heads (was... unknown, V2-Lite has query groups)
- num_experts_per_tok: 8 (was 6)
- head_dim: 256

Day 1 deliverable: constants + layer mapping + capture hook plan.
Day 2: capture code adaptation.
Day 3: training on RunPod.
Day 4: scheduler integration.
Day 5: bench vs MTP baseline, accept/reject.
"""

# ---------------------------------------------------------------------------
# Qwen3.6-35B Constants
# ---------------------------------------------------------------------------
HIDDEN_DIM = 2048          # Same as V2-Lite
VOCAB = 248320             # V2-Lite was 102400
N_MOE_LAYERS = 40          # V2-Lite was 26 (all layers are MoE in Qwen3.6)
N_ROUTED = 256             # V2-Lite was 64 — 4x increase!
TOP_K = 8                  # V2-Lite was 6
N_HEADS = 16               # Same
N_KV_HEADS = 2             # GQA
HEAD_DIM = 256             # V2-Lite had 128
INTERMEDIATE = 512         # MoE intermediate (per expert), V2-Lite had 5632
RMS_EPS = 1e-6             # Same
MAX_POS = 262144           # Rope max position
ROPE_THETA = 1000000.0     # Default for Qwen3

# ---------------------------------------------------------------------------
# Fusion layer selection (capture 3 layers from different depths)
# Qwen3.6 has 40 layers; pick low/mid/high spread:
#   low = layer 3  (10%)
#   mid = layer 20 (50%)
#   high = layer 38 (95%, second-to-last before final norm)
# ---------------------------------------------------------------------------
FUSION_LAYERS = (3, 20, 38)

# ---------------------------------------------------------------------------
# Training parameters (from eagle4, tuned for Qwen3.6)
# ---------------------------------------------------------------------------
LR = 3e-4
BATCH_SIZE = 32
SEQ_LEN = 16               # Training sequence window
WARMUP_STEPS = 500         # CE ramp from corpus→target argmax
ALPHA_INIT = 0.05          # Residual gate init
N_RECORDS = 1_000_000      # Training examples
SHARD_ROWS = 8192          # Parquet shard size
