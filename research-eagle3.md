# EAGLE-3 / EAGLE-4 Speculative Decoding Research Summary

**Date**: 2026-05-23
**Target**: MLX port for Qwen3.6-35B (MoE) on Apple Silicon (M5 Max)

---

## 1. Key Links

| Resource | URL |
|---|---|
| EAGLE-3 Paper (NeurIPS 2025) | https://arxiv.org/abs/2503.01840 |
| EAGLE-1 Paper (ICML 2024) | https://arxiv.org/abs/2401.15077 |
| EAGLE-2 Paper (EMNLP 2024) | https://arxiv.org/abs/2406.16858 |
| Official Repo (PyTorch) | https://github.com/SafeAILab/EAGLE |
| EAGLE-4 MLX Repo (Apple Silicon) | https://github.com/joshuahickscorp/eagle4 |
| EAGLE-3 Weights for Qwen3-30B-A3B | https://huggingface.co/Tengyunw/qwen3_30b_moe_eagle3 |
| EAGLE-3 Weights for Qwen3-30B-A3B (alt) | https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3 |
| SpecForge (SGLang training) | https://github.com/sgl-project/SpecForge |

---

## 2. Architecture Summary

### 2.1 EAGLE-3 Core Design

EAGLE-3 abandons the feature-prediction constraint from EAGLE-1/2. Key innovations:

**Input**: Fusion of 3 hidden state vectors from different layers of the TARGET model:
- Low-layer features (embedding + early layer output)
- Mid-layer features
- High-layer features (second-to-last layer, before final norm)

**Forward pass** (single transformer decoder layer):
x = Linear(3 * hidden_size -> hidden_size)(concat(h_low, h_mid, h_high))
x = DecoderLayer(x, inputs_embeds, attention_mask)  # ~1 transformer block
x = RMSNorm(x)
logits = lm_head(x)  # frozen, shared with target model (or reduced vocab)

**Draft vocab**: Uses a reduced vocabulary (~tens of thousands of tokens) to make the head smaller. d2t/t2d mapping converts between draft and target token IDs.

**Parameters**: ~50-100M params (single decoder layer + input projection).

**Training-time test**: The key EAGLE-3 innovation. The draft model is rolled out autoregressively for length=7 steps during training, with each step feeding the previous step's argmax as input. Loss is teacher-forced from the target model's argmax distribution. This teaches recovery from draft errors.

### 2.2 EAGLE-3 vs MTP (Medusa-like Tree Prediction)

| Aspect | MTP (Qwen-style) | EAGLE-3 |
|---|---|---|
| Input | Hidden states + previous token embedding | 3-layer feature fusion |
| Architecture | Multiple parallel heads (d=5) | Single decoder layer, autoregressive rollout |
| Tree structure | Fixed tree, top-k per step | Dynamic sparse tree (top-k beam search) |
| Training | Joint CE on target tokens | Training-time test (7-step rollout) |
| Feature constraint | Predicts features then decodes | Direct token prediction |
| Parameter count | ~5 independent heads | 1 shared transformer block |
| Acceptance rate | ~95% at depth-1 | ~73-75% at depth-1 |

### 2.3 EAGLE-4 (Eagle4 MLX Repo) - MoE-Aware Extension

The joshuahickscorp/eagle4 repo is an MLX-native implementation (~500 lines) with critical MoE innovations:

**Architecture differences from EAGLE-3**:
1. **5-input fusion** (vs 3): adds prev_token_embedding + shared_expert_hidden
2. **2 additional output heads**:
   - **Mask head**: Predicts which experts activate per layer (26 layers x 64 experts)
   - **Calibration head**: Per-token P(accept) scalar for runtime confidence
3. **Residual gate**: draft_hidden = post_norm(h_high) + alpha * block(...) with alpha=0.05 init
4. **Diagonal attention**: Each position processed independently

**Results on DeepSeek-V2-Lite-Chat Q4_K_M**:
- tau-at-depth-4 (mean accepted prefix length): **3.57** vs EAGLE-3 baseline 2.15 (+66%)
- Single-step target-argmax acceptance: **95.3%** vs 75.8%
- Depth-4 full acceptance: **83.6%** vs 37.4%
- Mask head top-8 recall: **17-21%** (enables expert prefetching)
- Head size: ~60M params, Q4 quantized to **46 MB**
- Training: ~21 min on M3 Pro, 1M records, 1 epoch

---

## 3. Training Requirements

### EAGLE-3 Official Training
- **Data**: ShareGPT/chat data, filtered for target language
- **Hardware**: 8x RTX 3090 (1-2 days), trainable on 2 GPUs
- **Process**: 
  1. Generate hidden states from frozen target model (~70K records)
  2. Compute draft vocab mapping from top-N tokens
  3. Train draft model with training-time test (7-step rollout)
  4. Loss: CE against target argmax distribution at each rollout step
- **SpecForge** (SGLang): Recommended for production training, supports Qwen3

### EAGLE-4 Training (MLX, Apple Silicon)
- **Data**: 1M records from frozen model captures (~1 hr on M3 Pro for capture)
- **Training**: ~21 min on M3 Pro, batch_size=32, seq_len=16, lr=3e-4
- **Loss**: Hybrid CE ramp (corpus CE -> target argmax CE over 500 warmup steps) + MSE (identity pull) + BCE (mask) + BCE (calibration)
- **Key trick**: First 3 positions of each sequence excluded (BOS-adjacent tokens mess up L2 norm)
- **Multi-step training (k=2)**: Tested, does not help (~no difference from k=1)
- **Quantization**: Q4 group_size=64 shrinks head from 297 MB (bf16) to 46 MB, 99.9% argmax parity

---

## 4. Inference Algorithm

### EAGLE-3 Inference Pipeline (from official code)
1. PREFILL: Run target model on input_ids -> hidden_states, KV cache
2. TREE DRAFT (draft model forward, repeated depth=7 times):
   a. Fuse low/mid/high features from target hidden states
   b. Pass through single transformer layer -> logits
   c. Top-k beam search: keep top-k candidates, expand tree
   d. Feed argmax token as next input_ids
3. VERIFY (target model forward on tree):
   a. Run target model on draft tree with tree attention mask
   b. Get logits for all positions
   c. Speculative verification: accept tokens where draft matches target
4. UPDATE: Shift accepted tokens, prepare next iteration

### Key hyperparameters
- depth: 7 (draft tree depth)
- top_k: 10 (beam width per step)
- total_tokens: 60 (total draft candidates per iteration)
- threshold: 1.0 (acceptance threshold)

### Speedups reported
- EAGLE-3: **5.6x** over vanilla decoding (Vicuna-13B)
- EAGLE-3: **1.8x** over EAGLE-1
- EAGLE-3 in SGLang: **1.38x** throughput at batch_size=64

---

## 5. Existing MLX / Apple Silicon Ports

### eagle4 (joshuahickscorp/eagle4)
- **Status**: Complete implementation (~500 lines MLX)
- **Target**: DeepSeek-V2-Lite-Chat (MoE) on Apple Silicon
- **Features**: Full training pipeline, Q4 quantization, tau-at-depth-K eval
- **Missing**: Wall-clock tps measurement (runtime in separate dismantle repo)
- **License**: Apache 2.0
- **Dependencies**: mlx, mlx.nn, mlx.optimizers, pyarrow, numpy

### No other known MLX EAGLE ports
- GitHub search for "eagle mlx speculative decoding apple" returns only eagle4
- No mlx-community EAGLE models on HuggingFace
- vLLM has EAGLE support (CUDA only)
- SGLang has EAGLE support (CUDA/ROCm)
- MLC-LLM has EAGLE support (cross-platform, including Metal)

---

## 6. Feasibility Assessment for Qwen3.6-35B-A3B (MoE)

### Strong case for porting EAGLE-3/4 to MLX

**Pros**:
1. eagle4 repo is a near-direct starting point: Already MLX, MoE-aware, handles rms_norm, SwiGLU, GQA
2. Qwen3.6-35B is architecturally similar to DeepSeek-V2-Lite: Both MoE, both SwiGLU
3. Existing EAGLE-3 weights for Qwen3-30B-A3B on HuggingFace can serve as reference
4. Training is fast on Apple Silicon: ~21 min for 1M records on M3 Pro; M5 Max should be faster
5. Mask head (EAGLE-4) is uniquely valuable for MoE: Expert prefetching turns memory-bound verify into compute-bound
6. Q4 quantization works: 6.4x size reduction at 99.9% argmax parity

**Challenges**:
1. Qwen3.6-35B has ~35B total params (3B active): KV cache is large; runtime needs careful memory mgmt
2. No wall-clock tps data for eagle4: Only offline acceptance metrics; runtime integration needed
3. Existing MTP (d=5, ~95% acceptance) sets a high bar for speedup improvement
4. Mask head recall is modest (17-21%): May limit prefetch benefits on Qwen's expert layout
5. Feature capture requires monkey-patching the target model

### Comparison: EAGLE-3/4 vs MTP for Qwen3.6-35B

| Metric | MTP (d=5) | EAGLE-3 | EAGLE-4 (est.) |
|---|---|---|---|
| Depth-1 acceptance | ~95% | ~75% | ~95% |
| tau-at-depth-4 | Unknown | ~2.15 | ~3.57 |
| Head params | 5 heads | 1 block | 1 block + mask |
| Training needed | Built into model | Separate training | Separate training |
| MoE-aware | No | No | Yes (mask head) |
| MLX availability | Via mlx-lm | None | eagle4 repo |

### Recommendation

The EAGLE-4 approach (from eagle4 repo) is the most promising path:
1. Start with the eagle4 codebase as reference
2. Adapt for Qwen3.6-35B: change hidden dim, num layers, MoE config, capture hooks
3. Train on Qwen3.6's own feature captures (1M records, ~1 hr capture, ~30 min train on M5 Max)
4. Compare tau-at-depth-4 against existing MTP at equivalent batch sizes
5. The mask head could enable expert prefetching that MTP cannot match on MoE

**Bottom line**: EAGLE-4's tau=3.57 at depth-4 acceptance of 83.6% suggests it could beat MTP's effective throughput IF the runtime overhead is lower than MTP's 5-head parallel decode. The key advantage is that EAGLE uses 1 shared transformer block vs MTP's 5 independent heads, and the mask+calibration heads add MoE-aware optimization that MTP lacks entirely.
