# NAX / Metal4 Tensor API — Implementation Plan

**Status:** PLANNING
**Owner:** unassigned
**Created:** 2026-05-23
**Source motivation:** antirez/ds4 reports 40–80% prefill speedup on M5 Max via Metal4 Neural Accelerators (NAX). Lightning-MLX should capture the same wins.

---

## 0. Glossary

- **NAX** — per-GPU-core Neural Accelerators on Apple M5+ silicon. Matmul tensor units in the GPU. Distinct from ANE.
- **Metal4 Tensor API** — `<metal_tensor>` + `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` (MPP). Exposes `matmul2d`, cooperative tensors, `execution_simdgroups<N>`.
- **MTPLX** — this repo's MTP + n-gram speculation stack (`vllm_mlx/speculative/`).
- **DS4_METAL_HAS_TENSOR** — antirez's compile-time gate symbol.

---

## 1. Goal

Match antirez/ds4 prefill gains on M5+ hardware (target: prefill +30–50%, gen +10–20%) **without regressing M1–M4** and without breaking MTPLX/MTP/n-gram acceptance.

Pass criteria:
- P1: prefill t/s @ 1k / 10k / 50k / 100k ctx ≥ +20% on M5 Max vs current baseline.
- P2: generation t/s ≥ +5% on M5 Max vs baseline.
- P3: zero regression on M1–M4 (within ±2% noise band).
- P4: logit drift < 1e-3 mean abs at 4k / 16k / 64k ctx vs legacy kernel path.
- P5: MTP/n-gram acceptance rate stable (within ±1pp).

---

## 2. How antirez/ds4 did it (reference)

### Detection (`ds4_metal.m`)

```objc
g_metal4_family_supported = !metal4_disabled
    && [g_device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
g_metal4_queue_supported = [g_device respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0;
if (g_metal4_family_supported && ds4_gpu_device_name_contains("M5")) {
    g_metal4_m5_neural_accelerators_hint = 1;
}
```

Plus runtime probe: compile a test kernel; on failure, log `"Metal 4 tensor API probe failed; using legacy Metal kernels"` and fall back.

### Kernels (`metal/dense.metal` 578 LOC, `metal/moe.metal` 753 LOC)

Guard: `#ifdef DS4_METAL_HAS_TENSOR`.

Includes:
```cpp
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
```

APIs used:
- `matmul2d` + `matmul2d_descriptor`
- `tensor()` with `dextents<int32_t,2>`
- `execution_simdgroups<4>`
- `get_destination_cooperative_tensor`
- `simdgroup_load` / `simdgroup_store` / `simdgroup_multiply_accumulate`

Tile params:
- 64×32 output tile, NK=32 inner, NL0=2 / NL1=4 loader splits
- Direct-RHS variants: NR0=64 fixed, NR1 ∈ {32, 64, 128}

Strategy:
- Dense GEMM: dequant weight → fp16 staging tile in tg mem → cooperative `matmul2d` on 4 simdgroups → store.
- MoE: pre-computed expert-major index map ⇒ cooperative matmul stays dense per tile.
- Quant matvec (decode): **stays legacy** (memory-bound, tensor doesn't help).

### Validation harness (cleanup commit dropped 11K LOC)

- Logit drift gate (`compare_logit_drift.py`)
- Chunked prefill drift gate (`run_chunked_prefill_drift_gate.py`)
- MPP compare probe (`run_mpp_compare_probe.py`)
- Quality drift gate (`run_quality_drift_gate.py`)

---

## 3. Current Lightning-MLX state

- `pyproject.toml`: `mlx>=0.29.0` ; lockfile resolves `mlx==0.31.2`.
- MLX 0.30.0 release notes: **"Support for Neural Accelerators on M5 (macOS >= 26.2)"** — already upstream.
- MLX 0.30.4: faster fused GQA for long ctx (Metal).
- Custom kernel sites (this repo, only ones that touch Metal directly):
  - `vllm_mlx/models/deepseek_v4.py:420` — `_hc_split_sinkhorn_kernel`
  - `vllm_mlx/models/deepseek_v4.py:606` — `_hc_sinkhorn_collapse_kernel`
  - Both = HyperConnection fused ops, NOT GEMM. Tensor API gives little here.
- SDPA / GEMM / RoPE / RMSNorm go through `mx.fast.*` → upstream MLX path → NAX-aware automatically when MLX picks it.

---

## 4. Metrics framework — before/after

**Mandatory**: every tier captures a fresh `BEFORE` snapshot, applies changes, captures `AFTER`, and writes a comparison row. No tier ships without paired data.

### 4.1 Snapshot capture procedure

For each (model, ctx, mode) cell, record into `reports/nax/<tier>/<timestamp>.json`:

- `prefill_tok_s` (median of N=3 runs)
- `gen_tok_s` (median of N=3 runs)
- `first_token_ms` (TTFT)
- `peak_rss_mb`
- `peak_gpu_mem_mb` (via `mx.metal.get_peak_memory()`)
- `mtp_accept_rate` (when MTP on)
- `ngram_accept_rate` (when n-gram on)
- `mlx_version`, `lightning_mlx_sha`, `macos_version`, `device_name`, `nax_enabled` (bool from runtime probe)

### 4.2 Cell matrix

Models (3): `qwen3.6-27b-8bit`, `qwen3.6-35b-8bit`, `qwen3.6-35b-nsc-ace-saber-8bit`.
Contexts (4): `1024`, `10240`, `51200`, `102400`.
Modes (4): `mtp_only`, `mtp+ngram`, `mtp_only_no_nax`, `mtp+ngram_no_nax`.
Hardware sweep (3): `M1-Max`, `M3-Max`, `M5-Max` (must hit all 3 for P3 regression check).

Total: 3 × 4 × 4 × 3 = **144 cells**. Skip cells where model OOMs on smaller HW; document the skip.

### 4.3 Bench commands (anchor — same flags every run)

```bash
lightning-mlx bench <model> \
  --num-prompts 3 --max-tokens 512 --disable-prefix-cache \
  --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
  --prefill-step-size 8192 --mtp-num-draft-tokens 3 --mtp-optimistic \
  --ctx-size <ctx> \
  --report-json reports/nax/<tier>/<run-id>.json
```

NAX off variant: prefix with `LIGHTNING_DISABLE_METAL4=1` (env var introduced in T2.3; for T1 baseline use `MLX_DISABLE_METAL4=1` if MLX exposes it, else rebuild MLX without 0.30+ support).

### 4.4 Comparison table (auto-generated, committed)

Output: `reports/nax/<tier>/summary.md` with rows:

| model | ctx | mode | hw | prefill_before | prefill_after | Δ% | gen_before | gen_after | Δ% | drift | accept_Δpp |

Color-coded thresholds:
- Δ% ≥ +20% prefill ⇒ **PASS** (green)
- 0 < Δ% < +20% prefill ⇒ **WEAK** (yellow)
- Δ% ≤ 0 prefill on M5 ⇒ **FAIL** (red)
- |Δ%| > 2% on M1/M3 ⇒ **REGRESSION** (red)
- drift_mean_abs > 1e-3 ⇒ **FAIL** (red)
- |accept_Δpp| > 1 ⇒ **FAIL** (red)

### 4.5 Drift measurement

Beside speed, every tier captures logit drift:

```bash
python evals/nax_drift_gate.py \
  --model <model> --prompts evals/golden_prompts.jsonl \
  --ctx 4096,16384,65536 \
  --output reports/nax/<tier>/drift.json
```

Records: `mean_abs`, `max_abs`, `kl_divergence`, `top1_match_rate`, `top5_match_rate`.

### 4.6 Bench tooling tasks (prerequisite to T1)

- M1 — extend `lightning-mlx bench` with `--report-json <path>` flag if absent.
- M2 — add `--ctx-size` flag if absent (use existing prefill-step + prompt repeat).
- M3 — wire `mx.metal.get_peak_memory()` into bench output.
- M4 — create `reports/nax/{t0_baseline,t1,t2,t3}/` skeleton with `README.md` describing capture cmd.
- M5 — write `scripts/nax_bench_sweep.sh` that loops the matrix and dumps JSON.
- M6 — write `scripts/nax_summarize.py` that reads tier JSON dirs and emits `summary.md`.
- M7 — write `evals/nax_drift_gate.py` (slim port of ds4 `compare_logit_drift.py`).

### 4.7 Acceptance ritual (per tier)

1. Capture `BEFORE` snapshot on **all 3 hw** before any code change. Commit JSON.
2. Apply tier changes.
3. Capture `AFTER` snapshot on same 3 hw, same flags.
4. Generate `summary.md`, commit.
5. Compare against P1–P5 gates from §1.
6. If any gate red → do not merge tier; iterate or downgrade scope.
7. If gates green → merge, write tier-completion entry in §10 decision log.

---

## 5. Plan (three tiers, gated)

### T1 — Free win (audit + verify + measure)

**Effort:** hours. **Risk:** none.

Steps:

0. `prereq_bench_tooling` — complete §4.6 M1–M7 (bench JSON flag, ctx flag, peak mem, report dirs, sweep script, summarizer, drift gate). Blocking for everything below.
1. `capture_baseline_t0` — on **all 3 hw** (M1/M3/M5), run `scripts/nax_bench_sweep.sh` with current MLX 0.31.2. Output → `reports/nax/t0_baseline/`. Commit JSON.
2. `verify_mlx_nax_path` — on M5 Max + macOS ≥ 26.2, dump `mx.metal.device_info()`; confirm MLX picks tensor kernels. Pass: log shows Metal4 family ON and tensor matmul selected for hot shapes.
3. `bump_mlx_floor` — raise `pyproject.toml` from `mlx>=0.29.0` to `mlx>=0.30.0` (or current latest stable). Re-lock.
4. `capture_after_t1` — re-run sweep on all 3 hw. Output → `reports/nax/t1/`.
5. `bench_decode_breakout` — extract decode t/s rows from sweep; cross-check first-token-ms.
6. `bench_mtp_acceptance` — verify MTP + n-gram acceptance Δpp ≤ 1.
7. `drift_gate_t1` — `evals/nax_drift_gate.py` against t0 baseline on M5. Mean abs ≤ 1e-3 required.
8. `gen_summary_t1` — `scripts/nax_summarize.py reports/nax/t0_baseline reports/nax/t1 > reports/nax/t1/summary.md`. Commit.
9. `doc_update` — README section: "M5 Max + macOS 26.2+ auto-enables NAX via MLX 0.30+". Note hardware/OS requirement.

Files touched: `pyproject.toml`, `uv.lock`, `README.md`, new `reports/nax-t1-bench.md`.

Pass gates:
- P1, P2 met by upstream MLX alone ⇒ **stop, ship T1, no T2/T3 needed**.
- P1, P2 partially met (>50% of antirez's gain) ⇒ ship T1, plan T2.
- P1, P2 not met ⇒ T2 mandatory.

### T2 — MTPLX custom kernel audit + Metal4 path

**Effort:** days. **Risk:** medium. **Gate:** only if T1 leaves >15% headroom on M5.

Steps:

1. `audit_custom_kernels` — exhaustive grep for `mx.fast.metal_kernel`, `@mx.compile`, raw Metal sources in:
   - `vllm_mlx/models/`
   - `vllm_mlx/speculative/`
   - `vllm_mlx/attention.py`
   - `vllm_mlx/optimizations.py`
   - Classify each: GEMM-shaped vs reduction/permutation/fused-elementwise.
2. `identify_matmul_hot_paths` — profile MTP draft verify, n-gram verify batched matvec, any custom prefill GEMM. Drop anything < 2% of total runtime.
3. `add_metal4_gate` — introduce single capability detection module:
   - File: `vllm_mlx/runtime/metal4.py`
   - Detect via `mx.metal.device_info()` + optional probe kernel.
   - Expose `metal4_available() -> bool` and `m5_nax_hint() -> bool`.
   - Honor env override `LIGHTNING_DISABLE_METAL4=1`.
4. `port_matmul_kernels` — for each hot GEMM kernel:
   - Add Metal4 variant guarded by `#ifdef LMLX_METAL_HAS_TENSOR`.
   - Use ds4 tile params as starting point (64×32, NK=32, 4 simdgroups).
   - Keep legacy kernel as fallback path.
5. `drift_harness` — port slimmed version of ds4's `compare_logit_drift.py`:
   - File: `evals/nax_drift_gate.py`
   - Compare logits NAX-on vs NAX-off at 4k / 16k / 64k ctx for golden prompts.
   - Fail if mean abs diff > 1e-3 or max abs > 1e-2.
6. `ci_gate` — make drift harness blocking on M5 CI runner (if exists; else manual gate documented).
7. `capture_before_t2` — re-snapshot from `reports/nax/t1/` (this is T2's baseline). Confirm hashes match committed JSON.
8. `bench_t2` — re-run sweep on all 3 hw after kernel ports. Output → `reports/nax/t2/`.
9. `drift_gate_t2` — diff vs T1 baseline AND vs legacy-kernel-off path. Both required.
10. `gen_summary_t2` — `scripts/nax_summarize.py reports/nax/t1 reports/nax/t2 > reports/nax/t2/summary.md`. Commit.

Files touched: `vllm_mlx/runtime/metal4.py` (new), affected `vllm_mlx/models/*.py` and `vllm_mlx/speculative/*.py`, `evals/nax_drift_gate.py` (extended), `Makefile` build flag.

Pass gates: P1, P2, P3, P4, P5 all met.

### T3 — Full custom Metal4 kernel suite (fork-and-tune)

**Effort:** weeks. **Risk:** high. **Gate:** only if T1+T2 still leaves >15% headroom AND product priority justifies maintenance burden.

Steps:

1. Replace MLX's matmul with hand-tuned Metal4 kernels for hot shapes only:
   - q_proj / k_proj / v_proj / o_proj fused
   - gate_up_proj fused for SwiGLU
   - MoE expert GEMM (mirror ds4's `metal/moe.metal`)
2. Ship as MLX patch loaded at import time (already pattern in `vllm_mlx/patches/`).
3. Expand drift harness to per-layer logit gates.
4. `capture_before_t3` — snapshot from `reports/nax/t2/`.
5. `bench_t3` — sweep on all 3 hw. Output → `reports/nax/t3/`.
6. `drift_gate_t3` — per-layer logit drift, max abs ≤ 1e-2 per layer.
7. `gen_summary_t3` — diff vs T2. Commit.
8. Document M5-only maintenance burden; pin commitment.

**Default recommendation: do NOT enter T3 unless data forces it.**

---

## 6. Risks

- **R1 hw fragmentation** — NAX is M5+. Must runtime-gate, not compile-time only. M1–M4 users must hit legacy path with zero overhead.
- **R2 OS requirement** — macOS ≥ 26.2. Older OS on M5 hw must fall back gracefully.
- **R3 MLX version churn** — upstream may rewrite NAX hooks; pin minor version on `mlx-lm` + `mlx` together.
- **R4 MTPLX interaction** — custom kernels must compose with MTP draft/verify dispatch order. Risk of breaking acceptance rate.
- **R5 quant path** — most lightning-mlx models ship 4/6/8-bit. NAX helps fp16 prefill GEMM; quant dequant→fp16 staging is needed (ds4 pattern). Bench must confirm gain holds with quant.
- **R6 maintenance** — every custom kernel is debt. T3 only if measured ROI > 1 quarter of eng time.

---

## 7. Hard no-go

- N1: no compile-time-only gating. Runtime detect mandatory.
- N2: no M5-only ship. Legacy fallback always present.
- N3: no kernel merge without drift gate green.
- N4: no MTP acceptance regression > 1pp.
- N5: no removal of existing custom HC kernels (lines 420, 606 in `deepseek_v4.py`) — they are not GEMM, T2 must leave them alone.

---

## 8. Open questions

- Q1: does MLX 0.30+ expose a runtime flag to disable NAX for A/B benching, or must we rebuild MLX? (T1.3 depends.)
- Q2: does mlx-lm pin a max MLX version that blocks the 0.30+ bump? (T1.2 blocker check.)
- Q3: is there CI hardware on M5? If not, T2 drift gate is manual-only.
- Q4: do MTPLX custom kernels in `vllm_mlx/speculative/` include any GEMM-shaped Metal kernel beyond `mx.fast.scaled_dot_product_attention`? (grep at T2.1 answers.)

---

## 9. Decision log

- 2026-05-23 — antirez published M5 NAX result (40–80% prefill on DS4). Inspired this plan.
- 2026-05-23 — confirmed MLX 0.30.0 already supports NAX upstream. Plan biased toward T1 first.
- 2026-05-23 — confirmed lightning-mlx custom kernels at `deepseek_v4.py:420,606` are HyperConnection, not GEMM ⇒ out of scope for NAX port.

---

## 10. Next action

Execute T1.0 (`prereq_bench_tooling`) — implement bench JSON output, ctx flag, peak-mem capture, sweep script, summarizer, drift gate. Then T1.1 baseline capture on all 3 hw. Then T1.2 NAX verify on M5 Max + macOS ≥ 26.2.

**No code change ships without paired before/after snapshot committed to `reports/nax/`.**
