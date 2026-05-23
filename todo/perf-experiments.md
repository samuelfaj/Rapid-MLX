# Performance Experiments — Agentic Workload

**Status:** PLANNING
**Workload:** agentic (short bursts, tool calls, mixed turns, ≤32k typical ctx)
**HW focus:** M5 primary; M1–M4 must not regress
**Maintenance budget:** solo. Reject anything with non-trivial ongoing cost.
**Workflow:** one change at a time → measure → keep if win, revert if loss → next.

Companion doc: `nax-metal4-plan.md` (NAX/Metal4 track, runs in parallel).

---

## 0. Audit — what is already shipped

Inventory pulled from repo. **Do not re-implement these.**

### Speculative decoding
- [x] MTP via mlx-lm batched engine (`vllm_mlx/engine/batched.py`)
- [x] MTP optimistic mode (auto-on, `--mtp-optimistic`)
- [x] N-gram (prompt-lookup) drafter (`vllm_mlx/speculative/ngram_drafter.py`)
- [x] `<think>`-aware + `<tool_call>`-aware state machines
- [x] Adaptive K per match confidence
- [x] Hybrid verify (MTP head appended after n-gram tail)
- [x] Self-tuning per-request + global auto-disable (no-regression guarantee vs MTP-only)
- [x] DFlash block-diffusion drafter (`vllm_mlx/speculative/dflash_drafter.py`)
- [x] Native MTP path (`vllm_mlx/speculative/native_mtp/`)

### KV / cache
- [x] Prompt cache (always on, 5–30× TTFT)
- [x] Prefix cache w/ disk persistence (`load_prefix_cache_from_disk`/`save_prefix_cache_to_disk` on shutdown/start)
- [x] KV cache quantization (`--kv-cache-quantization`, configurable bits)
- [x] KV cache turboquant (`--kv-cache-turboquant`, auto-bits by head_dim, K stays fp16)
- [x] Paged cache (`--use-paged-cache --paged-cache-block-size`)
- [x] Chunked prefill (`--chunked-prefill-tokens`)

### Attention
- [x] Flash attention via `mx.fast.scaled_dot_product_attention`
- [x] Sliding window plumbed (`vllm_mlx/attention.py:154`) — coverage TBD

### Quantization
- [x] 4 / 6 / 8 bit MTPLX-optimized weights per alias
- [x] 2-bit DQ for DeepSeek-V4-Flash

### System
- [x] Daemon + boot persistence (LaunchAgent / systemd-user)
- [x] Hardware detection M1–M4 (`vllm_mlx/optimizations.py` HARDWARE_PROFILES)
- [x] 17 tool-call parsers (per ROADMAP.md)
- [x] Per-agent profiles (codex, claude, aider, cline, goose, hermes, openhands, etc.)

### Already rejected (do not revisit)
- Medusa (superseded by EAGLE-3)
- Prompt lookup standalone (kept only as layered n-gram)
- `--no-thinking` as default (per GOAL.md — degrades agentic quality)

### Gaps spotted in audit
- **M5 missing** from `HARDWARE_PROFILES` dict — only M1–M4 listed
- **NAX/Metal4** detection not yet wired (see `nax-metal4-plan.md`)
- KV-quant is **opt-in**; agentic short-turn defaults may benefit from auto-on at conservative bits

---

## 1. Workflow protocol — try / eval / keep / revert

**Every experiment is one change. No bundling.**

### Steps per experiment

1. `pre` — confirm previous experiment landed clean (kept or reverted; no dangling state).
2. `branch` — `exp/<id>-<slug>` from current main.
3. `snapshot_before` — run agentic sweep (§2), commit JSON under `reports/exp/<id>/before/`.
4. `change` — apply the single change. Commit with `exp(<id>): <description>`.
5. `snapshot_after` — re-run agentic sweep. Commit JSON under `reports/exp/<id>/after/`.
6. `compare` — `scripts/nax_summarize.py before after > reports/exp/<id>/summary.md`.
7. `verdict` — apply gates (§1.2). Record decision in `reports/exp/<id>/verdict.md`.
8. `act` —
   - **Keep**: merge branch to main. Update `todo/perf-experiments.md` checkbox.
   - **Revert**: delete branch. Update `todo/perf-experiments.md` with reason.
9. `log` — append to `reports/exp/INDEX.md`: `id | slug | verdict | Δ%agentic_short | Δ%agentic_long | merged_sha or reverted`.

### Keep gates (ALL must pass)

- **G1 agentic-short** Δ ≥ +5% tok/s on M5 (`create snake game` fixture)
- **G2 agentic-long** Δ ≥ 0% tok/s on M5 (no regression)
- **G3 M1/M3 regression** within ±2% on both
- **G4 drift** mean abs ≤ 1e-3 vs baseline (if change touches numerics)
- **G5 acceptance** MTP + n-gram acceptance Δpp within ±1pp
- **G6 maintenance** change adds ≤ 1 config flag OR ≤ 50 LOC

### Revert triggers (ANY triggers revert)

- Any G1–G5 fails
- G6 fails (too much maint cost)
- Test suite regression
- Crash on M1/M3/M5

### Bench fixture — agentic sweep

```bash
# Short turn (chat hello, tool-only)
lightning-mlx bench <model> --prompt-file evals/fixtures/agentic_short.txt \
  --num-prompts 5 --max-tokens 256 --report-json reports/exp/<id>/<dir>/short.json

# Long turn (snake game artifact)
lightning-mlx bench <model> --prompt-file evals/fixtures/agentic_long.txt \
  --num-prompts 3 --max-tokens 2048 --report-json reports/exp/<id>/<dir>/long.json

# Mixed (5 short + 3 long, interleaved)
scripts/agentic_mixed.sh <model> reports/exp/<id>/<dir>/mixed.json
```

Models: `qwen3.6-35b-nsc-ace-saber-8bit`, `qwen3.6-27b-8bit`.
HW: M5 mandatory; M1 + M3 nightly (or before merge if M5 win is >10%).

### Prereq tooling (blocking E0)

- `scripts/agentic_mixed.sh` — runs short+long mixed
- `evals/fixtures/agentic_{short,long}.txt` — golden prompts
- `reports/exp/INDEX.md` skeleton
- Reuse summarizer + drift gate from NAX plan §4.6

---

## 2. Experiment queue (ordered by ROI for agentic + solo)

Format per card:
- **id**: short tag
- **change**: single edit
- **hypothesis**: why this should win
- **effort**: hours / days / week
- **risk**: low / med / high
- **expected gain**: agentic %
- **rollback**: how to revert

### Quick wins — config-only, hours

#### E1 — `m5_hw_profile_entry`
- **change**: add M5 / M5 Pro / M5 Max / M5 Ultra entries to `HARDWARE_PROFILES` in `vllm_mlx/optimizations.py`. Estimate bandwidth + GPU cores from public Apple specs.
- **hypothesis**: unknown chip falls back to 200 GB/s default → suboptimal auto-tuning of step sizes.
- **effort**: hours
- **risk**: low
- **expected gain**: 0% directly; unlocks correct heuristics for E5/E7
- **rollback**: revert one file

#### E2 — `mtp_depth_sweep`
- **change**: sweep `--mtp-num-draft-tokens` ∈ {2, 3, 4, 5} on agentic fixture. Pick best per model, write into alias preset.
- **hypothesis**: current default 3 may not be optimal for agentic-short (tool-call structure repeats).
- **effort**: hours
- **risk**: low (config-only)
- **expected gain**: +5–15% agentic-short
- **rollback**: revert alias preset

#### E3 — `kv_turboquant_default_on`
- **change**: enable `--kv-cache-turboquant` by default in agentic agent profiles (`vllm_mlx/agents/profiles/*.yaml`). Keep flag overridable.
- **hypothesis**: agentic decode is BW-bound; turboquant halves KV bytes at near-zero quality cost; already shipped but opt-in.
- **effort**: hours
- **risk**: low (existing code path)
- **expected gain**: +10–25% agentic-long decode
- **rollback**: flip default back to off

#### E4 — `prefix_cache_size_tune`
- **change**: raise `--prefix-cache-size` default per agentic profile. Measure hit rate.
- **hypothesis**: agentic sessions repeat tool-call scaffolding; bigger prefix cache → fewer prefills.
- **effort**: hours
- **risk**: low (memory only)
- **expected gain**: +10–30% TTFT for repeat turns
- **rollback**: revert default

#### E5 — `chunked_prefill_default_on`
- **change**: enable `--chunked-prefill-tokens 8192` by default for agentic profiles when ctx > 16k.
- **hypothesis**: prefill latency dominates long-turn TTFT; chunking lets generation start sooner on next turn.
- **effort**: hours
- **risk**: low
- **expected gain**: +5–15% mixed throughput
- **rollback**: revert default

#### E6 — `power_mode_doc`
- **change**: docs only — recommend `sudo pmset -a powermode 2` (high power) and Game Mode for sustained bench. No code.
- **hypothesis**: thermal/power throttling skews bench noise band.
- **effort**: hours
- **risk**: none
- **expected gain**: bench stability +5%; user gain marginal
- **rollback**: delete doc lines

### Medium — small code, days

#### E7 — `auto_kv_bits_by_hw`
- **change**: in `vllm_mlx/scheduler.py`, auto-pick KV bits based on detected hw (M1/M2 → keep fp16, M3+ → 8-bit, M5 → 6-bit). Single function.
- **hypothesis**: lower-bw machines waste cycles on dequant; higher-bw can absorb 6-bit.
- **effort**: 1 day
- **risk**: low–med
- **expected gain**: +5–15% on M5, no regression on M1
- **rollback**: revert function

#### E8 — `ngram_K_per_workload`
- **change**: per-agent profile override of n-gram max-K (`--ngram-max-k`). Codex profile = higher K (more repetition), generic = current.
- **hypothesis**: agent traces have profile-specific repetition rates.
- **effort**: 1 day
- **risk**: low
- **expected gain**: +3–10% agentic
- **rollback**: revert profiles

#### E9 — `nax_runtime_detect`
- **change**: thin module `vllm_mlx/runtime/metal4.py` — detect M5 + macOS 26.2 + Metal4 family. Log + expose `metal4_available()`. No kernel changes. Wires into TUI status panel.
- **hypothesis**: precondition for any later Metal4 kernel work; also surfaces hw capability to users.
- **effort**: 1 day
- **risk**: low
- **expected gain**: 0% directly; observability + foundation
- **rollback**: delete module

#### E10 — `mlx_floor_bump`
- **change**: `pyproject.toml` `mlx>=0.30.0` (was 0.29). Re-lock.
- **hypothesis**: MLX 0.30 adds upstream NAX kernels; 0.30.4 adds faster fused vector GQA — free wins.
- **effort**: hours
- **risk**: low (semver minor, watch mlx-lm compat)
- **expected gain**: +5–20% on M5; 0% elsewhere; faster GQA broad
- **rollback**: pin back to 0.29

#### E11 — `dflash_sweep`
- **change**: sweep DFlash block size + adaptive bounds on agentic fixture. Pick best per agent profile.
- **hypothesis**: DFlash is shipped but un-tuned for agentic; defaults may favor raw decode bench.
- **effort**: 1 day
- **risk**: low
- **expected gain**: +5–15% if model has DFlash sidecar
- **rollback**: revert profile

#### E12 — `sampler_gpu_path`
- **change**: ensure top-p/top-k sampling stays on GPU (no CPU sync per token). Audit `vllm_mlx/pipeline/decode.py`.
- **hypothesis**: any per-token CPU sync caps tok/s on M5 (fast GPU).
- **effort**: 1–2 days
- **risk**: low–med
- **expected gain**: +3–8% small-model decode
- **rollback**: revert change

#### E13 — `mx_compile_coverage`
- **change**: wrap remaining hot-path Python helpers in `@mx.compile` (audit grep `@mx.compile` vs hot fns in deepseek_v4.py / qwen models).
- **hypothesis**: graph fusion saves dispatches; already done in some spots, not all.
- **effort**: 1–2 days
- **risk**: low (compile failure → fallback)
- **expected gain**: +2–8%
- **rollback**: remove decorator

### Larger — week+, gated on prior wins

#### E14 — `sliding_window_default_long_ctx`
- **change**: enable sliding window attention (already plumbed) for ctx > 32k in agentic profiles. Window = 16k.
- **hypothesis**: long agentic sessions rarely need full ctx attention; window cuts attn cost.
- **effort**: 2–3 days (validation heavy)
- **risk**: med (quality)
- **expected gain**: +30–80% prefill @ 64k+ ctx; 0% short ctx
- **rollback**: revert default
- **gate**: only if agentic-long ctx typically > 32k (measure first)

#### E15 — `eagle3_evaluation`
- **change**: small spike — port one EAGLE-3 reference to MLX, single model (e.g. qwen3.6-35b). Measure acceptance vs MTP+ngram.
- **hypothesis**: EAGLE-3 reportedly 3–6× decode; may compose with MTP or replace it.
- **effort**: 1 week (spike only; not productionizing)
- **risk**: high (maint debt if kept)
- **expected gain**: +30–60% if kept; 0 if spike abandoned
- **rollback**: delete spike branch
- **gate**: only if E1–E13 cumulative falls short of agentic targets

#### E16 — `nax_kernel_port_t2`
- **change**: see `nax-metal4-plan.md` §5 T2. Audit `mx.fast.metal_kernel` sites for GEMM-shape; port to Metal4 tensor.
- **gate**: only if NAX T1 (E10 + verify) leaves >15% gap on M5

### Skip / deferred indefinitely

- **ReDrafter** — needs trained heads per model. Maintenance cost prohibitive for solo.
- **Full Metal4 fork (T3 in NAX plan)** — weeks of work, M5-only debt.
- **W4A4 activation quant** — depends on HW + custom kernels.
- **Lookahead / Jacobi standalone** — composes poorly with MTP; revisit only if EAGLE-3 not adopted.

---

## 3. Sequencing

Run experiments **strictly serial**. No parallel branches.

Phase 1 (week 1 — config quick wins):
E0 (tooling) → E1 → E2 → E3 → E4 → E5 → E6

Phase 2 (week 2 — small code):
E7 → E8 → E9 → E10 → E11 → E12 → E13

Phase 3 (gated):
E14 (if long ctx data justifies) → E15 (if cumulative gain short of target) → E16 (if NAX T1 not enough)

After each phase, compute cumulative agentic Δ% and decide whether next phase is worth it.

---

## 4. Stop conditions

Stop running experiments when **any** of:

- Agentic-short M5 ≥ +50% over baseline (T0 from NAX plan)
- Three consecutive experiments revert
- Maintenance cost ledger (§5) exceeds 200 LOC added beyond baseline

---

## 5. Maintenance ledger

Append per kept experiment:

| id | merged_sha | LOC added | flags added | est ongoing cost |
|----|------------|----------:|------------:|------------------|

If cumulative LOC > 200, halt phase 2/3 entries.

---

## 6. Open questions

- Q1 — does current bench fixture (`create snake game`) reflect real agentic mix? Should we add coding-agent traces from Codex/Claude profiles?
- Q2 — what's the M5 device name string? Need exact match for `HARDWARE_PROFILES` (M5 Pro? M5 Max?). Pull from `mx.device_info()` on actual hw.
- Q3 — does `--kv-cache-turboquant` cause drift on `qwen3.6-35b-nsc-ace-saber-8bit`? Run drift gate first before E3.
- Q4 — does mlx 0.30 + mlx-lm latest pair cleanly? Check before E10.

---

## 7. Next action

Build E0 prereq tooling:
1. Create `evals/fixtures/agentic_short.txt`, `agentic_long.txt` (extract from existing GOAL.md fixture + 1 tool-only short).
2. Write `scripts/agentic_mixed.sh`.
3. Create `reports/exp/INDEX.md` skeleton.
4. Reuse `scripts/nax_summarize.py` from NAX plan §4.6 (build it once).

Then start E1 (`m5_hw_profile_entry`) — simplest, lowest risk, prerequisite for several others.
