# TODO — Execution Playbook

**Audience:** any LLM coding agent, even a limited one.
**Working dir:** `/Users/samuelfajreldines/dev/rapid-mlx`
**Goal:** capture every perf win possible on agentic workloads, M5-first, M1–M4 safe, solo-maintainable.

---

## 0. Read first — operating rules

You MUST follow these without exception.

### 0.1 One change per step
Never bundle. Each step does exactly one thing. If you find a second thing, write it down — do not do it.

### 0.2 Always paired snapshot
For every code/config change:
1. Snapshot bench BEFORE.
2. Apply change.
3. Snapshot bench AFTER.
4. Decide keep/revert.

No exceptions. If you can't snapshot (no hw access), STOP and tell the user.

### 0.3 Branch per step
```bash
git checkout -b exp/<id>-<slug>
```
Where `<id>` is the step id (E1, E2, …) and `<slug>` is a kebab-case short label.

### 0.4 Commit format
```
exp(<id>): <description>

Refs: todo/todo.md
```

### 0.5 Decision gates — same for every step

KEEP if all true:
- `G1` agentic-short Δtok/s ≥ +5% on M5
- `G2` agentic-long Δtok/s ≥ 0% on M5 (no regression)
- `G3` M1/M3 within ±2% (skip cell if hw unavailable; mark in verdict)
- `G4` numeric drift mean_abs ≤ 1e-3 (if change touches numerics; skip otherwise)
- `G5` MTP+ngram acceptance Δpp within ±1pp
- `G6` change ≤ 1 new flag AND ≤ 50 LOC

REVERT if ANY false.

### 0.6 Logging
After verdict, append one row to `reports/exp/INDEX.md`:
```
| <id> | <slug> | keep|revert | <Δ short M5%> | <Δ long M5%> | <sha or "reverted"> |
```

### 0.7 Stop conditions
Halt entire playbook when ANY:
- 3 consecutive reverts.
- Maintenance ledger cumulative LOC > 200.
- Agentic-short M5 cumulative gain ≥ +50% over T0 baseline.

### 0.8 Hard no-go
- N1 — never break M1/M3 (G3).
- N2 — never compile-time-only gate; always runtime detect.
- N3 — never skip drift gate (G4) when numerics change.
- N4 — never touch `vllm_mlx/models/deepseek_v4.py` lines 420 / 606 (HyperConnection custom kernels — not GEMM).
- N5 — never use `--no-thinking` as default (per `GOAL.md`).
- N6 — never push to remote without user OK.

---

## 1. Phase 0 — Build tooling (DO FIRST)

All later steps require these. Implement in this order.

### T0.1 ✅ — Create reports skeleton

```bash
mkdir -p reports/exp reports/nax/{t0_baseline,t1,t2,t3} evals/fixtures scripts
```

Then create `reports/exp/INDEX.md`:

```markdown
# Experiment Index

| id | slug | verdict | Δ short M5 % | Δ long M5 % | merged sha or reverted |
|----|------|---------|-------------:|------------:|------------------------|
```

Verify:
```bash
test -f reports/exp/INDEX.md && echo OK
```

### T0.2 ✅ — Write agentic fixture prompts

Create `evals/fixtures/agentic_short.txt` (exactly one prompt):
```
List the files in the current directory and tell me which file is the largest. Use the available tools.
```

Create `evals/fixtures/agentic_long.txt` (exactly one prompt):
```
Create a Snake game using HTML and TypeScript. Make sure it runs in the browser, has score tracking, game over state, and clean code. Output every file needed and explain how to run it.
```

Create `evals/fixtures/agentic_tool_heavy.txt`:
```
Read pyproject.toml, then list every dependency under [project.dependencies], then for each one tell me its purpose in one short line.
```

Verify:
```bash
wc -l evals/fixtures/agentic_*.txt
```

### T0.3 ✅ — Add `--prompt-file` and `--report-json` to bench

File: `vllm_mlx/cli.py`

After line 2248 (`--max-tokens` arg) add two args inside `bench_parser`:
```python
bench_parser.add_argument(
    "--prompt-file",
    type=str,
    default=None,
    help="Path to a text file containing the prompt(s) — one per line. "
    "Overrides synthetic prompts.",
)
bench_parser.add_argument(
    "--report-json",
    type=str,
    default=None,
    help="Write a JSON metrics report to this path.",
)
```

In `bench_command` (line 968), after `model, tokenizer = load_model_with_fallback(...)`:
- If `args.prompt_file` set, read lines from file into `prompts` (skip blanks, strip `\n`).
- Replace the synthetic prompt list with that.

At the end of the bench (after summary print), if `args.report_json` set, write a dict:
```python
{
  "model": args.model,
  "num_prompts": len(prompts),
  "prompt_file": args.prompt_file,
  "max_tokens": args.max_tokens,
  "prefill_tok_s": <median>,
  "gen_tok_s": <median>,
  "first_token_ms": <median>,
  "peak_gpu_mem_mb": mx.metal.get_peak_memory() / 1024 / 1024,
  "mtp_accept_rate": <if available>,
  "ngram_accept_rate": <if available>,
  "device_name": mx.device_info().get("device_name"),
  "mlx_version": <import mlx; mlx.__version__>,
  "git_sha": <subprocess git rev-parse HEAD>,
}
```

Verify:
```bash
lightning-mlx bench qwen3.5-4b --prompt-file evals/fixtures/agentic_short.txt \
  --num-prompts 1 --max-tokens 64 --report-json /tmp/test-report.json
test -f /tmp/test-report.json && cat /tmp/test-report.json | python3 -m json.tool
```

### T0.4 ✅ — Write summarizer

Create `scripts/nax_summarize.py`:

```python
#!/usr/bin/env python3
"""Compare two bench report directories. Output markdown delta table."""
import json
import sys
from pathlib import Path


def load_reports(dir_path):
    reports = {}
    for f in Path(dir_path).rglob("*.json"):
        with f.open() as fh:
            data = json.load(fh)
        reports[f.stem] = data
    return reports


def pct(before, after):
    if before == 0:
        return 0.0
    return ((after - before) / before) * 100.0


def main():
    if len(sys.argv) != 3:
        print("usage: nax_summarize.py <before_dir> <after_dir>", file=sys.stderr)
        sys.exit(2)
    before = load_reports(sys.argv[1])
    after = load_reports(sys.argv[2])
    keys = sorted(set(before) | set(after))
    print("| cell | prefill before | prefill after | Δ% | gen before | gen after | Δ% |")
    print("|------|---------------:|--------------:|---:|-----------:|----------:|---:|")
    for k in keys:
        b = before.get(k, {})
        a = after.get(k, {})
        pb = b.get("prefill_tok_s", 0)
        pa = a.get("prefill_tok_s", 0)
        gb = b.get("gen_tok_s", 0)
        ga = a.get("gen_tok_s", 0)
        print(
            f"| {k} | {pb:.2f} | {pa:.2f} | {pct(pb, pa):+.1f}% | "
            f"{gb:.2f} | {ga:.2f} | {pct(gb, ga):+.1f}% |"
        )


if __name__ == "__main__":
    main()
```

Make executable:
```bash
chmod +x scripts/nax_summarize.py
```

Verify:
```bash
mkdir -p /tmp/before /tmp/after
echo '{"prefill_tok_s": 100, "gen_tok_s": 50}' > /tmp/before/a.json
echo '{"prefill_tok_s": 120, "gen_tok_s": 55}' > /tmp/after/a.json
python3 scripts/nax_summarize.py /tmp/before /tmp/after
# Expect: +20.0% / +10.0%
```

### T0.5 ✅ — Write agentic sweep script

Create `scripts/agentic_sweep.sh`:

```bash
#!/usr/bin/env bash
# Run the standard agentic bench cell matrix.
# Usage: agentic_sweep.sh <model> <out_dir>
set -euo pipefail

MODEL="${1:?missing model}"
OUT="${2:?missing out dir}"
mkdir -p "$OUT"

# Short turn — tool-only
lightning-mlx bench "$MODEL" \
  --prompt-file evals/fixtures/agentic_short.txt \
  --num-prompts 5 --max-tokens 256 \
  --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
  --prefill-step-size 8192 --mtp-optimistic \
  --disable-prefix-cache \
  --report-json "$OUT/short.json"

# Tool-heavy
lightning-mlx bench "$MODEL" \
  --prompt-file evals/fixtures/agentic_tool_heavy.txt \
  --num-prompts 3 --max-tokens 512 \
  --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
  --prefill-step-size 8192 --mtp-optimistic \
  --disable-prefix-cache \
  --report-json "$OUT/tool_heavy.json"

# Long-turn artifact
lightning-mlx bench "$MODEL" \
  --prompt-file evals/fixtures/agentic_long.txt \
  --num-prompts 3 --max-tokens 2048 \
  --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
  --prefill-step-size 8192 --mtp-optimistic \
  --disable-prefix-cache \
  --report-json "$OUT/long.json"

echo "DONE: $OUT"
```

```bash
chmod +x scripts/agentic_sweep.sh
```

Verify it runs end-to-end on smallest model:
```bash
./scripts/agentic_sweep.sh qwen3.5-4b /tmp/test-sweep
ls /tmp/test-sweep
```

### T0.6 ✅ — Write drift gate

Create `evals/nax_drift_gate.py`:

```python
#!/usr/bin/env python3
"""Compare logits of two configurations on golden prompts.

Usage:
    nax_drift_gate.py --model <model> --prompts <file> --output <json>

Reads logits via mlx_lm.utils.generate(..., return_logits=True) for top-K only.
Fails if mean abs diff > 1e-3 or max abs > 1e-2.
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


def get_topk_logits(model, tokenizer, prompt, k=50, max_tokens=64):
    tokens = tokenizer.encode(prompt)
    arr = mx.array(tokens)[None, :]
    logits = model(arr)
    # last-token logits
    last = logits[0, -1, :]
    top_vals = mx.topk(last, k)
    return np.asarray(top_vals.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model, tokenizer = load(args.model)
    prompts = [
        line.strip() for line in Path(args.prompts).read_text().splitlines()
        if line.strip()
    ]

    out = {"prompts": [], "mean_abs": 0.0, "max_abs": 0.0}
    # NOTE: this is the BASELINE capture. The COMPARE step needs two captures.
    # Save top-K logits per prompt for later diffing.
    for prompt in prompts:
        topk = get_topk_logits(model, tokenizer, prompt)
        out["prompts"].append({"prompt": prompt[:80], "topk": topk.tolist()})

    Path(args.output).write_text(json.dumps(out))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
```

Companion `evals/nax_drift_compare.py`:

```python
#!/usr/bin/env python3
"""Compare two drift gate captures. Fail if drift > threshold."""
import json
import sys
import numpy as np

BEFORE = sys.argv[1]
AFTER = sys.argv[2]
b = json.loads(open(BEFORE).read())
a = json.loads(open(AFTER).read())
assert len(b["prompts"]) == len(a["prompts"]), "prompt count mismatch"

diffs = []
for bp, ap in zip(b["prompts"], a["prompts"]):
    bp_arr = np.array(bp["topk"])
    ap_arr = np.array(ap["topk"])
    diffs.append(np.abs(bp_arr - ap_arr))
diffs = np.concatenate(diffs)
mean_abs = float(diffs.mean())
max_abs = float(diffs.max())
print(f"mean_abs={mean_abs:.6f}  max_abs={max_abs:.6f}")

ok = mean_abs <= 1e-3 and max_abs <= 1e-2
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
```

Verify drift gate runs at all:
```bash
python3 evals/nax_drift_gate.py --model qwen3.5-4b \
  --prompts evals/fixtures/agentic_short.txt \
  --output /tmp/drift.json
```

### T0.7 ✅ — Commit tooling

```bash
git checkout -b exp/T0-tooling
git add reports/ evals/fixtures/ evals/nax_drift_gate.py evals/nax_drift_compare.py scripts/ vllm_mlx/cli.py
git commit -m "$(cat <<'EOF'
exp(T0): add agentic bench tooling — fixtures, sweep script, summarizer, drift gate

Refs: todo/todo.md
EOF
)"
```

Merge to main (no perf gates apply — tooling only):
```bash
git checkout main
git merge --no-ff exp/T0-tooling
```

---

## 2. Phase 1 — Baseline + config quick wins

### T1.0 ✅ — Capture T0 baseline (BLOCKING)

This is the reference for every later step. Run on **every available** Apple Silicon machine.

```bash
mkdir -p reports/exp/T0/<hw>
# replace <hw> with: m1max, m3max, m5max (whichever you have)

./scripts/agentic_sweep.sh qwen3.6-35b-nsc-ace-saber-8bit \
  reports/exp/T0/<hw>/qwen35b-nsc-ace-saber-8bit
./scripts/agentic_sweep.sh qwen3.6-27b-8bit \
  reports/exp/T0/<hw>/qwen27b-8bit
```

Commit baselines:
```bash
git checkout -b exp/T0-baseline
git add reports/exp/T0/
git commit -m "exp(T0): baseline agentic sweep on <hw>"
git checkout main && git merge --no-ff exp/T0-baseline
```

### E1 ✅ — m5_hw_profile_entry

**Goal:** Add M5 family to `HARDWARE_PROFILES`.

Step:
```bash
git checkout -b exp/E1-m5-profile
```

Edit `vllm_mlx/optimizations.py`. After the M4 block in `HARDWARE_PROFILES` (around line 75), add:

```python
    # M5 Series — verified on 2026-05 hw. Bandwidth from Apple specs.
    "M5": {"bandwidth": 153, "gpu_cores": 10},
    "M5 Pro": {"bandwidth": 320, "gpu_cores": 16},
    "M5 Max": {"bandwidth": 600, "gpu_cores": 40},
    "M5 Ultra": {"bandwidth": 1200, "gpu_cores": 80},
```

**Bandwidth note:** if you don't have authoritative numbers, leave a TODO comment and use M4 Max + 10% as conservative estimate. Mark in commit msg.

Verify:
```bash
python3 -c "
from vllm_mlx.optimizations import HARDWARE_PROFILES
assert 'M5 Max' in HARDWARE_PROFILES, 'M5 Max missing'
print('OK')
"
```

Snapshot before (already in `reports/exp/T0/`). Snapshot after:
```bash
./scripts/agentic_sweep.sh qwen3.6-35b-nsc-ace-saber-8bit \
  reports/exp/E1/after/<hw>/qwen35b-nsc-ace-saber-8bit
```

Compare:
```bash
python3 scripts/nax_summarize.py reports/exp/T0/<hw> reports/exp/E1/after/<hw> \
  > reports/exp/E1/summary.md
```

Verdict: this change has no numeric effect, so G1/G2 should be ~0% (noise band). Keep if no regression.

Commit + merge:
```bash
git add vllm_mlx/optimizations.py reports/exp/E1/
git commit -m "exp(E1): add M5 family to HARDWARE_PROFILES"
git checkout main && git merge --no-ff exp/E1-m5-profile
```

Append to `reports/exp/INDEX.md`:
```
| E1 | m5-profile | keep | ~0% | ~0% | <sha> |
```

### E2 ✅ — mtp_depth_sweep

**Goal:** Find best `--mtp-num-draft-tokens` per agentic model.

```bash
git checkout -b exp/E2-mtp-depth
```

Run sweep (no code change, pure measurement):
```bash
for d in 2 3 4 5; do
  mkdir -p reports/exp/E2/sweep/d$d
  lightning-mlx bench qwen3.6-35b-nsc-ace-saber-8bit \
    --prompt-file evals/fixtures/agentic_short.txt \
    --num-prompts 5 --max-tokens 256 \
    --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
    --prefill-step-size 8192 --mtp-optimistic \
    --mtp-num-draft-tokens $d \
    --disable-prefix-cache \
    --report-json reports/exp/E2/sweep/d$d/short.json
done
```

Pick winner. If winner != default (which is 1 in current code, 3 in README example), bake into alias preset.

Find the alias config: `vllm_mlx/aliases.json` and any per-model preset in `vllm_mlx/cli.py` (search `mtp_num_draft_tokens` for the preset setter).

Edit ONLY the default for matched aliases. ≤ 5 LOC.

Snapshot after:
```bash
./scripts/agentic_sweep.sh qwen3.6-35b-nsc-ace-saber-8bit reports/exp/E2/after/<hw>/qwen35b
```

Compare vs T0 baseline. Apply gates. Decide.

If revert:
```bash
git checkout main
git branch -D exp/E2-mtp-depth
```

If keep, commit + merge + log.

### E3 — kv_turboquant_default_on

**Goal:** Default turboquant on for agentic profiles.

```bash
git checkout -b exp/E3-turboquant-default
```

**Drift gate FIRST** (G4 mandatory — touches numerics):
```bash
mkdir -p reports/exp/E3/drift
python3 evals/nax_drift_gate.py --model qwen3.6-35b-nsc-ace-saber-8bit \
  --prompts evals/fixtures/agentic_short.txt \
  --output reports/exp/E3/drift/before.json
# Now turn turboquant on manually:
LIGHTNING_TURBOQUANT=1 python3 evals/nax_drift_gate.py ... --output reports/exp/E3/drift/after.json
# (modify drift gate to honor an env var, OR just bench with --kv-cache-turboquant flag added)
python3 evals/nax_drift_compare.py reports/exp/E3/drift/before.json reports/exp/E3/drift/after.json
```

If drift FAIL → revert immediately.

If drift PASS, modify default. Where:
- File: `vllm_mlx/cli.py`
- Find the `--kv-cache-turboquant` arg (search for `kv_cache_turboquant`). It is currently `store_true`. Add a `--no-kv-cache-turboquant` companion, and flip default to `True` when serving / benching agentic.
- Alternative (lower-risk): keep flag default `False`, but auto-enable inside `cli.py` when the loaded alias name matches agentic models (`qwen3.6-*`, `ornstein*`, `qwopus*`). Add ≤ 15 LOC heuristic.

Snapshot before/after using sweep script. Apply gates.

### E4 — prefix_cache_size_tune

**Goal:** Tune `--prefix-cache-size` upward where memory allows.

```bash
git checkout -b exp/E4-prefix-cache-size
```

Sweep values: 4096, 8192, 16384, 32768 (tokens). For each value run sweep with prefix cache ENABLED (drop `--disable-prefix-cache` from sweep script for this experiment only). Run on the agentic_tool_heavy fixture twice consecutively — second run measures cache benefit.

Pick value that maximizes second-run TTFT improvement on M5 without exceeding 5GB extra GPU mem.

Edit default in `vllm_mlx/cli.py` (`--prefix-cache-size` definition).

Apply gates.

### E5 — chunked_prefill_default_on

```bash
git checkout -b exp/E5-chunked-prefill
```

Edit `vllm_mlx/cli.py`: default `--chunked-prefill-tokens` from 0 to 8192 for agentic-aliased models only. ≤ 15 LOC.

Snapshot sweep w/ `agentic_long.txt` (long ctx) before/after. Apply gates.

### E6 ✅ — power_mode_doc

```bash
git checkout -b exp/E6-power-mode-doc
```

Add section to `README.md`:

```markdown
## Sustained-performance tips

For sustained generation speed (e.g. coding agents over many turns):

```bash
sudo pmset -a powermode 2          # macOS — high power mode
caffeinate -dimsu &                 # prevent display sleep during long sessions
```

Enable macOS Game Mode in System Settings for thread priority boost.
```

No bench needed (doc-only). Commit + merge.

---

## 3. Phase 2 — Small code, gated

### E7 — auto_kv_bits_by_hw

```bash
git checkout -b exp/E7-auto-kv-bits
```

File: `vllm_mlx/scheduler.py`.

After the existing `kv_cache_turboquant_*` fields in `SchedulerConfig` (around line 92), add a helper:

```python
def auto_kv_bits_for_hw():
    """Pick KV bits per detected Apple Silicon chip."""
    from vllm_mlx.optimizations import detect_hardware
    hw = detect_hardware()
    if hw.chip_name.startswith("M1") or hw.chip_name.startswith("M2"):
        return None  # keep fp16
    if hw.chip_name.startswith("M5"):
        return 6
    return 8  # M3, M4, unknown
```

Wire into config init: when user does NOT pass `--kv-cache-quantization-bits`, call `auto_kv_bits_for_hw()` and use result. Only when KV quant is also auto-enabled (see E3). ≤ 30 LOC total.

Drift gate mandatory.

Snapshot all 3 hw before/after. Apply gates.

### E8 — ngram_K_per_workload

```bash
git checkout -b exp/E8-ngram-k-profile
```

Extend `vllm_mlx/agents/profiles/<name>.yaml` schema to accept optional `engine:` block:

```yaml
engine:
  ngram_max_k: 6  # codex / claude profiles only
```

In `vllm_mlx/cli.py` resolve loader, when serving via `--agent <name>`, override `args.ngram_num_draft_tokens` from profile if present.

Update `codex.yaml`, `openclaude.yaml`, `cline.yaml` with `ngram_max_k: 6`. Leave `generic.yaml` untouched.

Snapshot before (with `--agent codex` if supported) and after. Apply gates.

### E9 — nax_runtime_detect

```bash
git checkout -b exp/E9-metal4-detect
```

Create `vllm_mlx/runtime/metal4.py`:

```python
"""Metal 4 / Neural Accelerator runtime detection.

Pure observability. Does NOT enable kernels yet.
"""
from __future__ import annotations
import functools
import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def device_name() -> str:
    try:
        return mx.device_info().get("device_name", "")
    except Exception:
        return ""


@functools.lru_cache(maxsize=1)
def macos_version() -> tuple[int, int]:
    import platform
    parts = platform.mac_ver()[0].split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return (0, 0)


@functools.lru_cache(maxsize=1)
def metal4_available() -> bool:
    if os.environ.get("LIGHTNING_DISABLE_METAL4") == "1":
        return False
    major, minor = macos_version()
    if (major, minor) < (26, 2):
        return False
    return True  # MLX 0.30+ enables NAX internally when supported


@functools.lru_cache(maxsize=1)
def m5_nax_hint() -> bool:
    if not metal4_available():
        return False
    return "M5" in device_name()


def log_capabilities() -> None:
    logger.info(
        "metal4=%s m5_nax=%s device=%s macos=%s",
        metal4_available(), m5_nax_hint(), device_name(), macos_version()
    )
```

Wire into startup: call `log_capabilities()` in `vllm_mlx/server.py` startup. Surface in `vllm_mlx/tui.py` status panel. ≤ 50 LOC.

No bench impact expected (G1 ~ 0%). Keep if no regression.

### E10 — mlx_floor_bump

```bash
git checkout -b exp/E10-mlx-floor
```

Edit `pyproject.toml`: change `"mlx>=0.29.0"` to `"mlx>=0.30.0"`.

Re-lock:
```bash
uv lock
```

Drift gate mandatory (MLX internals changed).

Snapshot all 3 hw before/after. M5 expected to show biggest gain.

If `mlx-lm` is pinned to incompatible MLX, abort and document.

### E11 — dflash_sweep

```bash
git checkout -b exp/E11-dflash-tune
```

DFlash already shipped. Sweep DFlash adaptive bounds.

Find flags by `grep -nE "dflash|--adaptive" vllm_mlx/cli.py`. For each block size in {32, 64, 96, 128}, run sweep on a DFlash-enabled model. Pick winner.

Update default in `vllm_mlx/speculative/dflash_drafter.py` if winner != current default. ≤ 20 LOC.

### E12 — sampler_gpu_path

```bash
git checkout -b exp/E12-sampler-gpu
```

Audit `vllm_mlx/pipeline/decode.py` for any `.item()` / `.tolist()` / `np.asarray` inside the per-token loop. Each is a CPU sync.

Replace with `mx.eval(...)` + on-device sampling. Keep public API unchanged. ≤ 50 LOC.

Drift gate mandatory.

### E13 — mx_compile_coverage

```bash
git checkout -b exp/E13-mx-compile
```

Grep candidates:
```bash
grep -nE "^def " vllm_mlx/models/*.py | grep -v "@mx.compile"
```

For each helper that is pure (no I/O, no Python branches on tensor values, only mx ops), add `@mx.compile` decorator. Skip anything with `if x.item():` or list comprehensions over tensor elements.

Drift gate mandatory. ≤ 30 LOC additions.

---

## 4. Phase 3 — Larger, gated

### E14 — sliding_window_default_long_ctx

**Gate:** only run if agentic-long fixture exceeds 32k ctx in practice. Measure first.

```bash
git checkout -b exp/E14-sliding-window
```

`vllm_mlx/attention.py:154` already plumbs `sliding_window`. Wire default `sliding_window=16384` when ctx > 32k on agentic profiles. Gate behind config flag.

Quality bench mandatory — drift gate alone insufficient. Run gsm8k on `evals/gsm8k_qwen3_0.6b_results.json` reference.

### E15 — eagle3_evaluation

**Gate:** only if E1–E13 cumulative gain < +30% agentic-short on M5.

```bash
git checkout -b exp/E15-eagle3-spike
```

Spike only. Port one EAGLE-3 reference implementation to MLX for `qwen3.6-35b`. Time-box 5 days. If acceptance rate × draft length not > MTP+ngram, abandon and delete branch.

If kept: production hardening is a separate experiment (E15b). Plan separately.

### E16 — nax_kernel_port_t2

**Gate:** only if NAX T1 (E10 + E9 verify) leaves > 15% M5 gap vs antirez ds4 numbers.

See `todo/nax-metal4-plan.md` §5 T2 for full procedure. Implement via custom Metal4 kernel wrapping `mx.fast.metal_kernel`.

---

## 5. Skip indefinitely

Do NOT attempt without explicit user approval:
- ReDrafter — per-model trained heads, maintenance prohibitive solo.
- Full Metal4 fork (T3 from NAX plan).
- W4A4 activation quantization.
- Lookahead / Jacobi standalone decoding.

---

## 6. Maintenance ledger

Update after every KEEP. Halt Phase 2/3 entries when cumulative LOC > 200.

| id | sha | LOC added | flags added | est ongoing cost |
|----|-----|----------:|------------:|------------------|

---

## 7. Snapshot reference — sweep cell list

When asked to "snapshot before/after", run this matrix per hw:

| model | fixture | max_tokens | num_prompts | cell file |
|-------|---------|-----------:|------------:|-----------|
| qwen3.6-35b-nsc-ace-saber-8bit | agentic_short.txt | 256 | 5 | short.json |
| qwen3.6-35b-nsc-ace-saber-8bit | agentic_tool_heavy.txt | 512 | 3 | tool_heavy.json |
| qwen3.6-35b-nsc-ace-saber-8bit | agentic_long.txt | 2048 | 3 | long.json |
| qwen3.6-27b-8bit | agentic_short.txt | 256 | 5 | short.json |
| qwen3.6-27b-8bit | agentic_long.txt | 2048 | 3 | long.json |

The `agentic_sweep.sh` script already encodes this. Use it.

---

## 8. Verdict template

After each step, write `reports/exp/<id>/verdict.md`:

```markdown
# Verdict — <id> <slug>

## Snapshot
- Before: reports/exp/T0/<hw>/<model>
- After:  reports/exp/<id>/after/<hw>/<model>
- Diff summary: reports/exp/<id>/summary.md

## Gates
- G1 (agentic-short Δ% M5): <value>%  — <PASS|FAIL>
- G2 (agentic-long Δ% M5): <value>%  — <PASS|FAIL>
- G3 (M1/M3 regression): <value>% / <value>%  — <PASS|FAIL|SKIP>
- G4 (drift mean_abs): <value>  — <PASS|FAIL|N/A>
- G5 (acceptance Δpp): <value>  — <PASS|FAIL>
- G6 (LOC added): <value>  — <PASS|FAIL>

## Decision
<KEEP | REVERT>

## Reason
<one-paragraph rationale>

## Next
<next step id>
```

---

## 9. Hard reference — file paths

Common paths you will edit:

- `vllm_mlx/cli.py` — flag definitions, bench_command, alias resolution
- `vllm_mlx/scheduler.py` — `SchedulerConfig` fields
- `vllm_mlx/optimizations.py` — `HARDWARE_PROFILES`
- `vllm_mlx/agents/profiles/*.yaml` — per-agent config
- `vllm_mlx/aliases.json` — model alias map
- `vllm_mlx/server.py` — startup hooks
- `vllm_mlx/runtime/metal4.py` — NEW, capability detection (E9)
- `vllm_mlx/speculative/dflash_drafter.py` — DFlash params
- `vllm_mlx/pipeline/decode.py` — per-token loop
- `pyproject.toml` — dep versions
- `uv.lock` — re-lock after dep changes
- `reports/exp/INDEX.md` — append-only log
- `reports/exp/<id>/` — per-step artifacts
- `scripts/agentic_sweep.sh` — bench fixture runner
- `scripts/nax_summarize.py` — delta summary
- `evals/nax_drift_gate.py` — logit drift capture
- `evals/nax_drift_compare.py` — drift comparator
- `evals/fixtures/agentic_*.txt` — bench prompts

---

## 10. Next action (start here)

1. Read §0 fully.
2. Execute Phase 0 (T0.1 → T0.7).
3. Capture T0 baseline (§2 T1.0) on whatever hw you have access to.
4. Run E1. Follow gates.
5. Continue serially through queue.
