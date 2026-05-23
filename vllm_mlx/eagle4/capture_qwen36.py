"""EAGLE-4 capture for Qwen3.6-35B-A3B on MLX.

Adapted from joshuahickscorp/eagle4 (Apache 2.0).
Captures per-token hidden states + MoE router logits from frozen target model.

Usage:
    python -m vllm_mlx.eagle4.capture_qwen36 --out-dir data/captures --n-records 1000000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load

from eagle4_qwen36_config import (
    FUSION_LAYERS, N_MOE_LAYERS, N_ROUTED, TOP_K, HIDDEN_DIM, SHARD_ROWS
)

# ---------- Schema ----------
SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("position", pa.int32()),
    ("prev_token", pa.int32()),
    ("next_token", pa.int32()),
    ("hidden_low", pa.binary()),
    ("hidden_mid", pa.binary()),
    ("hidden_high", pa.binary()),
    ("shared_hidden", pa.binary()),
    ("router_logits_per_layer", pa.binary()),
    ("routed_mask_per_layer", pa.binary()),
])


def _install_hooks(model):
    """Wrap each MoE layer to capture (mlp_input, gate_logits).

    Qwen3.6 uses Qwen3NextSparseMoeBlock with:
      - gate: Linear(hidden_dim, num_experts) — but Qwen3.6 gate is (256, 512)
      - shared_experts: shared FFN
      - experts: SwitchMLP or similar

    The gate weight shape (num_experts, intermediate_dim) where intermediate_dim
    may differ from hidden_dim. We capture the raw gate logits.
    """
    buf: dict[int, tuple[mx.array, mx.array]] = {}

    class Hooked:
        def __init__(self, real, idx):
            self._real = real
            self._idx = idx
            # Pass through all attributes
            self.gate = real.gate
            self.shared_experts = getattr(real, "shared_experts", None)
            self.experts = getattr(real, "experts", real.switch_mlp if hasattr(real, "switch_mlp") else None)
            self.num_experts_per_tok = getattr(real, "num_experts_per_tok", TOP_K)

        def __call__(self, x):
            # x shape: (batch, seq, hidden_dim) = (1, seq, 2048)
            # gate weight: (num_experts, gate_input_dim) — Qwen3.6 is (256, 512)
            # The gate internally projects hidden_dim -> gate_input_dim
            gate_in = x  # save the input to mlp
            gate_logits = self._real.gate(x)  # call the real gate
            buf[self._idx] = (gate_in, gate_logits)
            return self._real(x)

    moe_idx = 0
    layers = model.language_model.layers if hasattr(model, 'language_model') else model.layers
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        if hasattr(mlp, 'gate'):
            layer.mlp = Hooked(mlp, moe_idx)
            moe_idx += 1

    print(f"[capture] hooks installed on {moe_idx} MoE layers", flush=True)
    return buf


def _format_msgs(tokenizer, messages, max_ctx):
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    except Exception:
        return None
    if len(ids) < 8:
        return None
    return ids[:max_ctx]


def capture(out_dir: Path, n_records: int, skip_n: int, max_ctx: int,
            model_path: str, dataset_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] loading {model_path}", flush=True)
    model, tok = load(model_path)
    inner = model.language_model if hasattr(model, 'language_model') else model
    layers = inner.layers
    capture_buf = _install_hooks(model)
    last_moe = max(capture_buf.keys()) if capture_buf else 37

    ds = load_dataset(dataset_id, split="train_sft", streaming=True)

    rows: list[dict] = []
    shard = 0
    total = 0
    convs = 0
    t0 = time.time()
    for idx, sample in enumerate(ds):
        if idx < skip_n:
            continue
        if total >= n_records:
            break
        msgs = sample.get("messages")
        if not msgs:
            continue
        ids = _format_msgs(tok, msgs, max_ctx)
        if ids is None:
            continue
        input_ids = mx.array([ids], dtype=mx.int32)
        seq_len = len(ids)

        # Forward pass through layers, capturing fusion points
        h = inner.embed_tokens(input_ids) if hasattr(inner, 'embed_tokens') else layers[0](input_ids)
        if hasattr(inner, 'embed_tokens'):
            attn_mask = create_attention_mask(h, None)

        fusion: dict[int, mx.array] = {}
        for li, layer in enumerate(layers):
            if hasattr(inner, 'embed_tokens'):
                h = layer(h, attn_mask, None)
            else:
                h = layer(h, None, None)
            if li in FUSION_LAYERS:
                fusion[li] = h

        low = fusion[FUSION_LAYERS[0]][0].astype(mx.float16)
        mid = fusion[FUSION_LAYERS[1]][0].astype(mx.float16)
        hi = fusion[FUSION_LAYERS[2]][0].astype(mx.float16)

        # Shared expert output from last MoE layer
        last_in, _ = capture_buf[last_moe]
        last_mlp = layers[last_moe].mlp
        if hasattr(last_mlp, 'shared_experts') and last_mlp.shared_experts:
            shared_h = last_mlp.shared_experts(last_in)[0].astype(mx.float16)
        else:
            shared_h = mx.zeros_like(hi)

        # Router logits: stack per-layer gate logits
        rl_list = []
        for mi in range(len(capture_buf)):
            _, g_logits = capture_buf[mi]
            rl_list.append(g_logits[0].astype(mx.float16))
        rl = mx.stack(rl_list, axis=0)
        mx.eval(low, mid, hi, shared_h, rl)

        # Compute routed mask (top-K experts per layer per token)
        rl_np = np.array(rl)
        topk = np.argpartition(-rl_np, kth=TOP_K - 1, axis=-1)[..., :TOP_K]
        mask_np = np.zeros((N_MOE_LAYERS, seq_len, N_ROUTED), dtype=np.uint8)
        np.put_along_axis(mask_np, topk, 1, axis=-1)

        low_b = bytes(memoryview(low))
        mid_b = bytes(memoryview(mid))
        hi_b = bytes(memoryview(hi))
        sh_b = bytes(memoryview(shared_h))
        Hb = HIDDEN_DIM * 2

        sample_id = f"sample_{idx}"
        for pos in range(seq_len - 1):
            rows.append({
                "sample_id": sample_id,
                "position": int(pos),
                "prev_token": int(ids[pos]),
                "next_token": int(ids[pos + 1]),
                "hidden_low": low_b[pos * Hb:(pos + 1) * Hb],
                "hidden_mid": mid_b[pos * Hb:(pos + 1) * Hb],
                "hidden_high": hi_b[pos * Hb:(pos + 1) * Hb],
                "shared_hidden": sh_b[pos * Hb:(pos + 1) * Hb],
                "router_logits_per_layer": rl_np[:, pos, :].astype(np.float16).tobytes(),
                "routed_mask_per_layer": mask_np[:, pos, :].tobytes(),
            })
            total += 1
            if total >= n_records:
                break

        while len(rows) >= SHARD_ROWS:
            path = out_dir / f"shard_{shard:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows[:SHARD_ROWS], schema=SCHEMA),
                           path, compression="zstd")
            print(f"[capture] wrote {path.name} ({SHARD_ROWS} rows)", flush=True)
            rows = rows[SHARD_ROWS:]
            shard += 1

        convs += 1
        if convs % 5 == 0:
            elapsed = time.time() - t0
            rate = total / max(elapsed, 1e-3)
            eta = (n_records - total) / max(rate, 1e-3) / 60
            print(f"[capture] conv={convs} rec={total}/{n_records} {rate:.0f} rec/s eta={eta:.1f}m", flush=True)

    if rows:
        path = out_dir / f"shard_{shard:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")
        shard += 1

    elapsed = time.time() - t0
    print(f"[capture] done: {total} records in {shard} shards ({elapsed:.1f}s, {total/max(elapsed,1):.0f} rec/s)", flush=True)


def main():
    p = argparse.ArgumentParser(prog="eagle4-capture-qwen36")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-records", type=int, default=1_000_000)
    p.add_argument("--skip-n", type=int, default=0)
    p.add_argument("--max-ctx", type=int, default=256)
    p.add_argument("--model", default="/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-NSC-ACE-SABER-MLX-8bit-MTPLX-Optimized-Speed")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    args = p.parse_args()
    capture(args.out_dir, args.n_records, args.skip_n, args.max_ctx, args.model, args.dataset)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
