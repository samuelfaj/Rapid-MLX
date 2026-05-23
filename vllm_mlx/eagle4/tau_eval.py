"""τ-at-depth-K evaluation for EAGLE-4 head on Qwen3.6.

Adapted from joshuahickscorp/eagle4/tau_eval.py (Apache 2.0).
Measures accepted prefix length under autoregressive rollout.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq

from eagle4_qwen36_config import HIDDEN_DIM, N_MOE_LAYERS, N_ROUTED


def tau_at_depth_k(ckpt_path: str, parquet_paths: list[str], depth: int = 4,
                   n_samples: int = 500, seed: int = 0):
    """Evaluate τ-at-depth-K from saved checkpoint and captured parquet.

    Returns: mean accepted prefix length at given depth.
    """
    from mlx_lm.utils import load
    from eagle4_qwen36 import build_head

    # Build head from checkpoint
    head = build_head(Path(ckpt_path))

    # Load evaluation data
    rows = []
    for s in parquet_paths:
        t = pq.read_table(s)
        for i in range(t.num_rows):
            rows.append({k: t[k][i].as_py() for k in t.column_names})

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_samples]

    total_accepted = 0
    total_eval = 0

    for r in rows:
        prev = mx.array([[r["prev_token"]]], dtype=mx.int32)
        low = mx.array([np.frombuffer(r["hidden_low"], dtype=np.float16).reshape(1, 1, -1)], dtype=mx.float32)
        mid = mx.array([np.frombuffer(r["hidden_mid"], dtype=np.float16).reshape(1, 1, -1)], dtype=mx.float32)
        high = mx.array([np.frombuffer(r["hidden_high"], dtype=np.float16).reshape(1, 1, -1)], dtype=mx.float32)
        shared = mx.array([np.frombuffer(r["shared_hidden"], dtype=np.float16).reshape(1, 1, -1)], dtype=mx.float32)
        target_tok = mx.array([[r["next_token"]]], dtype=mx.int32)

        accepted = 0
        for d in range(depth):
            token_logits, _, _, _ = head(prev, low, mid, high, shared)
            pred = mx.argmax(token_logits, axis=-1)
            if pred.item() == target_tok.item():
                accepted += 1
                prev = pred  # feed own argmax as next input
            else:
                break

        total_accepted += accepted
        total_eval += 1

    tau = total_accepted / max(total_eval, 1)
    print(f"[tau_eval] depth={depth} n={total_eval} tau={tau:.3f}", flush=True)
    return tau


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .npz checkpoint")
    p.add_argument("--parquet-dir", required=True, help="Directory with parquet shards")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=500)
    args = p.parse_args()

    import glob
    shards = sorted(glob.glob(f"{args.parquet_dir}/shard_*.parquet"))
    tau_at_depth_k(args.ckpt, shards, args.depth, args.n_samples)
