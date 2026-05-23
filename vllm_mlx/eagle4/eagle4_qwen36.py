"""EAGLE-4 Head for Qwen3.6-35B-A3B — adapted from joshuahickscorp/eagle4 (Apache 2.0).

Provides: EagleHead, extract_frozen, train, save_ckpt, load_ckpt, build_head.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as mxoptim
import numpy as np
import pyarrow.parquet as pq

from eagle4_qwen36_config import (
    HIDDEN_DIM, VOCAB, N_MOE_LAYERS, N_ROUTED, TOP_K,
    N_HEADS, INTERMEDIATE, RMS_EPS, HEAD_DIM, N_KV_HEADS,
    LR, BATCH_SIZE, SEQ_LEN, WARMUP_STEPS, ALPHA_INIT, N_RECORDS
)

# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
class _SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN_DIM, INTERMEDIATE, bias=False)
        self.up = nn.Linear(HIDDEN_DIM, INTERMEDIATE, bias=False)
        self.down = nn.Linear(INTERMEDIATE, HIDDEN_DIM, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = mx.ones((HIDDEN_DIM,))
        self.attn = nn.MultiHeadAttention(HIDDEN_DIM, N_HEADS, bias=False)
        self.mlp_norm = mx.ones((HIDDEN_DIM,))
        self.mlp = _SwiGLU()

    def __call__(self, x, mask):
        h = mx.fast.rms_norm(x, self.attn_norm, RMS_EPS)
        x = x + self.attn(h, h, h, mask=mask)
        h = mx.fast.rms_norm(x, self.mlp_norm, RMS_EPS)
        x = x + self.mlp(h)
        return x


class EagleHead(nn.Module):
    def __init__(self, token_embd, lm_head, output_norm):
        super().__init__()
        self._token_embd = token_embd
        self._lm_head = lm_head
        self._output_norm = output_norm
        self.in_proj = nn.Linear(5 * HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.block = _Block()
        self.residual_gate = mx.array([ALPHA_INIT])
        # Mask head: MoE routing prediction per layer
        self.mask_proj_in = nn.Linear(HIDDEN_DIM, 512, bias=False)
        self.mask_proj_out = nn.Linear(512, N_MOE_LAYERS * N_ROUTED, bias=False)
        self.calib_proj = nn.Linear(HIDDEN_DIM, 1, bias=True)

    def trainable_parameters(self):
        p = self.parameters()
        for k in ("_token_embd", "_lm_head", "_output_norm"):
            p.pop(k, None)
        return p

    def __call__(self, prev_tok, h_low, h_mid, h_high, h_shared):
        B, S = prev_tok.shape
        attn_mask = (mx.eye(S) - 1.0) * 1e9
        embed_table = mx.transpose(self._token_embd, (1, 0))
        prev_embed = embed_table[prev_tok]
        x = mx.concatenate([prev_embed, h_low, h_mid, h_high, h_shared], axis=-1)
        x = self.in_proj(x)
        x = self.block(x, attn_mask)
        baseline = mx.fast.rms_norm(h_high, self._output_norm, RMS_EPS)
        draft_hidden = baseline.astype(x.dtype) + self.residual_gate * x
        token_logits = draft_hidden @ self._lm_head
        mask_logits = self.mask_proj_out(nn.silu(self.mask_proj_in(draft_hidden)))
        mask_logits = mask_logits.reshape(B, S, N_MOE_LAYERS, N_ROUTED)
        calib_logit = self.calib_proj(draft_hidden).squeeze(-1)
        return token_logits, mask_logits, draft_hidden, calib_logit


# ---------------------------------------------------------------------------
# Freeze extraction
# ---------------------------------------------------------------------------
def extract_frozen(model_path: str, out: Path):
    """Extract token_embd, lm_head, output_norm from frozen Qwen3.6 model.

    Handles both quantized (scales/biases) and unquantized weights.
    For quantized models, extracts the raw weight (with scales/biases applied
    at runtime by mlx-lm's dequantization).
    """
    from mlx_lm.utils import load
    import mlx.core as mx
    print(f"[extract] loading {model_path}", flush=True)
    model, _ = load(model_path)

    # Walk parameters to find the right paths
    def find(p, path=""):
        if isinstance(p, dict):
            for k, v in p.items():
                if isinstance(v, (dict, mx.array)):
                    yield from find(v, f"{path}.{k}" if path else k)
        elif isinstance(p, mx.array):
            yield (path, p)

    params = dict(find(model.parameters()))
    print(f"[extract] found {len(params)} parameters", flush=True)

    # Find token embeddings, lm_head, output norm
    token_embd = None
    lm_head = None
    output_norm = None
    for k, v in params.items():
        if k.endswith("embed_tokens.weight"):
            token_embd = v
        elif k.endswith("lm_head.weight"):
            lm_head = v
        elif k.endswith("norm.weight"):
            output_norm = v

    if token_embd is None:
        raise RuntimeError("token_embd not found in model parameters")
    if lm_head is None:
        raise RuntimeError("lm_head not found in model parameters")
    if output_norm is None:
        raise RuntimeError("output_norm not found in model parameters")

    np.savez(out,
             token_embd=np.array(token_embd),
             lm_head=np.array(lm_head),
             output_norm=np.array(output_norm))
    print(f"[extract] wrote {out} (token_embd={token_embd.shape}, lm_head={lm_head.shape}, output_norm={output_norm.shape})", flush=True)


def build_head(frozen_npz: Path) -> EagleHead:
    z = np.load(frozen_npz)
    return EagleHead(mx.array(z["token_embd"]), mx.array(z["lm_head"]), mx.array(z["output_norm"]))


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def _flat_params(d, prefix=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else enumerate(d)):
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, (dict, list)):
            out.update(_flat_params(v, key))
        elif hasattr(v, "shape"):
            out[key] = v
    return out


def save_ckpt(head: EagleHead, path: Path, step: int = 0):
    flat = {k: np.array(v) for k, v in _flat_params(head.trainable_parameters()).items()}
    flat["__step__"] = np.int32(step)
    np.savez(path, **flat)


def load_ckpt(head: EagleHead, path: Path) -> int:
    z = np.load(path, allow_pickle=False)
    params = head.trainable_parameters()
    def walk(d, prefix=""):
        if isinstance(d, dict):
            for k in list(d.keys()):
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(d[k], (dict, list)):
                    walk(d[k], key)
                elif key in z:
                    d[k] = mx.array(z[key])
        elif isinstance(d, list):
            for i in range(len(d)):
                walk(d[i], f"{prefix}.{i}")
    walk(params)
    head.update(params)
    return int(z.get("__step__", -1))


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def _iter_batches(shards: list[Path], batch_size: int, seq_len: int, epochs: int, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for s in shards:
        t = pq.read_table(s)
        for i in range(t.num_rows):
            rows.append({k: t[k][i].as_py() for k in t.column_names})
    print(f"[data] loaded {len(rows)} records from {len(shards)} shard(s)", flush=True)

    rows.sort(key=lambda r: (r["sample_id"], r["position"]))
    windows: list[list[dict]] = []
    cur_sid = None
    cur: list[dict] = []
    for r in rows:
        if r["sample_id"] != cur_sid:
            cur_sid = r["sample_id"]
            cur = []
        cur.append(r)
        if len(cur) == seq_len:
            windows.append(cur)
            cur = []
    print(f"[data] {len(windows)} ordered windows of len {seq_len}", flush=True)

    for epoch in range(epochs):
        rng.shuffle(windows)
        for i in range(0, len(windows) - batch_size + 1, batch_size):
            batch = windows[i:i + batch_size]
            flat = [r for w in batch for r in w]
            prev = np.array([r["prev_token"] for r in flat], dtype=np.int32).reshape(batch_size, seq_len)
            nxt = np.array([r["next_token"] for r in flat], dtype=np.int32).reshape(batch_size, seq_len)

            def stack(field, dtype):
                return np.frombuffer(b"".join(r[field] for r in flat), dtype=dtype).reshape(batch_size, seq_len, -1)

            yield {
                "prev": mx.array(prev),
                "next": mx.array(nxt),
                "low": mx.array(stack("hidden_low", np.float16)).astype(mx.float32),
                "mid": mx.array(stack("hidden_mid", np.float16)).astype(mx.float32),
                "high": mx.array(stack("hidden_high", np.float16)).astype(mx.float32),
                "shared": mx.array(stack("shared_hidden", np.float16)).astype(mx.float32),
                "mask": mx.array(
                    stack("routed_mask_per_layer", np.uint8)
                    .reshape(batch_size, seq_len, N_MOE_LAYERS, N_ROUTED)
                    .astype(np.float32)
                ),
                "epoch": epoch,
            }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(parquet_paths, frozen, ckpt_dir, epochs=1, batch_size=BATCH_SIZE,
          seq_len=SEQ_LEN, lr=LR, aux_weight=0.5, mask_weight=0.3, calib_weight=0.1,
          multi_step_k=1, multi_step_decay=0.7, target_argmax_warmup_steps=WARMUP_STEPS):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    head = build_head(Path(frozen))
    parquet_paths = [Path(p) for p in parquet_paths]
    V = VOCAB
    print(f"[train] head built, V={V} aux={aux_weight} mask={mask_weight} calib={calib_weight}", flush=True)

    def _step_loss(head, b, prev_tok, weight, target_alpha):
        tok, mask_logits, draft_h, calib_logit = head(prev_tok, b["low"], b["mid"], b["high"], b["shared"])
        B, S = prev_tok.shape
        V2 = tok.shape[-1]
        pos_mask = mx.concatenate([mx.zeros((B, 3)), mx.ones((B, S - 3))], axis=1).reshape(-1)
        N = mx.maximum(pos_mask.sum(), mx.array(1.0))

        baseline = mx.fast.rms_norm(b["high"], head._output_norm, RMS_EPS)
        target_logits = baseline @ head._lm_head.astype(mx.float32)
        target_arg_flat = mx.stop_gradient(mx.argmax(target_logits.reshape(-1, V2), axis=-1))
        corpus_tok = b["next"].reshape(-1)
        tok_flat = tok.reshape(-1, V2)

        ce_corpus = nn.losses.cross_entropy(tok_flat, corpus_tok, reduction="none")
        ce_target_argmax = nn.losses.cross_entropy(tok_flat, target_arg_flat, reduction="none")
        ce_hybrid = (1.0 - target_alpha) * ce_corpus + target_alpha * ce_target_argmax
        token_loss = weight * (ce_hybrid * pos_mask).sum() / N

        aux_loss = 0.0
        if aux_weight > 0:
            base = baseline.reshape(-1, HIDDEN_DIM)
            dh = draft_h.reshape(-1, HIDDEN_DIM)
            aux_loss = aux_weight * ((base - dh) ** 2).sum() / (N * HIDDEN_DIM)

        mask_loss = 0.0
        if mask_weight > 0:
            m = mask_logits.reshape(-1, N_MOE_LAYERS * N_ROUTED)
            t = b["mask"].reshape(-1, N_MOE_LAYERS * N_ROUTED)
            mask_loss = mask_weight * nn.losses.binary_cross_entropy_with_logits(m, t, reduction="mean")

        calib_loss = 0.0
        if calib_weight > 0:
            head_arg = mx.argmax(tok_flat.reshape(B, S, V2), axis=-1).reshape(-1)
            target_arg = target_arg_flat.reshape(B, S)
            accepted = (head_arg.reshape(B, S) == target_arg).astype(mx.float32).reshape(-1)
            calib_loss = calib_weight * nn.losses.binary_cross_entropy_with_logits(
                calib_logit.reshape(-1), accepted, reduction="mean")

        return token_loss + aux_loss + mask_loss + calib_loss

    loss_and_grad = nn.value_and_grad(head, _step_loss)
    optimizer = mxoptim.AdamW(learning_rate=lr)
    state = [optimizer.init(head.trainable_parameters())]

    total_steps = 0
    t0 = time.time()
    best_loss = float("inf")

    for batch in _iter_batches(parquet_paths, batch_size, seq_len, epochs):
        total_steps += 1
        target_alpha = min(1.0, total_steps / max(target_argmax_warmup_steps, 1))
        prev_tok = batch["prev"]

        # Multi-step rollout
        for k in range(multi_step_k):
            weight = multi_step_decay ** k if multi_step_k > 1 else 1.0
            loss, grads = loss_and_grad(head, batch, prev_tok, weight, target_alpha)
            optimizer.update(head.trainable_parameters(), grads, state)
            mx.eval(head.trainable_parameters(), state)
            if multi_step_k > 1 and k < multi_step_k - 1:
                tok, _, _, _ = head(prev_tok, batch["low"], batch["mid"], batch["high"], batch["shared"])
                prev_tok = mx.argmax(tok, axis=-1)

        if total_steps % 50 == 0:
            elapsed = time.time() - t0
            rate = total_steps / max(elapsed, 1e-3)
            print(f"[train] step={total_steps} loss={loss.item():.4f} α={target_alpha:.2f} {rate:.0f} step/s", flush=True)
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_ckpt(head, ckpt_dir / "best.npz", total_steps)

    save_ckpt(head, ckpt_dir / "final.npz", total_steps)
    elapsed = time.time() - t0
    print(f"[train] done: {total_steps} steps in {elapsed:.0f}s, best_loss={best_loss:.4f}", flush=True)
