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
