#!/usr/bin/env python3
"""Clean AR-vs-DDTree decode benchmark against running OpenAI-compatible servers."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx


PROMPTS = {
    "prose": "Write a concise technical explanation of TCP congestion control.",
    "crud": (
        "Create a TypeScript function that validates a CRUD JSON payload with "
        "id, name, status, tags, createdAt, updatedAt, and nested metadata."
    ),
    "agent": (
        "Plan implementation steps for a React TypeScript snake game using "
        "feature-oriented architecture and test driven development."
    ),
}


def _detect_model(base_url: str) -> str:
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{base_url.rstrip('/')}/models")
        response.raise_for_status()
        data = response.json()
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models returned by {base_url}")
    return str(models[0]["id"])


def _completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=None) as client:
        response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    elapsed = time.perf_counter() - started
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "decode_tps": completion_tokens / elapsed if elapsed > 0 else 0.0,
        "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
    }


def run_suite(
    *,
    label: str,
    base_url: str,
    runs: int,
    max_tokens: int,
) -> dict[str, Any]:
    model = _detect_model(base_url)
    tests: dict[str, Any] = {}
    for name, prompt in PROMPTS.items():
        samples = [
            _completion(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            for _ in range(runs)
        ]
        tests[name] = {
            "samples": samples,
            "avg_decode_tps": mean(sample["decode_tps"] for sample in samples),
            "avg_completion_tokens": mean(
                sample["completion_tokens"] for sample in samples
            ),
        }
    return {
        "label": label,
        "url": base_url,
        "model": model,
        "temperature": 0,
        "stream": False,
        "max_tokens": max_tokens,
        "runs": runs,
        "tests": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark target-only AR and DDTree servers with the same prompts."
    )
    parser.add_argument("--ar-url", required=True, help="Target-only server /v1 base URL")
    parser.add_argument("--ddtree-url", required=True, help="DDTree server /v1 base URL")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suites": [
            run_suite(
                label="ar",
                base_url=args.ar_url,
                runs=max(1, args.runs),
                max_tokens=max(1, args.max_tokens),
            ),
            run_suite(
                label="ddtree",
                base_url=args.ddtree_url,
                runs=max(1, args.runs),
                max_tokens=max(1, args.max_tokens),
            ),
        ],
    }

    output = (
        Path(args.output)
        if args.output
        else Path("reports")
        / f"ar_vs_ddtree_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for suite in result["suites"]:
        print(f"{suite['label']} {suite['model']}")
        for name, data in suite["tests"].items():
            print(f"  {name}: {data['avg_decode_tps']:.2f} tok/s")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
