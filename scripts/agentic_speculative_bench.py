#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark agentic speculative policy choices against a running server."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "create a REST api using express and bun and typescript and "
    "sequelize-typescript. It must be vertical sliced. You should create models, "
    "seeders and migrations. You must create unit tests for each service."
)

PROFILES = {
    "target": "target-only + prefix cache",
    "adaptive": "drafter + auto adaptive",
    "forced-ddtree-4": "forced DDTree budget 4",
    "adaptive-budget": "adaptive DDTree budget sweep",
    "ngram-long-text": "n-gram enabled for long text only",
}

WORKLOADS = {
    "initial-scaffold": DEFAULT_PROMPT,
    "long-code": "Generate a large TypeScript service module with repositories, DTOs, validation, and unit tests.",
    "tool-loop": "Inspect the project, list files, then make the smallest safe change needed.",
    "repair": "Fix the latest failing TypeScript unit test and explain the exact failure.",
    "validation": "Run validation, summarize pass/fail status, and return final next action.",
}


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _last_request(status: dict[str, Any]) -> dict[str, Any]:
    entries = status.get("requests") or []
    if entries:
        return dict(entries[-1])
    dflash = status.get("dflash") or {}
    history = dflash.get("agentic_policy_history") or []
    return dict(history[-1]) if history else {}


def _path_from_request(row: dict[str, Any]) -> str:
    mode = str(row.get("spec_mode") or row.get("mode") or "-")
    proposed = int(row.get("speculative_proposed_tokens") or row.get("proposed_tokens") or 0)
    steps = int(row.get("speculative_steps") or 0)
    ngram_cycles = int(row.get("ngram_cycles") or 0)
    if mode in {"target-fallback", "target-prefix-cache"}:
        return "-"
    if mode == "ddtree-ngram" and ngram_cycles > 0 and proposed > 0:
        return "ng+tree"
    if mode == "ddtree-ngram" and ngram_cycles > 0:
        return "ngram"
    if mode in {"ddtree", "ddtree-ngram"} and (proposed > 0 or steps > 0):
        return "ddtree"
    return "-"


def _recommend(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no data"
    by_path: dict[str, list[float]] = {}
    for row in rows:
        path = str(row["path"])
        tps = float(row.get("effective_tps") or row.get("tokens_per_second") or 0.0)
        if tps > 0:
            by_path.setdefault(path, []).append(tps)
    if not by_path:
        return "target-only: insufficient throughput data"
    winners = {
        path: statistics.mean(values)
        for path, values in by_path.items()
        if values
    }
    best = max(winners, key=winners.get)
    if best == "-":
        return "use target-only/prefix-cache for this workload"
    return f"use adaptive speculative path; current winner={best}"


def _mock_rows(count: int, profiles: list[str], workloads: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for profile in profiles:
        for workload in workloads:
            for _ in range(count):
                path = "-" if index % 3 else "ddtree"
                rows.append(
                    {
                        "index": index,
                        "profile": profile,
                        "workload": workload,
                        "path": path,
                        "wall_s": 1.0 + index / 10,
                        "effective_tps": 80.0 if path == "-" else 55.0,
                        "acceptance_length": 0.0 if path == "-" else 3.5,
                        "acceptance_ratio": 0.0 if path == "-" else 0.4,
                        "tree_budget": 0 if path == "-" else 4,
                    }
                )
                index += 1
    return rows


def _select_names(value: str, choices: dict[str, str]) -> list[str]:
    if value == "all":
        return list(choices)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in choices]
    if unknown:
        raise ValueError(f"unknown names: {', '.join(unknown)}")
    return names


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    profiles = _select_names(args.profile, PROFILES)
    workloads = _select_names(args.workload, WORKLOADS)

    if args.mock:
        rows = _mock_rows(args.count, profiles, workloads)
    else:
        rows = []
        index = 0
        for profile in profiles:
            for workload in workloads:
                prompt = args.prompt if args.prompt != DEFAULT_PROMPT else WORKLOADS[workload]
                for _ in range(args.count):
                    payload = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": args.max_tokens,
                    }
                    started = time.perf_counter()
                    completion = _post_json(
                        f"{args.base_url.rstrip('/')}/v1/chat/completions",
                        payload,
                        args.timeout,
                    )
                    wall_s = time.perf_counter() - started
                    status = _get_json(
                        f"{args.base_url.rstrip('/')}/v1/status",
                        args.timeout,
                    )
                    request_row = _last_request(status)
                    dflash = status.get("dflash") or {}
                    history = dflash.get("agentic_policy_history") or []
                    policy_row = dict(history[-1]) if history else {}
                    text = (
                        completion.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    row = {
                        "index": index,
                        "profile": profile,
                        "profile_description": PROFILES[profile],
                        "workload": workload,
                        "wall_s": wall_s,
                        "path": _path_from_request(request_row | policy_row),
                        "output_chars": len(text),
                        "finish_reason": completion.get("choices", [{}])[0].get(
                            "finish_reason"
                        ),
                        "request": request_row,
                        "policy": policy_row,
                        "effective_tps": policy_row.get("effective_tps"),
                        "decode_tps": policy_row.get("generation_tps"),
                        "acceptance_length": policy_row.get("acceptance_length"),
                        "acceptance_ratio": policy_row.get("acceptance_ratio"),
                        "tree_budget": policy_row.get("tree_budget"),
                    }
                    rows.append(row)
                    index += 1

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "requests": len(rows),
        "profiles": profiles,
        "workloads": workloads,
        "paths": {path: sum(1 for row in rows if row["path"] == path) for path in sorted({row["path"] for row in rows})},
        "mean_wall_s": statistics.mean(row["wall_s"] for row in rows) if rows else 0.0,
        "recommendation": _recommend(rows),
        "jsonl": str(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="local")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--profile", default="adaptive", help="Profile name, comma list, or all")
    parser.add_argument("--workload", default="initial-scaffold", help="Workload name, comma list, or all")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", default="reports/benchmarks/agentic-speculative.jsonl")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except urllib.error.URLError as exc:
        print(f"server_error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
