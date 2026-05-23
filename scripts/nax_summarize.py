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
