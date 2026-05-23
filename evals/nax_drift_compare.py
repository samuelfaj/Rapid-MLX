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
