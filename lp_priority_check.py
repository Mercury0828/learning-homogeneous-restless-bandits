"""Baseline check: is the LP-priority learner covered by the fluid
re-solve learner on the d = H = 2 numerical instance?

At reset half the arms are in each state and the budget is B = N/2, so a priority (index) rule fills the whole
budget from one state: it activates c0 = B state-0 arms or c0 = 0, whichever state ranks first. The fluid re-solve
rule plays the rounded LP allocation. For every placement and N this script computes, with the true kernel, the
exact per-episode gap W - V of the fluid action and of both priority corners (exact binomial convolution, no Monte
Carlo). A learner's per-episode loss tends to this known-kernel gap, so a larger gap means a larger regret slope.
Writes out/dirsweep/lp_priority_check.csv. Usage: python lp_priority_check.py"""
import csv
import itertools
import os

import online_regret as o
from rounding import round_active

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for N in (50, 200, 800):
    B = N // 2
    for code in ["".join(c) for c in itertools.product("pm", repeat=4)]:
        th = o.placement(code)
        u = o.fluid_u(B / N, B / N, th)
        W = o.b_val(N, B, o.exact_action(N, B, th), th)
        cf = round_active(N, B, u)
        rows.append(dict(placement=code, N=N, fluid_u=u, gap_fluid=W - o.b_val(N, B, cf, th),
                         gap_priority_c0=W - o.b_val(N, B, 0, th), gap_priority_cB=W - o.b_val(N, B, B, th)))
out = os.path.join(HERE, "out", "dirsweep", "lp_priority_check.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
ratio = min(min(r["gap_priority_c0"], r["gap_priority_cB"]) / r["gap_fluid"] for r in rows)
print(f"{len(rows)} cases; max fluid gap {max(r['gap_fluid'] for r in rows):.2e}; "
      f"min priority gap {min(min(r['gap_priority_c0'], r['gap_priority_cB']) for r in rows):.2e}; "
      f"min ratio priority/fluid {ratio:.2f}")
