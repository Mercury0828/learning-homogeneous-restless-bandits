"""Summarize the 16-kernel sweep: cumulative per-arm regret at T for every placement, N and planner,
with 95% half-widths over the runs, next to the frozen-model baseline. Reads out/dirsweep/<code>/ and
writes out/dirsweep/summary_T{T}.csv. All placements are reported; nothing is filtered.
Usage: python summarize_dirsweep.py [T]"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "out", "dirsweep")
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rows = []
for code in sorted(c for c in os.listdir(D) if os.path.isdir(os.path.join(D, c))):
    ff = os.path.join(D, code, f"frozen_baseline_T{T}.csv")   # the frozen baseline file is optional
    frozen = ({(int(r["N"]), r["planner"]): float(r["frozen_cumulative"]) for r in csv.DictReader(open(ff))}
              if os.path.exists(ff) else {})
    for N in (50, 200, 800):
        fn = os.path.join(D, code, f"online_regret_N{N}_T{T}.csv")
        if not os.path.exists(fn):
            continue
        for r in csv.DictReader(open(fn)):
            if int(r["T"]) == T:
                rows.append(dict(placement=code, N=N, planner=r["planner"], mean=float(r["mean"]),
                                 ci95=float(r["ci95"]), frozen=frozen.get((N, r["planner"]), float("nan"))))
with open(os.path.join(D, f"summary_T{T}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['placement']} N={r['N']:4d} {r['planner']:8s} learner {r['mean']:8.4f} ± {r['ci95']:.4f}   frozen {r['frozen']:8.4f}")
