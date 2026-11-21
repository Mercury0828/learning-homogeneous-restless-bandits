"""Diagnosis of the sweep: why does the fluid learner beat the exact learner at T = 2000
on some placements? For every placement and N: the known-kernel per-episode gap of each planner at the truth
(what it pays per episode once it has learned the truth), each learner's mean regret per episode over episodes
1601-2000 (from the cumulative checkpoints), and, where the fluid learner is ahead at T, the episode count at which
the exact learner would overtake it if both kept their last-400-episode slopes. Writes
out/dirsweep/diagnosis_T2000.csv. Reads only existing outputs plus exact finite-population values."""
import csv
import os

import numpy as np

import online_regret as o
from rounding import round_active

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "out", "dirsweep")
import sys

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2000           # the sweep: python diagnose_dirsweep.py 20000 12800
T0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1600          # slope window (T0, T]
rows = []
for code in sorted(c for c in os.listdir(D) if os.path.isdir(os.path.join(D, c))):
    th = o.placement(code)
    for N in (50, 200, 800):
        if not os.path.exists(os.path.join(D, code, f"online_regret_N{N}_T{T}.csv")):
            continue
        B = N // 2
        a = B / N
        vals = np.array([o.b_val(N, B, c, th) for c in range(B + 1)])
        W = vals.max()
        gap = {"exact": 0.0,
               "fluid": float(W - vals[round_active(N, B, o.fluid_u(a, a, th))]),
               "gaussian": float(W - vals[o.gauss_action(N, B, th)])}
        cum = {}
        for r in csv.DictReader(open(os.path.join(D, code, f"online_regret_N{N}_T{T}.csv"))):
            cum.setdefault(r["planner"], {})[int(r["T"])] = float(r["mean"])
        slope = {p: (cum[p][T] - cum[p][T0]) / (T - T0) for p in cum}
        lead = cum["exact"][T] - cum["fluid"][T]
        ds = slope["fluid"] - slope["exact"]
        cross = T + lead / ds if lead > 0 and ds > 0 else (None if lead > 0 else T)
        rows.append(dict(placement=code, N=N, known_gap_fluid=gap["fluid"], known_gap_gauss=gap["gaussian"],
                         R_exact=cum["exact"][T], R_fluid=cum["fluid"][T], R_gauss=cum["gaussian"][T],
                         slope_exact=slope["exact"], slope_fluid=slope["fluid"], slope_gauss=slope["gaussian"],
                         exact_overtakes_fluid_at=cross))
with open(os.path.join(D, f"diagnosis_T{T}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['placement']} N={r['N']:4d} fluidgap {r['known_gap_fluid']:.2e} gaussgap {r['known_gap_gauss']:.2e} | "
          f"R ex {r['R_exact']:6.3f} fl {r['R_fluid']:6.3f} | slope ex {r['slope_exact']:.2e} fl {r['slope_fluid']:.2e} "
          f"ga {r['slope_gauss']:.2e} | overtake {r['exact_overtakes_fluid_at']}")
