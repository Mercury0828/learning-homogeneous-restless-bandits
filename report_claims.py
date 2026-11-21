"""Print a summary of numerical claims in Section 8 and Appendix D from the result files.

Reads out/transfer_sweep.csv and out/dirsweep/ (summary_T{2000,20000}.csv, diagnosis_T20000.csv,
lp_priority_check.csv, <kernel>/online_regret_N{N}_T20000.csv) and recomputes the theoretical constants used in
Section 8 (N_G, L_tr * rho, kappa_G * eps_tube). Nothing is filtered; each line names its source and formula.
Usage: python report_claims.py
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
D = os.path.join(OUT, "dirsweep")


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table2():
    print("== Table 2: 1000 x expected per-arm loss (out/transfer_sweep.csv; rows shown in the paper: "
          "delta in {0, 1/160, 1/80, 1/40, 1/32, 1/16, 1/4})")
    for r in rows(os.path.join(OUT, "transfer_sweep.csv")):
        print("  " + ", ".join(f"{k}={r[k]}" for k in r))


def learning():
    s20 = rows(os.path.join(D, "summary_T20000.csv"))
    m = {(r["placement"], int(r["N"]), r["planner"]): (float(r["mean"]), float(r["ci95"])) for r in s20}
    kernels = sorted({k for k, _, _ in m})
    print("== Learning at T = 20000 (out/dirsweep/summary_T20000.csv)")
    for N in (800, 200):
        ks = [k for k in kernels if (k, N, "exact") in m]
        ratio = [m[(k, N, "fluid")][0] / m[(k, N, "exact")][0] for k in ks]
        sep = all(m[(k, N, "exact")][0] + m[(k, N, "exact")][1] < m[(k, N, "fluid")][0] - m[(k, N, "fluid")][1]
                  for k in ks)
        gdev = max(abs(m[(k, N, "gaussian")][0] - m[(k, N, "exact")][0]) / m[(k, N, "exact")][0] for k in ks)
        print(f"  N={N}: exact < fluid on {sum(r > 1 for r in ratio)}/{len(ks)} kernels; fluid/exact ratio "
              f"{min(ratio):.1f} to {max(ratio):.1f}; 95% intervals disjoint on all kernels: {sep}; "
              f"max |gaussian - exact| / exact = {100 * gdev:.3f}%")
        glow = sum(m[(k, N, "gaussian")][0] < m[(k, N, "exact")][0] for k in ks)
        gov = sum(abs(m[(k, N, "gaussian")][0] - m[(k, N, "exact")][0]) <= m[(k, N, "gaussian")][1] + m[(k, N, "exact")][1]
                  for k in ks)
        print(f"  N={N}: gaussian mean below exact on {glow}/{len(ks)} kernels; gaussian and exact 95% intervals "
              f"overlap on {gov}/{len(ks)}")
    d = rows(os.path.join(D, "diagnosis_T20000.csv"))
    print("== Late window, episodes 12801-20000 (out/dirsweep/diagnosis_T20000.csv, built with "
          "'python diagnose_dirsweep.py 20000 12800')")
    for N in (800, 200):
        dn = [r for r in d if int(r["N"]) == N]
        frac = [float(r["slope_fluid"]) / float(r["known_gap_fluid"]) for r in dn]
        ex = max(float(r["slope_exact"]) for r in dn)
        print(f"  N={N}: fluid regret per episode / known-kernel fluid gap = {min(frac):.2f} to {max(frac):.2f}; "
              f"max exact regret per episode = {ex:.2e}")
    print("== Crossover on kernel mppm (out/dirsweep/mppm/online_regret_N{N}_T20000.csv, checkpoint curves)")
    for N in (200, 800):
        c = {}
        for r in rows(os.path.join(D, "mppm", f"online_regret_N{N}_T20000.csv")):
            c.setdefault(r["planner"], {})[int(r["T"])] = float(r["mean"])
        ts = sorted(c["exact"])
        first = next(t for t in ts if c["exact"][t] < c["fluid"][t])
        prev = ts[ts.index(first) - 1]
        print(f"  N={N}: fluid ahead at checkpoint {prev}, exact ahead from checkpoint {first}")
    s2 = rows(os.path.join(D, "summary_T2000.csv"))
    m2 = {(r["placement"], int(r["N"]), r["planner"]): float(r["mean"]) for r in s2}
    print("== Short horizon, T = 2000 (out/dirsweep/summary_T2000.csv)")
    for N in (200, 800):
        ks = sorted({k for k, n, _ in m2 if n == N})
        print(f"  N={N}: fluid regret below exact on {sum(m2[(k, N, 'fluid')] < m2[(k, N, 'exact')] for k in ks)}"
              f"/{len(ks)} kernels")


def priority():
    r = rows(os.path.join(D, "lp_priority_check.csv"))
    best = [(x["placement"], int(x["N"]), min(float(x["gap_priority_c0"]), float(x["gap_priority_cB"])),
             float(x["gap_fluid"])) for x in r]
    print("== Priority rule with the true kernel (out/dirsweep/lp_priority_check.csv; loss of the better of the two "
          "priority orders)")
    print(f"  minimum loss over {len(best)} cases = {min(b[2] for b in best):.4e}")
    ok = [b for b in best if b[2] >= b[3]]
    print(f"  priority loss >= fluid loss in {len(ok)}/{len(best)} cases; exceptions:")
    for b in best:
        if b[2] < b[3]:
            print(f"    kernel {b[0]}, N={b[1]}: priority {b[2]:.2e}, fluid {b[3]:.2e}")


def theory():
    rmax, d, H, mu, LLP = 1.0, 2, 2, 1 / 40, 1.0
    kG = max(2 + 6 * d, 3 + 2 * rmax * H * d / mu, 2 * LLP)
    sG = kG - LLP
    Cfls = rmax * math.sqrt(d) * H * (H - 1) / 2
    Cflo = rmax * d * H * (H + 1)
    Clocs, Cloco = Cfls / mu, 1 + (Cflo + 2 * rmax * H ** 2) / mu
    ell = max(2, (Clocs + Cloco) / sG, 2 * math.log(8 * d), math.sqrt(math.log(8 * H * d)))
    NG = math.ceil(math.exp(ell))
    Ltr = rmax * H * (H + 1) * (H + 2) * (2 + math.sqrt(d)) / 6
    print("== Theory constants of the running example (Section 4.3, Theorem 2)")
    print(f"  kappa_G = {kG:.0f}, N_G = {NG}, L_tr = {Ltr:.4f}, L_tr/32 = {Ltr / 32:.3f}")
    for N in (200, 800, 3200):
        print(f"  N={N}: eps_tube = {2 * math.log(N) / math.sqrt(N):.4f}, "
              f"kappa_G * eps_tube = {kG * 2 * math.log(N) / math.sqrt(N):.1f}")


if __name__ == "__main__":
    table2()
    learning()
    priority()
    theory()
