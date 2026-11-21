"""Online regret on the certified witness class.

Model. The witness of Proposition `prop:witness` (d = H = 2, alpha = 1/2, reward 1 for activating a state-0 arm).
Only the four stage-0 rows affect decisions; write theta = (A, B, C, D) for their probabilities of reaching state 0
from (0,1), (0,0), (1,1), (1,0). The known certified class is the ball of radius 1/32 around the witness center
(1/5, 9/10, 7/10, 1/4), on which Assumptions 1-5 hold globally. The unknown true kernel lies inside the ball.

Learner (Section `sec:learner`). Rowwise confidence intervals of half-width min{1, sqrt(ell/(2n))} with
ell = log(2^{d+1} D_row N T / beta), D_row = 2Hd, beta = 1/(2NT), intersected with the class box. Each episode the
learner selects a kernel with maximal planner value over this box and plays the planner's policy:
  * exact planner: W^N_Q is nondecreasing in each coordinate of theta (the next state-0 count increases
    stochastically, and E min{S, B_N} is nondecreasing), so the upper corner attains the supremum exactly;
  * fluid re-solve planner: F^{alpha_N}_Q is nondecreasing in each coordinate, so the upper corner is exact;
  * Gaussian planner (numerical variant): the projected Gaussian objective is evaluated at each of the 16 box
    vertices on a 401-point grid in u, the vertex with the largest grid value is selected, and its action is refined
    by bounded scalar minimization between the neighbouring grid points before rounding. This procedure has no
    certified selection tolerance.
Only the four stage-0 rows are tracked: terminal rows do not affect any planner, so their confidence intervals are
omitted (D_row = 2Hd = 8 is kept in the logarithm), and the terminal-row part of the empty-intersection test is not
simulated. An empty intersection falls back to the class center.

Randomness. Run s (s = 0, ..., SEEDS-1) uses numpy.random.default_rng(1000 N + s). One generator per run is consumed
sequentially by the exact, Gaussian, and fluid learners in that order, so changing the planner order changes the
Gaussian and fluid realizations.

Regret. Per episode, W^N_{P*} minus the exact finite-population value under P* of the played stage-0 active count
(terminal priority at stage 1 is exactly optimal for every kernel).
Usage: python online_regret.py N T SEEDS  ->  out/online_regret_N{N}_T{T}.csv and a _meta.txt record
       python online_regret.py N T SEEDS mppm  ->  truth at centre + (-1,+1,+1,-1)/40 (16-kernel sweep),
                                                  written to out/dirsweep/mppm/
"""
import csv
import itertools
import math
import os
import sys

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
from scipy.stats import binom, norm

from rounding import round_active

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CENTER = np.array([0.2, 0.9, 0.7, 0.25])
RADIUS = 1.0 / 32
TRUTH = CENTER + np.array([1, -1, -1, 1]) / 40.0          # inside the ball (1/40 < 1/32)
D_STATES, HORIZON = 2, 2


def pmf_next(N, B, c0, th):
    sizes = [(c0, th[0]), (B - c0, th[1]), (B - c0, th[2]), (N - 2 * B + c0, th[3])]
    pmf = np.array([1.0])
    for n, p in sizes:
        pmf = fftconvolve(pmf, binom.pmf(np.arange(n + 1), n, p))
    pmf = np.clip(pmf, 0.0, None)
    return pmf / pmf.sum()


def b_val(N, B, c0, th):
    pmf = pmf_next(N, B, c0, th)
    return c0 / N + float(np.dot(np.minimum(np.arange(pmf.size), B), pmf)) / N


def mean_next(alpha_n, u, th):
    return u * th[0] + (alpha_n - u) * th[1] + (alpha_n - u) * th[2] + (1 - 2 * alpha_n + u) * th[3]


def exact_action(N, B, th):
    """Exhaustive argmax over c0, pruned by the Jensen upper bound c/N + min{E S/N, alpha_N} (exact pruning)."""
    alpha_n = B / N
    cs = np.arange(B + 1)
    ub = cs / N + np.minimum(mean_next(alpha_n, cs / N, th), alpha_n)
    order = np.argsort(-ub)
    best_c, best_v = None, -1.0
    for c in order:
        if ub[c] < best_v - 1e-12:
            break
        v = b_val(N, B, int(c), th)
        if v > best_v:
            best_c, best_v = int(c), v
    return best_c


def fluid_u(x0, b, th):
    A, Bp, C, Dp = th
    c = Bp - A + C - Dp
    e = Bp - Dp
    kap = 1 - C + Dp
    lo, hi = max(0.0, x0 + b - 1), min(x0, b)
    return min(hi, max(lo, (Dp + e * x0 - kap * b) / c))


def h(a, t):
    return t * norm.cdf(t / a) + a * norm.pdf(t / a)


def gauss_sigma(th):
    """Frozen per-arm next-state variance at the nominal allocation (x_0 = (1/2, 1/2), budget 1/2)."""
    u = fluid_u(0.5, 0.5, th)
    A, Bp, C, Dp = th
    return math.sqrt(u * A * (1 - A) + (0.5 - u) * Bp * (1 - Bp) + (0.5 - u) * C * (1 - C) + u * Dp * (1 - Dp))


def gauss_objective(N, alpha_n, us, th, sigma):
    """Projected frozen-covariance Gaussian objective u + E min{clip_[0,1](m + a Z), alpha_N} (closed form)."""
    a = sigma / math.sqrt(N)
    m = mean_next(alpha_n, us, th)
    return us + h(a, m) - h(a, m - alpha_n)


GRID = np.linspace(0.0, 1.0, 401)


def gauss_grid_value(N, B, th):
    alpha_n = B / N
    us = GRID * alpha_n
    vals = gauss_objective(N, alpha_n, us, th, gauss_sigma(th))
    i = int(np.argmax(vals))
    return float(vals[i]), i


def gauss_action(N, B, th):
    alpha_n = B / N
    sigma = gauss_sigma(th)
    _, i = gauss_grid_value(N, B, th)
    us = GRID * alpha_n
    lo, hi = us[max(0, i - 1)], us[min(len(us) - 1, i + 1)]
    res = optimize.minimize_scalar(lambda u: -float(gauss_objective(N, alpha_n, u, th, sigma)), bounds=(lo, hi),
                                   method="bounded", options={"xatol": 1e-12})
    return round_active(N, B, float(res.x))


def run(N, T, seed, planners, truth_vals):
    rng = np.random.default_rng(seed)
    B = N // 2
    alpha_n = B / N
    beta = 1.0 / (2 * N * T)
    d_row = 2 * HORIZON * D_STATES
    ell = math.log(2 ** (D_STATES + 1) * d_row * N * T / beta)
    w_star = float(truth_vals.max())
    out = {}
    for planner in planners:
        n = np.zeros(4)
        s = np.zeros(4)
        cum = 0.0
        curve = []
        for k in range(T):
            phat = np.where(n > 0, s / np.maximum(n, 1), 0.5)
            rad = np.where(n > 0, np.minimum(1.0, np.sqrt(ell / (2 * np.maximum(n, 1)))), 1.0)
            lo = np.maximum(phat - rad, CENTER - RADIUS)
            hi = np.minimum(phat + rad, CENTER + RADIUS)
            if np.any(lo > hi):
                theta = CENTER.copy()
            elif planner in ("exact", "fluid"):
                theta = hi
            else:
                best_val, theta = -1.0, None
                for corner in itertools.product(*zip(lo, hi)):
                    th = np.array(corner)
                    val, _ = gauss_grid_value(N, B, th)
                    if val > best_val:
                        best_val, theta = val, th
            if planner == "exact":
                c0 = exact_action(N, B, theta)
            elif planner == "fluid":
                c0 = round_active(N, B, fluid_u(alpha_n, alpha_n, theta))
            else:
                c0 = gauss_action(N, B, theta)
            cum += w_star - truth_vals[c0]
            curve.append(cum)
            groups = [c0, B - c0, B - c0, N - 2 * B + c0]
            for g, size in enumerate(groups):
                if size > 0:
                    n[g] += size
                    s[g] += rng.binomial(size, TRUTH[g])
        out[planner] = curve
    return out


def placement(code):
    """Truth at centre + v/40 for a sign vector v written as four letters p/m, e.g. 'pmmp' = (1,-1,-1,1)."""
    if len(code) != 4 or set(code) - {"p", "m"}:
        raise ValueError("placement must be four letters from {p, m}, e.g. mppm")
    return CENTER + np.array([1.0 if ch == "p" else -1.0 for ch in code]) / 40.0


def main():
    global TRUTH, OUT
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    code = sys.argv[4] if len(sys.argv) > 4 else None
    if code is not None:          # 16-kernel sweep: other truth placements, separate output folder
        TRUTH = placement(code)
        OUT = os.path.join(HERE, "out", "dirsweep", code)
    os.makedirs(OUT, exist_ok=True)
    planners = ["exact", "gaussian", "fluid"]
    B = N // 2
    truth_vals = np.array([b_val(N, B, c, TRUTH) for c in range(B + 1)])
    curves = {p: [] for p in planners}
    for sd in range(seeds):
        res = run(N, T, 1000 * N + sd, planners, truth_vals)
        for p in planners:
            curves[p].append(res[p])
    # all exceed 2000, so T <= 2000 outputs are unchanged
    checkpoints = sorted(set([t for t in (1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 10000, 12800)
                              if t <= T] + [T]))
    rows = []
    for p in planners:
        arr = np.array(curves[p])
        for t in checkpoints:
            col = arr[:, t - 1]
            rows.append(dict(N=N, planner=p, T=t, mean=float(col.mean()),
                             ci95=float(1.96 * col.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0))
        print(f"N={N} {p:8s} cumulative regret at T={T}: {arr[:, -1].mean():.5f}")
    with open(os.path.join(OUT, f"online_regret_N{N}_T{T}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    import platform
    import matplotlib
    import scipy
    with open(os.path.join(OUT, f"online_regret_N{N}_T{T}_meta.txt"), "w") as f:
        f.write(f"command: python online_regret.py {N} {T} {seeds}" + (f" {code}" if code else "") + "\n")
        if code:
            f.write(f"truth placement: {code} -> theta = {TRUTH.tolist()}\n")
        f.write(f"seeds: 1000*N + s for s in 0..{seeds - 1}; planner order {planners}\n")
        f.write(f"python {platform.python_version()} numpy {np.__version__} scipy {scipy.__version__} "
                f"matplotlib {matplotlib.__version__}\n")


if __name__ == "__main__":
    main()
