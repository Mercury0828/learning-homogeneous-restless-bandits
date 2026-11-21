"""Transfer at a fixed certified candidate: run the planner computed at Q while the true kernel moves away from it.

Setting. The witness of Proposition `prop:witness` (d = H = 2, alpha = 1/2, reward 1 for activating a state-0 arm),
so only the four stage-0 rows matter; theta = (A, B, C, D) are their probabilities of reaching state 0 from
(0,1), (0,0), (1,1), (1,0). The candidate kernel is the class centre Q = (1/5, 9/10, 7/10, 1/4), which is certified:
Assumptions 1-5 hold on the ball of radius 1/32 around it. The true kernel is

    theta_P(delta) = theta_Q + delta * (1, -1, -1, 1),

so that rho(P_delta, Q) = delta for two-state rows, and P_delta leaves the certified class once delta > 1/32. The
planner is never recomputed: the stage-0 count is the one the planner chooses at Q, and stage 1 plays terminal
priority, which is exactly optimal for every kernel of this family and is what both the local Gaussian maximizer and
the budget-adjusted fallback select at even N.

Reported per population: the exact finite-population optimum under P_delta, the per-arm loss of the Gaussian
planner's action computed at Q, and the per-arm loss of the fluid re-solve action computed at Q. Both losses are
exact expectations, evaluated by binomial convolution, not simulations.

Usage: python transfer_sweep.py            -> out/transfer_sweep.csv
       python transfer_sweep.py 200 800     -> only those populations
"""
import csv
import math
import os
import sys

import numpy as np

from online_regret import CENTER, RADIUS, b_val, exact_action, fluid_u, gauss_action
from rounding import round_active

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DIRECTION = np.array([1.0, -1.0, -1.0, 1.0])
DELTAS = [0.0, 1 / 320, 1 / 160, 1 / 80, 1 / 40, RADIUS, 1 / 16, 1 / 8, 1 / 4]
POPS = [200, 800, 3200]


def fluid_action(N, B, th):
    """Rounded exact-budget fluid re-solve action at the reset."""
    alpha_n = B / N
    return round_active(N, B, fluid_u(alpha_n, alpha_n, th))


def main():
    pops = [int(a) for a in sys.argv[1:]] or POPS
    os.makedirs(OUT, exist_ok=True)
    q = np.array(CENTER)
    rows = []
    for N in pops:
        B = N // 2
        c_gauss = gauss_action(N, B, q)
        c_fluid = fluid_action(N, B, q)
        for delta in DELTAS:
            p = q + delta * DIRECTION
            assert np.all((p >= 0) & (p <= 1)), (delta, p)
            rho = float(np.max(np.abs(p - q)))
            c_star = exact_action(N, B, p)
            w_star = b_val(N, B, c_star, p)
            loss_g = w_star - b_val(N, B, c_gauss, p)
            loss_f = w_star - b_val(N, B, c_fluid, p)
            rows.append(dict(N=N, delta=delta, rho=rho, certified=int(rho <= RADIUS + 1e-15),
                             c_opt=c_star, c_gauss=c_gauss, c_fluid=c_fluid,
                             w_opt=w_star, loss_gauss=loss_g, loss_fluid=loss_f,
                             loss_gauss_sqrtN=loss_g * math.sqrt(N), loss_fluid_sqrtN=loss_f * math.sqrt(N)))
            print(f"N={N:6d} delta={delta:.6f} rho={rho:.6f} cert={rows[-1]['certified']} "
                  f"c*={c_star:5d} cG={c_gauss:5d} cF={c_fluid:5d} "
                  f"lossG={loss_g:.3e} lossF={loss_f:.3e}")
    path = os.path.join(OUT, "transfer_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
