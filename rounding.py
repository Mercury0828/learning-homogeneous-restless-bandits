"""Largest-remainder rounding of the witness active masses, shared by the scripts of this package.

With B = floor(N/2) activations, the active masses are (u, alpha_N - u) on states (0, 1). Each count is floored, and
the remaining activation goes to the larger fractional part, with ties broken in favour of state 0. Floating-point
products such as N * u can miss an integer or a half-integer by a few units in the last place, so floors and the
remainder comparison use the absolute tolerance TOL; for the populations used here (N <= 12800) the rounding error of
N * u is below 1e-11, far inside TOL.
"""
import math

TOL = 1e-9


def round_active(N, B, u):
    """Returns the state-0 active count."""
    a0 = N * u
    a1 = B - a0
    f0 = math.floor(a0 + TOL)
    f1 = math.floor(a1 + TOL)
    if B - f0 - f1 <= 0:
        return min(max(f0, 0), B)
    return min(max(f0 + 1 if (a0 - f0) >= (a1 - f1) - TOL else f0, 0), B)
