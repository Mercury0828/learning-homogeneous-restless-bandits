# Code for "Learning Homogeneous Restless Bandits with Accurate Finite-Population Planning"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24409930.svg)](https://doi.org/10.5281/zenodo.24409930)

Archived version: https://doi.org/10.5281/zenodo.24409930 (v1.0.0).

Every number in Section 8 and Appendix D of the paper is produced by the scripts below. All values are exact
finite-population expectations computed by binomial convolution, except the learners' own randomness, which is
averaged over 20 runs with seeds 1000*N + s, s = 0..19.

## Requirements
Python 3.10 or later with numpy, scipy and matplotlib (see requirements.txt).

## What reproduces what
| Paper item | Command (run inside this folder) | Output |
|---|---|---|
| Table 2 (transfer at a fixed candidate) | `python transfer_sweep.py` | `out/transfer_sweep.csv` |
| Long-horizon learning, 16 kernels, N = 200 and 800, T = 20000 | `python run_dirsweep.py 16 20000 200,800` | `out/dirsweep/<kernel>/online_regret_N{N}_T20000.csv` |
| Short-horizon runs, T = 2000 | `python run_dirsweep.py 16 2000 50,200,800` | `out/dirsweep/<kernel>/online_regret_N{N}_T2000.csv` |
| Ratios and intervals quoted in Section 8 | `python summarize_dirsweep.py 20000` (and `2000`) | `out/dirsweep/summary_T{T}.csv` |
| Late-window losses versus the known-kernel gap | `python diagnose_dirsweep.py 20000 12800` | `out/dirsweep/diagnosis_T{T}.csv` |
| Figure 1 | `python make_fig_dirsweep.py 20000 out` | `out/online_regret_long.pdf` |
| Priority-rule losses (Appendix D) | `python lp_priority_check.py` | `out/dirsweep/lp_priority_check.csv` |
| **Every number quoted in Section 8 and Appendix D** (ratios, interval separation, late-window fractions, crossover checkpoints, T = 2000 counts, Gaussian differences, priority-rule bounds and exceptions, N_G, L_tr rho, kappa_G eps_tube) | `python report_claims.py` | printed, with source file and formula per line |

A kernel is coded by four letters p/m: the true kernel is theta_Q + v/40 with v_i = +1 for p and -1 for m, where
theta_Q = (1/5, 9/10, 7/10, 1/4) lists the stage-0 probabilities of reaching state 0 from (0,1), (0,0), (1,1), (1,0).
The figure's panel (a) uses kernel `mppm`.

## Run time
The T = 20000 sweep takes about 70 CPU-hours (about 5 hours on 16 cores). The other scripts take minutes.
The result files of the paper are included under `out/`, so the summaries, the report and the figure can be
regenerated without rerunning the sweep. The sweep driver skips a job whose result file is already complete (the seeds
are fixed, so a rerun gives the same numbers); pass `--force` to rerun every job from scratch. Table 2 shows the seven
distances listed in its rows, multiplied by 1000 and rounded; `transfer_sweep.csv` also holds delta = 1/320 and 1/8.

## License
Code and result files are released under the MIT License (see LICENSE).
