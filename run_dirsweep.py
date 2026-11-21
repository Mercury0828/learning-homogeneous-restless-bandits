"""Sweep driver: online regret at all 16 truth placements centre + v/40, v in {+-1}^4.

Runs `online_regret.py N T SEEDS <code>` for every placement code (letters p/m) and N in {50, 200, 800}, with
T = 2000 and 20 seeds, in a pool of worker processes. Each job writes
out/dirsweep/<code>/online_regret_N{N}_T{T}.csv and _meta.txt; this driver writes a job log to
out/dirsweep/driver_log.txt. Nothing is skipped or filtered here.

Usage: python run_dirsweep.py [WORKERS] [T] [N1,N2,...] [--force] [--dry]
       python run_dirsweep.py 16 20000 200,800   (long-horizon run of Section 8)

Resume: a job whose CSV already holds the row for checkpoint T for all three planners is not re-run (logged as
"skip"); its seeds are fixed (1000*N + s), so a re-run would reproduce the same numbers. Pass --force to rerun every job.
"""
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "dirsweep")
POPS = [50, 200, 800]
T, SEEDS = 2000, 20
CODES = ["".join(c) for c in itertools.product("pm", repeat=4)]


def job(args):
    N, code = args
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "online_regret.py"), str(N), str(T), str(SEEDS), code],
                       cwd=HERE, env=env, capture_output=True, text=True)
    return N, code, r.returncode, time.time() - t0, r.stdout.strip(), r.stderr.strip()[-2000:]


def done(N, code):
    f = os.path.join(OUT, code, f"online_regret_N{N}_T{T}.csv")
    if not os.path.exists(f):
        return False
    rows = [ln.split(",") for ln in open(f, encoding="utf-8").read().splitlines()[1:]]
    return {r[1] for r in rows if len(r) > 2 and r[2] == str(T)} == {"exact", "gaussian", "fluid"}


def main():
    global T, POPS
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    if len(sys.argv) > 2:
        T = int(sys.argv[2])
    if len(sys.argv) > 3:
        POPS = [int(n) for n in sys.argv[3].split(",")]
    os.makedirs(OUT, exist_ok=True)
    alljobs = [(N, c) for N in sorted(POPS, reverse=True) for c in CODES]   # largest N first: longest jobs start early
    jobs = alljobs if "--force" in sys.argv else [j for j in alljobs if not done(*j)]
    if "--dry" in sys.argv:
        print(f"{len(jobs)} of {len(alljobs)} jobs to run:", jobs)
        return
    with open(os.path.join(OUT, "driver_log.txt"), "a") as log, ThreadPoolExecutor(min(workers, max(1, len(jobs)))) as ex:
        log.write(f"start {time.ctime()} workers={workers} jobs={len(jobs)} T={T} seeds={SEEDS}\n")
        for j in alljobs:
            if j not in jobs:
                log.write(f"skip N={j[0]} code={j[1]} (CSV complete at T={T})\n")
        log.flush()
        for N, code, rc, dt, so, se in ex.map(job, jobs):
            log.write(f"N={N} code={code} rc={rc} seconds={dt:.0f}\n{so}\n" + (f"STDERR {se}\n" if rc else ""))
            log.flush()
        log.write(f"end {time.ctime()}\n")


if __name__ == "__main__":
    main()
