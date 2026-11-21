"""Figure 1 of the paper: online per-arm regret over long
horizons. Panel (a): mean cumulative regret against episodes at the primary placement (mppm), exact vs fluid
learner, N = 200 and 800. Panel (b): every one of the 16 placements at T, exact (x) against fluid (y), with the
diagonal; a point above the diagonal is a placement where the exact learner has the smaller regret.

Encoding: planner = colour AND line style (exact solid, fluid dashed; panel (b) uses the axes), N = marker shape
and fill, so the figure reads in grayscale. The Gaussian variant is omitted (its largest relative difference from the
exact learner is printed and goes into the caption). Missing (N, placement) files are skipped and reported.
Usage: python make_fig_dirsweep.py [T] [outdir]   (default T = 20000, outdir = out/dirsweep)"""
import csv
import itertools
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "out", "dirsweep")
T = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else D
PRIMARY = "mppm"
POPS = [200, 800]
CODES = ["".join(c) for c in itertools.product("pm", repeat=4)]
COL = {"exact": "#2a78d6", "fluid": "#eb6834"}
LS = {"exact": "-", "fluid": "--"}
MK = {200: "o", 800: "^"}
FILL = {200: "white", 800: None}                         # N = 200 hollow, N = 800 filled
FIGW, FIGH, FONT = 6.5, 2.75, 9.0


def load(code, N):
    f = os.path.join(D, code, f"online_regret_N{N}_T{T}.csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        out.setdefault(r["planner"], []).append((int(r["T"]), float(r["mean"]), float(r["ci95"])))
    return {p: sorted(v) for p, v in out.items()}


def log_ticks(lo, hi):
    return [10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]


plt.rcParams.update({"font.size": FONT, "font.family": "DejaVu Sans", "mathtext.default": "regular",
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.edgecolor": "#55554f", "xtick.color": "#55554f", "ytick.color": "#55554f"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGW, FIGH), gridspec_kw={"width_ratios": [1.35, 1]})
missing, gauss_dev = [], 0.0

# (a) primary placement over episodes
lo, hi = float("inf"), 0.0
for N in POPS:
    d = load(PRIMARY, N)
    if d is None:
        missing.append((PRIMARY, N))
        continue
    for p in ("exact", "fluid"):
        ts, ms, _ = zip(*d[p])
        lo, hi = min(lo, min(ms)), max(hi, max(ms))
        ax1.plot(ts, ms, ls=LS[p], lw=1.4, color=COL[p], marker=MK[N], ms=4.2, markeredgewidth=0.9,
                 markerfacecolor=FILL[N] or COL[p], zorder=3)
# reference slopes 1/2 (statistical term, sqrt(T)) and 1 (accumulated planning error, linear in T), anchored at the
# last checkpoint of the N = 800 exact and fluid curves; thin grey so they read as guides, labelled at the right end
ref = load(PRIMARY, 800)
if ref is not None:
    for p, slope, lab in (("exact", 0.5, "slope 1/2"), ("fluid", 1.0, "slope 1")):
        tT, mT, _ = ref[p][-1]
        xs = [tT / 100.0, tT]
        ys = [mT * (x / tT) ** slope for x in xs]
        ax1.plot(xs, ys, color="#8a8a82", lw=0.8, ls=":", zorder=1)
        ax1.text(tT * 1.6, ys[-1] * (1.25 if slope == 1.0 else 0.8), lab, color="#55554f", fontsize=FONT - 1.5,
                 va="center", ha="left")
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(0.7, T * 12)
xt = [t for t in (1, 10, 100, 1000, 10000) if t <= T]
ax1.xaxis.set_major_locator(FixedLocator(xt))
ax1.xaxis.set_major_formatter(FixedFormatter([f"{t:,}".replace(",", " ") for t in xt]))
yt = log_ticks(lo, hi)
ax1.yaxis.set_major_locator(FixedLocator(yt))
ax1.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yt]))
for a in (ax1.xaxis, ax1.yaxis):
    a.set_minor_locator(NullLocator())
ax1.set_xlabel("episodes")
ax1.set_ylabel("cumulative per-arm regret")
ax1.set_title("(a) one true kernel", fontsize=FONT, loc="left")

# (b) all placements at T
pts = {N: [] for N in POPS}
for N in POPS:
    for c in CODES:
        d = load(c, N)
        if d is None:
            missing.append((c, N))
            continue
        e = dict((t, m) for t, m, _ in d["exact"])
        g = dict((t, m) for t, m, _ in d["gaussian"])
        fl = dict((t, m) for t, m, _ in d["fluid"])
        gauss_dev = max(gauss_dev, max(abs(g[t] - e[t]) / e[t] for t in e if e[t] > 0))
        pts[N].append((e[T], fl[T]))
allv = [v for N in POPS for xy in pts[N] for v in xy]
if allv:
    vlo, vhi = min(allv) / 1.3, max(allv) * 1.3
    ax2.plot([vlo, vhi], [vlo, vhi], color="#8a8a82", lw=0.8, ls=":", zorder=1)
    ax2.text(vhi / 1.15, vhi / 4.0, "equal", ha="right", va="top", color="#55554f", fontsize=FONT - 1)
    for N in POPS:
        if pts[N]:
            xs, ys = zip(*pts[N])
            ax2.scatter(xs, ys, marker=MK[N], s=26, facecolors=FILL[N] or "#33332f", edgecolors="#33332f",
                        linewidths=0.9, zorder=3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(vlo, vhi)
    ax2.set_ylim(vlo, vhi)
    tk = [v for v in (1, 2, 5, 10, 20, 50, 100) if vlo <= v <= vhi]
    for a in (ax2.xaxis, ax2.yaxis):
        a.set_major_locator(FixedLocator(tk))
        a.set_major_formatter(FixedFormatter([f"{v:g}" for v in tk]))
        a.set_minor_locator(NullLocator())
TSTR = f"{T:,}".replace(",", " ")
ax2.set_xlabel(f"exact learner at T = {TSTR}")
ax2.set_ylabel(f"fluid learner at T = {TSTR}")
ax2.set_title("(b) all 16 true kernels", fontsize=FONT, loc="left")
ax2.set_aspect("equal", adjustable="box")
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles = [Line2D([], [], color=COL["exact"], ls="-", lw=1.4, label="exact learner"),
           Line2D([], [], color=COL["fluid"], ls="--", lw=1.4, label="fluid learner")]
handles += [Line2D([], [], color="#33332f", ls="", marker=MK[N], ms=4.6, markeredgewidth=0.9,
                   markerfacecolor=FILL[N] or "#33332f", label=f"N = {N}") for N in POPS]
fig.legend(handles=handles, loc="upper center", frameon=False, ncol=4, handlelength=2.2, columnspacing=1.4,
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.90), w_pad=1.5)
os.makedirs(OUTDIR, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"online_regret_long.{ext}"), dpi=200)
print("wrote", os.path.join(OUTDIR, "online_regret_long.pdf"), "and .png")
print(f"Gaussian vs exact, largest relative difference over all plotted runs and checkpoints: {gauss_dev:.1%}")
print("missing:", missing if missing else "none")
