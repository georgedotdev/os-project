"""Live dashboard for the Docker demo. Reads results/live_demo.sqlite while run_live.py writes it.

    python demo/dashboard.py                    # live window, refreshes every second
    python demo/dashboard.py --snapshot x.png   # draw once and save (e.g. after a run)

Grey area   = the container's true demand at that minute (from the trace, used only for scoring)
Coloured    = the CPU limit Docker reports was in force (read back from Docker, not from ProDock)
Red dots    = SLO miss: true demand above the limit + 2%  (the same rule as the simulation)
Triangles   = ProDock raised the limit because the burst detector fired
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from prodock.store import read_live, scoreboard     # noqa: E402

DB = os.path.join(os.path.dirname(HERE), "results", "live_demo.sqlite")
COL = {"prodock": "#d9631e", "reactive": "#2f6db3", "static": "#6f6f6f"}
TITLE = {"prodock": "ProDock", "reactive": "Reactive (threshold rules)", "static": "Static limit"}


def series(rows, name, key):
    return [(r["sim_tick"], r[key]) for r in rows if r["container"] == name and r[key] is not None]


def cpu_panel(ax, rows, name, policy):
    tc = series(rows, name, "true_cpu")
    lc = series(rows, name, "limit_cpu")
    if tc:
        ax.fill_between(*zip(*tc), step="post", color="#cfcfcf", lw=0, label="true demand")
    if lc:
        ax.step(*zip(*lc), where="post", color=COL[policy], lw=1.8, label="CPU limit (from Docker)")
    miss = [(r["sim_tick"], r["true_cpu"]) for r in rows if r["container"] == name
            and r["true_cpu"] is not None and r["true_cpu"] > (r["limit_cpu"] or 0) * 1.02]
    if miss:
        ax.scatter(*zip(*miss), color="#c0392b", s=14, zorder=5, label="SLO miss")
    bu = [(r["sim_tick"], r["new_cpu"]) for r in rows
          if r["container"] == name and r["reason"] == "burst-up"]
    if bu:
        ax.scatter(*zip(*bu), marker="^", color="#8e44ad", s=30, zorder=6, label="burst-up")
    ax.set_ylabel("cores")
    ax.set_title(f"{TITLE[policy]}  —  {name}", loc="left", fontsize=10)


def draw(fig, axes, side, clock, window):
    rows = read_live(DB)
    for ax in axes + [side]:
        ax.clear()
    roles = clock["containers"]
    web = [n for n, r in roles.items() if r["trace"] == "web-frontend"]
    order = sorted(web, key=lambda n: ["prodock", "reactive", "static"].index(roles[n]["policy"]))
    now = max((r["sim_tick"] for r in rows), default=clock["start_tick"])
    lo = max(clock["start_tick"], now - window)

    for ax, n in zip(axes, order):
        cpu_panel(ax, rows, n, roles[n]["policy"])
    axes[0].legend(loc="upper left", fontsize=7, ncol=4, frameon=False)

    pd_web = next((n for n in order if roles[n]["policy"] == "prodock"), None)
    ax = axes[len(order)]
    b = series(rows, pd_web, "burst") if pd_web else []
    thr = next((r["burst_thr"] for r in rows if r["container"] == pd_web and r["burst_thr"]), None)
    if b:
        ax.plot(*zip(*b), color="#8e44ad", lw=1.2)
    if thr:
        ax.axhline(thr, color="#8e44ad", ls="--", lw=0.8)
        ax.text(lo, thr, " burst line", va="bottom", fontsize=7, color="#8e44ad")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Stage 1 burst score (Isolation Forest) for ProDock's web-frontend",
                 loc="left", fontsize=10)

    if "batch-worker" in roles and len(axes) > len(order) + 1:
        ax = axes[len(order) + 1]
        tm = series(rows, "batch-worker", "true_mem")
        lm = series(rows, "batch-worker", "limit_mem")
        if tm:
            ax.fill_between(*zip(*tm), step="post", color="#cfcfcf", lw=0, label="true memory")
        if lm:
            ax.step(*zip(*lm), where="post", color=COL["prodock"], lw=1.8, label="memory limit")
        dead = [(r["sim_tick"], 0.1) for r in rows if r["container"] == "batch-worker" and r["died"]]
        if dead:
            ax.scatter(*zip(*dead), marker="x", color="#c0392b", s=40, zorder=5, label="OOM kill")
        ax.set_ylabel("GB")
        ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)
        ax.set_title("ProDock  —  batch-worker memory (jobs every 2 sim-hours)", loc="left",
                     fontsize=10)

    for ax in axes:
        ax.set_xlim(lo, max(lo + 60, now + 2))
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda t, _: f"{int(t) % 1440 // 60:02d}:{int(t) % 60:02d}"))
    axes[-1].set_xlabel(f"simulated time of day (1 simulated minute = {clock['tick_seconds']:g} s)")

    side.axis("off")
    sb = scoreboard(rows)
    txt = ["LIVE SCOREBOARD", f"sim tick {now}", ""]
    if not sb:
        txt.append("ProDock is filling its\nhistory buffer ...")
    for n, s in sb.items():
        txt += [n, f"  policy     {s['policy']}",
                f"  SLO misses {s['slo']} ({s['slo_pct']:.1f}%)",
                f"  kernel thr {s['kernel']} ticks",
                f"  avg limit  {s['reserved']:.2f} cores",
                f"  wasted     {s['wasted']:.0f} core-min",
                f"  resizes    {s['resizes']}",
                f"  OOM/deaths {s['deaths']}", ""]
    side.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8.5)
    fig.suptitle("ProDock running LIVE on Docker", fontsize=13, x=0.02, ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=300, help="sim-minutes shown")
    ap.add_argument("--snapshot", default=None, help="save one PNG and exit")
    a = ap.parse_args()
    clock = json.load(open(os.path.join(HERE, "clock.json")))
    n_web = sum(r["trace"] == "web-frontend" for r in clock["containers"].values())
    n = n_web + 1 + ("batch-worker" in clock["containers"])
    fig = plt.figure(figsize=(14, 2.1 * n + 1))
    gs = fig.add_gridspec(n, 2, width_ratios=[4.2, 1], hspace=0.55, wspace=0.05,
                          top=0.94, bottom=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    side = fig.add_subplot(gs[:, 1])
    if a.snapshot:
        draw(fig, axes, side, clock, a.window)
        fig.savefig(a.snapshot, dpi=110, bbox_inches="tight")
        print("saved", a.snapshot)
        return
    _anim = FuncAnimation(fig, lambda _: draw(fig, axes, side, clock, a.window),
                          interval=1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
