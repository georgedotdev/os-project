"""Full experiment: train, evaluate, compare, and save every table and graph for the review.

    python run_experiment.py            # ~2 minutes on a laptop CPU
    python run_experiment.py --quick    # 3 epochs, for a fast check
"""
import argparse
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prodock import evaluate as ev
from prodock import workload
from prodock.config import Config
from prodock.pipeline import Pipeline
from prodock.store import DecisionLog

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
MOD = os.path.join(ROOT, "models")
COL = {"Static": "#8a8a8a", "Reactive": "#2f6db3",
       "Predictive (ConLAR-style)": "#7a4fa3", "ProDock (ours)": "#b35a1f"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def save(fig, name):
    # titles carry no "Fig N." prefix so they fit any report's own figure numbering
    strip = lambda txt: re.sub(r"^Fig \d+\.\s*", "", txt)
    if fig._suptitle is not None:
        fig._suptitle.set_text(strip(fig._suptitle.get_text()))
    for a in fig.axes:
        a.set_title(strip(a.get_title()), loc="center") if a.get_title() else None
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}")


def hours(t, cfg):
    return t * cfg.tick_seconds / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    for d in (FIG, RES, MOD):
        os.makedirs(d, exist_ok=True)
    cfg = Config(epochs=3 if args.quick else Config.epochs)
    tr_end, va_end, end = ev.split_idx(cfg)

    print("\n[1/6] Generating a simulated Docker host (4 containers, 7 days, 1 row per minute)")
    df = workload.generate(cfg)
    df.to_csv(os.path.join(RES, "workload_trace.csv"), index=False)
    print(f"  {len(df):,} rows  |  train days 1-{cfg.train_days}, validation day "
          f"{cfg.train_days + 1}, test days {cfg.train_days + cfg.val_days + 1}-{cfg.days}")

    print("\n[2/6] Training ProDock (Isolation Forest -> LSTM)")
    t0 = time.time()
    pipe = Pipeline(cfg, use_burst=True).fit(df)
    print(f"  done in {time.time() - t0:.1f}s")
    pipe.save(os.path.join(MOD, "prodock_pipeline.pkl"))
    print("\n      Ablation: same LSTM without the burst column")
    pipe_nb = Pipeline(cfg, use_burst=False).fit(df, verbose=False)

    print("\n[3/6] Forecast accuracy on the 2 unseen test days (peak CPU over next 3 min)")
    acc, series = ev.forecast_accuracy(cfg, df, {"LSTM (no burst score)": pipe_nb,
                                                 "ProDock: IsoForest + LSTM": pipe})
    acc.to_csv(os.path.join(RES, "forecast_accuracy.csv"), index=False)
    print(acc.round(3).to_string(index=False))

    print("\n[4/6] Replaying the test days under four allocation policies")
    facts = ev.factories(cfg)
    results, pre, rows = {}, None, []
    for name, f in facts.items():
        res, pre = ev.replay(cfg, df, pipe, f, pre)
        results[name] = res
        rows.append({"Policy": name, **ev.score(cfg, res)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RES, "policy_comparison.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(table.round(2).to_string(index=False))

    per = []
    for pol, res in results.items():
        for cname, r in res.items():
            s = ev.score(cfg, {cname: r})
            per.append({"Policy": pol, "Container": cname, **s})
    pd.DataFrame(per).to_csv(os.path.join(RES, "per_container.csv"), index=False)

    # decision log (operator path)
    dbp = os.path.join(RES, "decision_log.sqlite")
    if os.path.exists(dbp):
        os.remove(dbp)
    log = DecisionLog(dbp)
    r = results["ProDock (ours)"]
    for cname, rr in r.items():
        for k in range(len(rr["t"])):
            log.write(ts=float(rr["t"][k]), container=cname, policy="ProDock",
                      obs_cpu=float(rr["demand_cpu"][k - 1]) if k else 0.0,
                      obs_mem=float(rr["demand_mem"][k - 1]) if k else 0.0,
                      fc_cpu=float(rr["fc_cpu"][k]), fc_mem=0.0, burst=float(rr["burst"][k]),
                      limit_cpu=float(rr["limit_cpu"][k]), limit_mem=float(rr["limit_mem"][k]),
                      reason=rr["reasons"][k],
                      command=f"docker update --cpus {rr['limit_cpu'][k]:.2f} "
                              f"--memory {int(rr['limit_mem'][k] * 1024)}m {cname}")
    log.commit()

    print("\n[5/6] Headroom sweep (trade-off curve)")
    sweep = []
    for h in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        f = ev.factories(cfg, headroom=h)
        for name in ["Predictive (ConLAR-style)", "ProDock (ours)"]:
            res, _ = ev.replay(cfg, df, pipe, f[name], pre)
            sweep.append({"Policy": name, "headroom": h, **ev.score(cfg, res)})
    for mult in [0.8, 1.0, 1.2, 1.5, 2.0]:
        from prodock.allocator import StaticPolicy
        fs = lambda n, c, m, mult=mult: StaticPolicy(np.percentile(c, 95) * mult, m.max() * 1.1)
        res, _ = ev.replay(cfg, df, pipe, fs, pre)
        sweep.append({"Policy": "Static", "headroom": mult, **ev.score(cfg, res)})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(os.path.join(RES, "headroom_sweep.csv"), index=False)

    print("\n[6/6] Drawing graphs")
    names = cfg.containers
    # Fig 1: workload
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 7.5), sharex=True)
    for ax, n in zip(axes, names):
        d = df[df.container == n]
        ax.plot(hours(d.tick.values, cfg), d.cpu.values, lw=0.6, color="#333")
        ax.axvspan(0, hours(tr_end, cfg), color="#2f6db3", alpha=0.06)
        ax.axvspan(hours(tr_end, cfg), hours(va_end, cfg), color="#7a4fa3", alpha=0.08)
        ax.axvspan(hours(va_end, cfg), hours(end, cfg), color="#b35a1f", alpha=0.08)
        ax.set_ylabel(f"{n}\ncores", fontsize=9)
    axes[0].set_title("Fig 1. CPU demand of the four containers over 7 days "
                      "(blue = train, purple = validation, orange = test)")
    axes[-1].set_xlabel("hours")
    save(fig, "fig1_workload_traces.png")

    # Fig 2: burst detector
    n = "web-frontend"
    d = df[df.container == n].reset_index(drop=True)
    F, b = pipe.rows(n, d)
    sl = slice(va_end, va_end + 1440)
    thr = pipe.burst.threshold(n)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    x = hours(np.arange(sl.start, sl.stop), cfg)
    ax[0].plot(x, d.cpu.values[sl], lw=0.8, color="#333")
    flag = b[sl] > thr
    ax[0].scatter(x[flag], d.cpu.values[sl][flag], s=9, color="#b0233a", zorder=3,
                  label="flagged unusual by Isolation Forest")
    ax[0].set_ylabel("CPU (cores)")
    ax[0].legend(loc="upper left")
    ax[1].plot(x, b[sl], lw=0.8, color="#b35a1f")
    ax[1].axhline(thr, ls="--", color="#b0233a", lw=1, label=f"burst threshold ({thr:.2f})")
    ax[1].set_ylabel("burst score (0-1)")
    ax[1].set_xlabel("hours")
    ax[1].legend(loc="upper left")
    ax[0].set_title("Fig 2. Stage 1: Isolation Forest burst score for web-frontend (test day 1)")
    save(fig, "fig2_burst_detector.png")

    # Fig 3: training curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(pipe.history["train"]) + 1), pipe.history["train"], "o-",
            label="training MSE", color="#2f6db3")
    ax.plot(np.arange(1, len(pipe.history["val"]) + 1), pipe.history["val"], "s-",
            label="validation MSE", color="#b35a1f")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean squared error (scaled units)")
    ax.set_title("Fig 3. Stage 2: LSTM learning curve")
    ax.legend()
    save(fig, "fig3_lstm_training.png")

    # Fig 4: forecast vs actual, one panel per container
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for ax, n in zip(axes.flat, names):
        ts, true = series["_true"][n]
        t2, fc = series["ProDock: IsoForest + LSTM"][n]
        k = slice(600, 900)
        ax.plot(hours(ts[k], cfg), true[k], color="#333", lw=1.2, label="actual peak")
        ax.plot(hours(t2[k], cfg), fc[k], color="#b35a1f", lw=1.2, label="forecast")
        ax.set_title(n, fontsize=10)
        ax.set_ylabel("cores")
    axes[0, 0].legend()
    for a in axes[1]:
        a.set_xlabel("hours")
    fig.suptitle("Fig 4. Forecast vs actual (5 hours of unseen test data)", y=1.0)
    save(fig, "fig4_forecast_vs_actual.png")

    # Fig 5: forecast error bars
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#bbbbbb", "#999999", "#6f8fb5", "#7a4fa3", "#b35a1f"]
    for a, metric in zip(ax, ["MAE (cores)", "MAE on peak ticks"]):
        a.barh(acc["Model"], acc[metric], color=colors)
        for i, v in enumerate(acc[metric]):
            a.text(v, i, f" {v:.3f}", va="center", fontsize=9)
        a.set_title(metric + "  (lower is better)")
        a.invert_yaxis()
    ax[1].set_yticklabels([])
    fig.suptitle("Fig 5. Forecast error: baselines vs LSTM vs ProDock", y=1.02)
    save(fig, "fig5_forecast_error.png")

    # Fig 6: allocation vs demand under each policy
    n = "web-frontend"
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, sharey=True)
    k = slice(0, 720)
    for ax, (pol, res) in zip(axes, results.items()):
        rr = res[n]
        x = hours(rr["t"][k], cfg)
        dmd, lim = rr["demand_cpu"][k], rr["limit_cpu"][k]
        ax.fill_between(x, 0, dmd, color="#cfcfcf", label="demand")
        ax.plot(x, lim, color=COL[pol], lw=1.4, label="CPU limit set")
        over = dmd > lim * (1 + cfg.slo_tolerance)
        ax.scatter(x[over], dmd[over], s=10, color="#b0233a", zorder=3, label="throttled (SLO miss)")
        ax.set_title(pol, fontsize=10, loc="left")
        ax.set_ylabel("cores")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("hours")
    fig.suptitle("Fig 6. What each policy does to web-frontend over 12 test hours", y=1.0)
    save(fig, "fig6_policy_timelines.png")

    # Fig 7: policy comparison bars
    mets = ["SLO violations %", "Wasted CPU (core-h)", "Avg CPU utilisation %", "Limit changes",
            "Wasted memory (GB-h)", "OOM kills"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, m in zip(axes.flat, mets):
        vals = table[m].values
        ax.bar(range(len(table)), vals, color=[COL[p] for p in table.Policy])
        for i, v in enumerate(vals):
            lab = f"{int(v)}" if m in ("Limit changes", "OOM kills") or v >= 100 else f"{v:.1f}"
            ax.text(i, v, lab, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(["Static", "Reactive", "Predictive", "ProDock"], fontsize=9)
        better = "higher" if "utilisation" in m else "lower"
        ax.set_title(f"{m}\n({better} is better)", fontsize=10)
    fig.suptitle("Fig 7. Head-to-head on the 2 test days (all 4 containers)", y=1.01)
    save(fig, "fig7_policy_comparison.png")

    # Fig 8: trade-off curve
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for pol, g in sweep.groupby("Policy"):
        g = g.sort_values("Wasted CPU (core-h)")
        ax.plot(g["Wasted CPU (core-h)"], g["SLO violations %"], "o-", color=COL[pol], label=pol)
    rr = table.set_index("Policy").loc["Reactive"]
    ax.scatter(rr["Wasted CPU (core-h)"], rr["SLO violations %"], s=80, marker="D",
               color=COL["Reactive"], label="Reactive", zorder=4)
    ax.set_xlabel("wasted CPU (core-hours)  lower is better")
    ax.set_ylabel("SLO violations (% of minutes)  lower is better")
    ax.set_title("Fig 8. Trade-off: every point is one headroom setting\n"
                 "(closer to the bottom-left corner is better)")
    ax.legend()
    save(fig, "fig8_tradeoff_curve.png")

    # Fig 9: memory
    n = "batch-worker"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    rr0 = results["ProDock (ours)"][n]
    job = int(np.argmax(np.diff(rr0["demand_mem"][100:]) > 0.5)) + 100   # first job start after 100 min
    for ax, sl in [(axes[0], slice(0, 480)), (axes[1], slice(job - 6, job + 12))]:
        x = hours(rr0["t"][sl], cfg) if ax is axes[0] else rr0["t"][sl] - rr0["t"][job]
        ax.fill_between(x, 0, rr0["demand_mem"][sl], color="#cfcfcf", label="memory demand",
                        step="mid" if ax is axes[1] else None)
        for pol in ["Reactive", "Predictive (ConLAR-style)", "ProDock (ours)"]:
            rr = results[pol][n]
            ax.plot(x, rr["limit_mem"][sl], color=COL[pol], lw=1.3, label=f"{pol} limit",
                    drawstyle="steps-mid" if ax is axes[1] else "default")
            if ax is axes[1]:
                kill = rr["demand_mem"][sl] > rr["limit_mem"][sl]
                ax.scatter(np.asarray(x)[kill], rr["demand_mem"][sl][kill], marker="x", s=60,
                           color=COL[pol], zorder=4)
        ax.set_ylabel("GB")
    axes[0].set_xlabel("hours")
    axes[0].set_title("8 test hours: the job starts every 2 hours", fontsize=10)
    axes[1].set_xlabel("minutes from job start")
    axes[1].set_title("Zoom on one job start (x = OOM kill)", fontsize=10)
    axes[0].legend(ncol=2, fontsize=8, loc="upper right")
    fig.suptitle("Fig 9. batch-worker memory limits. Demand above a policy's line means that "
                 "policy's container is killed", y=1.02)
    save(fig, "fig9_memory_limits.png")

    base = table.set_index("Policy")
    ours, st, rc, pr = (base.loc["ProDock (ours)"], base.loc["Static"],
                        base.loc["Reactive"], base.loc["Predictive (ConLAR-style)"])
    summary = {
        "forecast_accuracy": acc.to_dict(orient="records"),
        "policy_comparison": table.to_dict(orient="records"),
        "headline": {
            "cpu_waste_cut_vs_static_%": round(100 * (1 - ours["Wasted CPU (core-h)"] / st["Wasted CPU (core-h)"]), 1),
            "cpu_waste_cut_vs_reactive_%": round(100 * (1 - ours["Wasted CPU (core-h)"] / rc["Wasted CPU (core-h)"]), 1),
            "slo_violations_%": {k: round(v, 2) for k, v in base["SLO violations %"].items()},
            "fewer_limit_changes_than_conlar_x": round(pr["Limit changes"] / ours["Limit changes"], 1),
            "oom_kills": {k: int(v) for k, v in base["OOM kills"].items()},
        },
    }
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nHEADLINE")
    for k, v in summary["headline"].items():
        print(f"  {k}: {v}")
    print("\nAll outputs are in ./results and ./figures")


if __name__ == "__main__":
    main()
