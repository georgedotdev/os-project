"""Evaluation harness: forecast accuracy + closed-loop allocation replay on the test days."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import features as fx
from .allocator import StaticPolicy, ReactivePolicy, PredictivePolicy, ProDockPolicy


def split_idx(cfg):
    tpd = cfg.ticks_per_day
    return cfg.train_days * tpd, (cfg.train_days + cfg.val_days) * tpd, cfg.days * tpd


def _metrics(pred, true, peak_mask):
    e = pred - true
    return {
        "MAE (cores)": float(np.mean(np.abs(e))),
        "RMSE (cores)": float(np.sqrt(np.mean(e ** 2))),
        "MAE on peak ticks": float(np.mean(np.abs(e[peak_mask]))),
        "Under-predicted %": float(100 * np.mean(e < 0)),
    }


def forecast_accuracy(cfg, df, pipes):
    """pipes: dict label -> fitted Pipeline. Adds naive / moving-average / ridge baselines."""
    tr_end, va_end, end = split_idx(cfg)
    H, w = cfg.horizon, cfg.window
    rows, series = [], {}
    all_pred = {k: [] for k in ["Naive (last value)", "Moving average", "Linear (Ridge)", *pipes]}
    all_true, all_peak = [], []
    for name in cfg.containers:
        d = df[df.container == name].reset_index(drop=True)
        cpu = d.cpu.values
        ts = np.arange(va_end, end - H)
        true = np.array([cpu[t + 1:t + 1 + H].max() for t in ts])
        p95 = np.percentile(cpu[:tr_end], 95)
        all_true.append(true); all_peak.append(true > p95)
        all_pred["Naive (last value)"].append(cpu[ts])
        all_pred["Moving average"].append(np.array([cpu[t - 9:t + 1].mean() for t in ts]))
        # ridge on flattened window of L2 rows (no burst column)
        base = next(iter(pipes.values()))
        sc = base.scales[name]
        F = fx.build(cpu, d.mem.values, d.net.values, d.minute_of_day.values, sc, cfg.roll,
                     cfg.ticks_per_day)
        Y = fx.targets(cpu, d.mem.values, sc, H)
        Xtr, Ytr, _ = fx.windows(F, Y, w, 0, tr_end - H)
        rg = Ridge(alpha=1.0).fit(Xtr.reshape(len(Xtr), -1), Ytr[:, 0])
        Xte = np.stack([F[t - w + 1:t + 1] for t in ts]).reshape(len(ts), -1)
        all_pred["Linear (Ridge)"].append(rg.predict(Xte) * sc[0])
        for label, p in pipes.items():
            t2, fc, _, _ = p.forecast_series(name, d, va_end, end - H)
            all_pred[label].append(fc)
            series.setdefault(label, {})[name] = (t2, fc)
        series.setdefault("_true", {})[name] = (ts, true)
    true = np.concatenate(all_true)
    peak = np.concatenate(all_peak)
    for label, parts in all_pred.items():
        rows.append({"Model": label, **_metrics(np.concatenate(parts), true, peak)})
    return pd.DataFrame(rows), series


def replay(cfg, df, pipe, policy_factory, precomputed=None):
    """Closed-loop replay of the test days for every container.

    At tick t the policy only sees: observed usage at t-1, forecast made at t-1, burst score at t-1.
    The limit it returns is in force during tick t, and we compare it with the true demand at t.
    """
    tr_end, va_end, end = split_idx(cfg)
    out = {}
    pre = precomputed if precomputed is not None else {}
    for name in cfg.containers:
        d = df[df.container == name].reset_index(drop=True)
        if name not in pre:
            ts, fc, fm, b = pipe.forecast_series(name, d, va_end - 1, end)
            pre[name] = (fc, fm, b)
        fc, fm, b = pre[name]
        thr = pipe.burst.threshold(name)
        cpu, mem = d.cpu.values, d.mem.values
        pol = policy_factory(name, cpu[:tr_end], mem[:tr_end])
        lim_c, lim_m, reasons = [], [], []
        for k, t in enumerate(range(va_end, end)):
            c, m = pol.decide(cpu[t - 1], mem[t - 1], fc[k], fm[k], b[k], thr)
            lim_c.append(c); lim_m.append(m)
            reasons.append(getattr(pol, "last_reason", ""))
        out[name] = {
            "t": np.arange(va_end, end), "demand_cpu": cpu[va_end:end],
            "demand_mem": mem[va_end:end], "limit_cpu": np.array(lim_c),
            "limit_mem": np.array(lim_m), "fc_cpu": fc, "burst": b, "thr": thr,
            "reasons": reasons,
        }
    return out, pre


def score(cfg, res):
    dc = np.concatenate([r["demand_cpu"] for r in res.values()])
    lc = np.concatenate([r["limit_cpu"] for r in res.values()])
    dm = np.concatenate([r["demand_mem"] for r in res.values()])
    lm = np.concatenate([r["limit_mem"] for r in res.values()])
    hours = cfg.tick_seconds / 3600
    resizes, ooms = 0, 0
    for r in res.values():
        ch = (np.abs(np.diff(r["limit_cpu"])) > 1e-3) | (np.abs(np.diff(r["limit_mem"])) > 1e-3)
        resizes += int(ch.sum())
        o = r["demand_mem"] > r["limit_mem"]
        ooms += int(np.sum(o[1:] & ~o[:-1]) + o[0])
    viol = dc > lc * (1 + cfg.slo_tolerance)
    return {
        "SLO violations %": float(100 * viol.mean()),
        "Unserved CPU (core-h)": float(np.clip(dc - lc, 0, None).sum() * hours),
        "Wasted CPU (core-h)": float(np.clip(lc - dc, 0, None).sum() * hours),
        "Avg CPU utilisation %": float(100 * np.mean(np.minimum(dc, lc) / lc)),
        "Reserved CPU (core-h)": float(lc.sum() * hours),
        "Wasted memory (GB-h)": float(np.clip(lm - dm, 0, None).sum() * hours),
        "OOM kills": ooms,
        "Limit changes": resizes,
    }


def factories(cfg, cpu_cap=4.0, mem_cap=6.0, headroom=None):
    def static_level(c_tr, m_tr):
        return float(np.percentile(c_tr, 95) * 1.2), float(m_tr.max() * 1.1)

    return {
        "Static": lambda n, c, m: StaticPolicy(*static_level(c, m)),
        "Reactive": lambda n, c, m: ReactivePolicy(cfg, cpu_cap, mem_cap, static_level(c, m)),
        "Predictive (ConLAR-style)": lambda n, c, m: PredictivePolicy(cfg, cpu_cap, mem_cap, headroom),
        "ProDock (ours)": lambda n, c, m: ProDockPolicy(cfg, cpu_cap, mem_cap, static_level(c, m), headroom),
    }
