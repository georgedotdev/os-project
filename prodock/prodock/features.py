"""L2 Feature extractor: raw samples -> one row of numbers per container per tick.

Every feature only uses the CURRENT and PAST samples, never the future, so the exact same
function works offline (training) and online (live Docker host).
"""
import numpy as np

FEATURES = [
    "cpu", "mem", "net",
    "cpu_delta", "cpu_roll_mean", "cpu_roll_std",
    "mem_delta", "tod_sin", "tod_cos",
]
BURST_COL = "burst_score"          # appended by Stage 1 -> 10th column


def build(cpu, mem, net, minute_of_day, scale, roll=10, tpd=1440):
    """cpu/mem/net are raw arrays for ONE container. scale = (cpu_s, mem_s, net_s)."""
    cs, ms, ns = scale
    c, m, n = cpu / cs, mem / ms, net / ns
    cd = np.diff(c, prepend=c[0])
    md = np.diff(m, prepend=m[0])
    # fast rolling via cumulative sums
    k = roll
    cs1 = np.cumsum(np.insert(c, 0, 0.0))
    cs2 = np.cumsum(np.insert(c * c, 0, 0.0))
    idx = np.arange(len(c))
    lo = np.maximum(0, idx - k + 1)
    cnt = idx - lo + 1
    mean = (cs1[idx + 1] - cs1[lo]) / cnt
    var = (cs2[idx + 1] - cs2[lo]) / cnt - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))
    ang = 2 * np.pi * minute_of_day / tpd
    return np.column_stack([c, m, n, cd, mean, std, md, np.sin(ang), np.cos(ang)]).astype(np.float64)


def targets(cpu, mem, scale, horizon):
    """y[t] = peak of the next `horizon` ticks (t+1 .. t+horizon), scaled."""
    cs, ms, _ = scale
    n = len(cpu)
    yc = np.full(n, np.nan)
    ym = np.full(n, np.nan)
    for t in range(n - horizon):
        yc[t] = cpu[t + 1: t + 1 + horizon].max() / cs
        ym[t] = mem[t + 1: t + 1 + horizon].max() / ms
    return np.column_stack([yc, ym])


def windows(F, Y, w, start, stop):
    """Stack sliding windows ending at t for t in [start, stop)."""
    ts = [t for t in range(max(start, w - 1), stop) if not np.isnan(Y[t]).any()]
    X = np.stack([F[t - w + 1: t + 1] for t in ts])
    return X, Y[ts], np.array(ts)
