"""Simulated Docker host.

We do not have a production cluster, so we generate a week of per-minute demand for four
containers with very different personalities. Each trace is TRUE DEMAND: what the container
would use if it were never limited. On a real host this is estimated as
usage + throttled time (Docker reports both in `docker stats`).
"""
import numpy as np
import pandas as pd

from .config import Config


def _bursts(rng, n, per_day, tpd, height, dur=(10, 35), ramp=(3, 7)):
    """Flash-crowd style spikes: short ramp up, plateau, decay."""
    out = np.zeros(n)
    count = rng.poisson(per_day * n / tpd)
    for _ in range(count):
        s = rng.integers(0, n - 60)
        r = rng.integers(*ramp)
        d = rng.integers(*dur)
        h = rng.uniform(*height)
        shape = np.concatenate([
            np.linspace(0, 1, r, endpoint=False),
            np.ones(d),
            np.linspace(1, 0, 8),
        ])
        e = min(n, s + len(shape))
        out[s:e] += h * shape[: e - s]
    return out


def generate(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    tpd = cfg.ticks_per_day
    n = cfg.days * tpd
    t = np.arange(n)
    hour = (t % tpd) / tpd * 24.0
    day = t // tpd
    weekend = (day % 7 >= 5).astype(float)
    frames = []

    def diurnal(peak_hour, width=4.5):
        d = np.minimum(np.abs(hour - peak_hour), 24 - np.abs(hour - peak_hour))
        return np.exp(-(d ** 2) / (2 * width ** 2))

    # 1. web-frontend: evening peak + flash crowds
    cpu = 0.35 + 1.25 * diurnal(20) + rng.normal(0, 0.06, n)
    cpu += _bursts(rng, n, per_day=2.5, tpd=tpd, height=(1.2, 2.4))
    mem = 0.45 + 0.35 * cpu + rng.normal(0, 0.02, n)
    net = 20 + 60 * cpu + rng.normal(0, 4, n)
    frames.append(("web-frontend", cpu, mem, net))

    # 2. api-service: lunchtime peak, quieter weekends, frequent small bursts
    cpu = (0.3 + 0.9 * diurnal(13, 3.5)) * (1 - 0.35 * weekend) + rng.normal(0, 0.05, n)
    cpu += _bursts(rng, n, per_day=5, tpd=tpd, height=(0.4, 1.0), dur=(5, 15))
    mem = 0.7 + 0.25 * cpu + rng.normal(0, 0.02, n)
    net = 10 + 45 * cpu + rng.normal(0, 3, n)
    frames.append(("api-service", cpu, mem, net))

    # 3. batch-worker: cron job every 2 hours, near idle otherwise
    phase = t % 120
    job = ((phase >= 0) & (phase < 25)).astype(float)
    ramp = np.clip(phase / 3.0, 0, 1)
    cpu = 0.08 + job * ramp * rng.uniform(2.2, 2.8) + rng.normal(0, 0.03, n)
    mem = 0.25 + job * 1.6 + rng.normal(0, 0.03, n)
    net = 2 + job * 15 + rng.normal(0, 1, n)
    frames.append(("batch-worker", cpu, mem, net))

    # 4. cache-db: steady CPU, memory creeps up then compacts nightly at 03:00
    cpu = 0.55 + 0.15 * diurnal(14, 6) + rng.normal(0, 0.04, n)
    cpu += _bursts(rng, n, per_day=0.7, tpd=tpd, height=(0.4, 0.8))
    since_compact = ((t - 3 * 60) % tpd) / tpd
    mem = 1.4 + 0.9 * since_compact + rng.normal(0, 0.015, n)
    net = 15 + 20 * cpu + rng.normal(0, 2, n)
    frames.append(("cache-db", cpu, mem, net))

    rows = []
    for name, c, m, nt in frames:
        rows.append(pd.DataFrame({
            "tick": t,
            "minute_of_day": t % tpd,
            "container": name,
            "cpu": np.clip(c, 0.01, cfg.host_cpus),
            "mem": np.clip(m, 0.05, cfg.host_mem_gb),
            "net": np.clip(nt, 0, None),
        }))
    return pd.concat(rows, ignore_index=True)
