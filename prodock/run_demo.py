"""Live terminal demo on the simulated host (no Docker needed). Good for showing in the review.

    python run_demo.py                  # replays test data, 20 ticks per second
    python run_demo.py --speed 5 --container web-frontend
Run `python run_experiment.py` first so the trained model exists.
"""
import argparse
import os
import time

from prodock import workload
from prodock.allocator import ProDockPolicy
from prodock.docker_io import DockerExecutor
from prodock.evaluate import split_idx
from prodock.pipeline import Pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="web-frontend")
    ap.add_argument("--speed", type=float, default=20, help="ticks per second")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--start", type=int, default=100, help="offset into the test days")
    a = ap.parse_args()

    pipe = Pipeline.load(os.path.join(ROOT, "models", "prodock_pipeline.pkl"))
    cfg = pipe.cfg
    df = workload.generate(cfg)
    d = df[df.container == a.container].reset_index(drop=True)
    _, va_end, end = split_idx(cfg)
    t0 = min(va_end + a.start, end - a.ticks - 1)
    thr = pipe.burst.threshold(a.container)
    pol = ProDockPolicy(cfg, 4.0, 6.0, start=(1.5, 1.5))
    ex = DockerExecutor(dry_run=True)
    need = cfg.window + cfg.roll
    misses = 0
    print(f"ProDock live demo | container={a.container} | burst threshold={thr:.2f}\n")
    print(f"{'min':>5} {'demand':>7} {'forecast':>8} {'burst':<10} {'limit':>6}  status   action")
    for t in range(t0, t0 + a.ticks):
        h = d.iloc[t - need:t]
        fc, fm, b = pipe.forecast_live(a.container, h.cpu, h.mem, h.net, h.minute_of_day)
        c, m = pol.decide(h.cpu.iloc[-1], h.mem.iloc[-1], fc, fm, b, thr)
        dmd = d.cpu.iloc[t]
        miss = dmd > c * (1 + cfg.slo_tolerance)
        misses += miss
        bar = "#" * int(b * 10)
        status = "THROTTLE" if miss else "ok"
        act = ex.apply(a.container, c, m) if pol.last_reason != "hold" else ""
        tag = f"{pol.last_reason:>8}" if act else ""
        print(f"{t:>5} {dmd:7.2f} {fc:8.2f} {bar:<10} {c:6.2f}  {status:<8} {tag} {act[10:].rsplit(' ', 1)[0] if act else ''}")
        time.sleep(1 / a.speed)
    print(f"\n{a.ticks} minutes replayed, {misses} throttled ({100 * misses / a.ticks:.1f}%)")


if __name__ == "__main__":
    main()
