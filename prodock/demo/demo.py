"""Start / stop the live demo containers.

    python demo/demo.py up                 # 20 sim-minutes of test day 6, 1 sim-minute = 2 s
    python demo/demo.py up --tick 1        # faster: 1 sim-minute = 1 s
    python demo/demo.py up --no-batch      # only the three web-frontend copies (less CPU)
    python demo/demo.py down

`up` does four things:
  1. regenerates the simulated workload (same seed as the experiment) and exports the slice to
     replay into demo/trace.csv (it starts 40 ticks early so ProDock can fill its history),
  2. computes the static starting limits exactly like the simulation (95th pct of training CPU
     x 1.2, training memory max x 1.1),
  3. writes demo/clock.json: the shared clock that containers and ProDock both follow,
  4. runs `docker compose up -d --force-recreate`.
By default the clock starts at tick 7200, the first minute of the test days, exactly where the
simulated evaluation starts. ProDock fills its 40-tick history during batch-worker's job at
7200-7225, so its memory floor knows how big a job gets (start the window between two jobs and
the first job gets OOM-killed: the floor has nothing to remember yet). ProDock's first decision
is at 7240; web-frontend has flash crowds at 7369 and 7495; batch-worker jobs at 7320 and 7440.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from prodock import workload                       # noqa: E402
from prodock.config import Config                  # noqa: E402

COMPOSE = ["docker", "compose", "-f", os.path.join(HERE, "docker-compose.yml"),
           "-p", "prodock-demo"]
ROLES = {                                           # container -> (policy, trace it replays)
    "web-frontend": ("prodock", "web-frontend"),
    "web-frontend-reactive": ("reactive", "web-frontend"),
    "web-frontend-static": ("static", "web-frontend"),
    "batch-worker": ("prodock", "batch-worker"),
}


def up(a):
    cfg = Config()
    df = workload.generate(cfg)
    tr_end = cfg.train_days * cfg.ticks_per_day
    warm = cfg.window + cfg.roll
    first, last = a.start - warm, a.start + int(a.minutes * 60 / a.tick)
    names = [n for n in ROLES if not (a.no_batch and n == "batch-worker")]
    traces = sorted({ROLES[n][1] for n in names})

    part = df[df.container.isin(traces) & (df.tick >= first) & (df.tick <= last)]
    part[["tick", "minute_of_day", "container", "cpu", "mem"]].to_csv(
        os.path.join(HERE, "trace.csv"), index=False, float_format="%.4f")

    env = dict(os.environ)
    for trace, key in [("web-frontend", "WEB"), ("batch-worker", "BATCH")]:
        d = df[df.container == trace].iloc[:tr_end]
        env[f"{key}_CPUS"] = f"{np.percentile(d.cpu, 95) * 1.2:.2f}"
        env[f"{key}_MEM"] = f"{int(d.mem.max() * 1.1 * 1024)}m"
    print("static starting limits:",
          {k: env[k] for k in ["WEB_CPUS", "WEB_MEM", "BATCH_CPUS", "BATCH_MEM"]})

    print("pulling the image (first time only) ...")
    subprocess.run(COMPOSE + ["pull", "-q"] + names, check=True, env=env)
    clock = {
        "start_epoch": time.time() + a.lead,
        "tick_seconds": a.tick,
        "start_tick": first,
        "end_tick": last,
        "ticks_per_day": cfg.ticks_per_day,
        "containers": {n: {"policy": ROLES[n][0], "trace": ROLES[n][1]} for n in names},
    }
    with open(os.path.join(HERE, "clock.json"), "w") as f:
        json.dump(clock, f, indent=2)
    subprocess.run(COMPOSE + ["up", "-d", "--force-recreate", "--remove-orphans"] + names,
                   check=True, env=env)
    secs = (last - first) * a.tick
    print(f"\nreplaying ticks {first}..{last} ({last - first} sim-minutes in {secs / 60:.0f} real "
          f"minutes), starting in {a.lead:.0f}s.")
    print("now run:   python run_live.py --demo --apply")
    print("and:       python demo/dashboard.py")


def down(a):
    subprocess.run(COMPOSE + ["down", "-t", "2"], check=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    u.add_argument("--start", type=int, default=7240, help="first tick ProDock decides on")
    u.add_argument("--minutes", type=float, default=20, help="real minutes to replay")
    u.add_argument("--tick", type=float, default=2.0, help="real seconds per simulated minute")
    u.add_argument("--lead", type=float, default=15, help="seconds before the clock starts")
    u.add_argument("--no-batch", action="store_true")
    sub.add_parser("down")
    a = ap.parse_args()
    {"up": up, "down": down}[a.cmd](a)


if __name__ == "__main__":
    main()
