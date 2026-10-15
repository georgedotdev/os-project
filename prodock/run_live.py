"""Run ProDock against REAL Docker containers on this machine.

LIVE DEMO (recommended, see demo/README.md):
    python demo/demo.py up                  # start the trace-driven containers
    python run_live.py --demo --apply       # ProDock + a reactive baseline really resize them
    python demo/dashboard.py                # live chart, in a second terminal
    python demo/demo.py down

  In demo mode one tick = one simulated minute (2 real seconds by default), read from
  demo/clock.json. That keeps the time scale identical to what the model was trained on:
  30 ticks of history, a forecast of the peak over the next 3 ticks.

GENERIC MODE (any containers named like the model's, wall clock):
    python run_live.py --check
    python run_live.py --containers web-frontend api-service [--apply] [--interval 60]
  The model was trained on one reading per MINUTE, so --interval should stay at 60 s here.
  For other workloads: collect a few days with `--record`, retrain on that CSV, then run.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque

from prodock.allocator import ProDockPolicy, ReactivePolicy
from prodock.docker_io import DockerCollector, DockerExecutor
from prodock.pipeline import Pipeline
from prodock.store import LiveLog, read_live, scoreboard

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT, "demo")
MODEL = os.path.join(ROOT, "models", "prodock_pipeline.pkl")
CPU_CAP, MEM_CAP = 4.0, 6.0         # same caps as the simulated evaluation (evaluate.factories)
SHORT = {"prodock": "PD", "reactive": "RE", "static": "ST"}


def say(*a):
    print(*a, flush=True)


def check(names, models=None):
    """Quick diagnostics: can we talk to Docker, find the containers, and read stats?"""
    try:
        col = DockerCollector(names)
        say("[ok] connected to Docker, version", col.client.version().get("Version"))
    except Exception as e:
        say("[FAIL] cannot talk to Docker. Is Docker Desktop running?  ", e)
        return
    try:
        cs = col.containers()
    except Exception as e:
        say("[FAIL] could not find the containers:", e)
        say("       running containers:", [c.name for c in col.client.containers.list()])
        return
    for c in cs:
        t = time.time()
        col.sample(c)
        time.sleep(1)
        s = col.sample(c)
        say(f"[ok] {c.name}: status={c.status}  cpu={s['cpu']:.2f} cores  mem={s['mem']:.3f} GB  "
            f"limit={s['limit_cpu']:.2f} cores / {s['limit_mem']:.2f} GB  "
            f"throttled={100 * s['thr_ratio']:.0f}% of periods  docker cpus={s['ncpu']}  "
            f"({time.time() - t:.1f}s)")
    pipe = Pipeline.load(MODEL)
    for n in sorted(set(models or names)):
        say(f"[{'ok' if n in pipe.scales else 'FAIL'}] model knows '{n}'"
            + ("" if n in pipe.scales else f"  (known names: {list(pipe.scales)})"))


class Controller:
    """One container: its history buffer, its policy, and the limits it last set."""

    def __init__(self, name, policy, model_name, pipe, ex):
        self.name, self.policy, self.model = name, policy, model_name
        self.pipe, self.cfg, self.ex = pipe, pipe.cfg, ex
        self.hist = deque(maxlen=self.cfg.window + self.cfg.roll)
        self.pol = None
        self.applied = None

    def step(self, s, minute_of_day):
        """s = collector sample for the tick that just finished. Returns fields for the log."""
        self.hist.append((s["cpu"], s["mem"], s["net"], minute_of_day))
        out = {}
        if self.policy == "static":
            return out
        if self.policy == "prodock" and len(self.hist) < self.hist.maxlen:
            out["reason"] = f"warm-up {len(self.hist)}/{self.hist.maxlen}"
            return out
        if self.pol is None:
            cur = (s["limit_cpu"] or 1.0, s["limit_mem"] or 1.0)
            if self.policy == "prodock":
                self.pol = ProDockPolicy(self.cfg, CPU_CAP, MEM_CAP, cur)
                # remember the memory seen while filling the buffer (e.g. the last job's peak)
                self.pol.mem_hist.extend(h[1] for h in self.hist)
            else:
                self.pol = ReactivePolicy(self.cfg, CPU_CAP, MEM_CAP, cur)
            self.applied = cur

        fc = fm = b = thr = None
        if self.policy == "prodock":
            cpu, mem, net, md = zip(*self.hist)
            fc, fm, b = self.pipe.forecast_live(self.model, cpu, mem, net, md)
            thr = self.pipe.burst.threshold(self.model)
        lc, lm = self.pol.decide(s["cpu"], s["mem"], fc, fm, b, thr)

        dc, dm = lc - self.applied[0], lm - self.applied[1]
        changed = abs(dc) > 1e-3 or abs(dm) > 1e-3
        reason = getattr(self.pol, "last_reason", "") if self.policy == "prodock" else ""
        if not changed:
            reason = "hold"
        elif reason in ("", "hold"):            # only memory moved, or a reactive decision
            reason = "up" if dc > 1e-3 else "down" if dc < -1e-3 else "mem"
        # apply whenever EITHER limit moved (a memory-only change used to be skipped as "hold")
        cmd = self.ex.apply(self.name, lc, lm) if changed else ""
        if changed:
            self.applied = (lc, lm)
        out.update(fc_cpu=fc, fc_mem=fm, burst=b, burst_thr=thr, new_cpu=lc, new_mem=lm,
                   reason=reason, command=cmd)
        return out


def line(sim, name, policy, s, o, true_cpu):
    lim = s["limit_cpu"]
    miss = true_cpu is not None and true_cpu > lim * 1.02
    fc = f"fc={o['fc_cpu']:.2f}" if o.get("fc_cpu") is not None else " " * 7
    bs = ""
    if o.get("burst") is not None:
        bs = f"burst={o['burst']:.2f}{'!' if o['burst'] > o['burst_thr'] else ' '}"
    act = ""
    if o.get("new_cpu") is not None and o["reason"] != "hold":
        act = f"{o['reason']:>8} -> {o['new_cpu']:.2f} cores {o['new_mem']:.2f}G"
        if o["command"].startswith(("[refused]", "[dry-run]")):
            act += "  " + o["command"].split("]")[0] + "]"
    elif o.get("reason", "").startswith("warm"):
        act = o["reason"]
    return (f"[{sim}] {name:<22}{SHORT[policy]} use={s['used']:.2f} dem={s['cpu']:.2f} "
            f"lim={lim:.2f} thr={100 * s['thr_ratio']:3.0f}% {'MISS' if miss else '    '} "
            f"{fc} {bs:<11} {act}{'  ** CONTAINER DIED (OOM?) **' if s['died'] else ''}")


def print_scoreboard(path):
    sb = scoreboard(read_live(path))
    if not sb:
        return
    say(f"\n{'container':<24}{'policy':<10}{'ticks':>6}{'SLO miss':>10}{'kernel thr':>11}"
        f"{'avg limit':>10}{'avg demand':>11}{'wasted':>8}{'resizes':>8}{'deaths':>7}")
    for n, s in sb.items():
        say(f"{n:<24}{s['policy']:<10}{s['ticks']:>6}{s['slo']:>5} ({s['slo_pct']:4.1f}%)"
            f"{s['kernel']:>8}{s['reserved']:>12.2f}{s['demand']:>11.2f}{s['wasted']:>8.0f}"
            f"{s['resizes']:>8}{s['deaths']:>7}")
    say("SLO miss = true demand > limit + 2% (paper rule). kernel thr = ticks where >= 10% of "
        "CFS periods were throttled. wasted = core-minutes reserved but unused.")


def run(a):
    pipe = Pipeline.load(MODEL)
    cfg = pipe.cfg
    if a.demo:
        clock = json.load(open(os.path.join(DEMO_DIR, "clock.json")))
        roles = {n: (r["policy"], r["trace"]) for n, r in clock["containers"].items()}
        tick_s, start, k0 = clock["tick_seconds"], clock["start_epoch"], clock["start_tick"]
        end_tick, tpd = clock["end_tick"], clock["ticks_per_day"]
        truth = {}
        with open(os.path.join(DEMO_DIR, "trace.csv")) as f:
            for r in csv.DictReader(f):
                truth[(r["container"], int(r["tick"]))] = (float(r["cpu"]), float(r["mem"]))
        log_path = os.path.join(ROOT, "results", "live_demo.sqlite")
    else:
        roles = {n: ("prodock", n) for n in a.containers}
        tick_s, start, k0, end_tick, tpd = a.interval, time.time(), 0, None, cfg.ticks_per_day
        truth = {}
        log_path = os.path.join(ROOT, "results", "live_decisions.sqlite")
        if a.interval < 30:
            say(f"WARNING: the model expects one reading per minute; --interval {a.interval} "
                "makes its 30-tick window and 3-tick forecast far shorter than it was trained on.")

    unknown = [t for p, t in roles.values() if p == "prodock" and t not in pipe.scales]
    if unknown:
        sys.exit(f"The model does not know {unknown}. Use one of {list(pipe.scales)} or retrain.")

    col = DockerCollector(list(roles))
    ex = None
    ctrls = {}
    log = LiveLog(log_path, fresh=True)
    writer = csv.writer(open(a.record, "a", newline="")) if a.record else None
    say(f"ProDock live | {'demo clock' if a.demo else 'wall clock'} | 1 tick = {tick_s}s | "
        f"{'APPLYING changes' if a.apply else 'dry run (add --apply to resize)'}")
    for n, (p, t) in roles.items():
        say(f"  {n:<24} {p:<9} (trace/model: {t})")
    say(f"ProDock needs {cfg.window + cfg.roll} ticks of history before its first decision. "
        "Ctrl+C to stop.\n")

    if time.time() < start:
        say(f"waiting {start - time.time():.0f}s for the shared clock to start ...")
        time.sleep(max(0.0, start - time.time()))
    for c in col.containers():                      # first snapshot: nothing to compare yet
        col.sample(c)
    # sample a little after each tick boundary, once the load generators have switched rows
    nxt = start + (math.floor((time.time() - start) / tick_s) + 1 + 0.15) * tick_s
    try:
        while True:
            time.sleep(max(0.0, nxt - time.time()))
            now = time.time()
            sim = k0 + math.floor((now - start) / tick_s) - 1   # the tick that just finished
            if a.demo:
                mod = sim % tpd                                  # simulated time of day
            else:
                lt = time.localtime(now)
                mod = lt.tm_hour * 60 + lt.tm_min
            for c in col.containers():
                try:
                    s = col.sample(c)
                except Exception as e:                           # e.g. restarting after OOM
                    say(f"[{sim}] {c.name}: no stats ({str(e)[:80]})")
                    continue
                if s is None:
                    continue
                name = s["container"]
                policy, trace = roles[name]
                if ex is None:
                    ex = DockerExecutor(dry_run=not a.apply, max_cpus=s["ncpu"])
                if name not in ctrls:
                    ctrls[name] = Controller(name, policy, trace, pipe, ex)
                if writer:
                    writer.writerow([now, mod, name, s["cpu"], s["mem"], s["net"]])
                o = ctrls[name].step(s, mod)
                tc, tm = truth.get((trace, sim), (None, None))
                log.write(wall_ts=now, sim_tick=sim, container=name, policy=policy,
                          used_cpu=s["used"], demand_cpu=s["cpu"], mem=s["mem"],
                          limit_cpu=s["limit_cpu"], limit_mem=s["limit_mem"],
                          thr_ratio=s["thr_ratio"], died=int(s["died"]),
                          true_cpu=tc, true_mem=tm, **o)
                say(line(sim, name, policy, s, o, tc))
            log.commit()
            if end_tick is not None and sim >= end_tick:
                say("\ntrace finished.")
                break
            nxt += tick_s
            if nxt < time.time():                                # fell behind: skip ahead
                late = math.ceil((time.time() - nxt) / tick_s)
                say(f"  (sampling took too long, skipping {late} tick(s))")
                nxt += late * tick_s
    except KeyboardInterrupt:
        say("\nstopped.")
    print_scoreboard(log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="follow demo/clock.json (live demo)")
    ap.add_argument("--containers", nargs="+", default=["web-frontend", "api-service"])
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between ticks")
    ap.add_argument("--apply", action="store_true", help="actually run docker update")
    ap.add_argument("--record", default=None, help="also append raw samples to this CSV")
    ap.add_argument("--check", action="store_true", help="run diagnostics and exit")
    a = ap.parse_args()
    if a.check:
        names, models = a.containers, None
        if a.demo:
            roles = json.load(open(os.path.join(DEMO_DIR, "clock.json")))["containers"]
            names = list(roles)
            models = [r["trace"] for r in roles.values() if r["policy"] == "prodock"]
        check(names, models)
        return
    run(a)


if __name__ == "__main__":
    main()
