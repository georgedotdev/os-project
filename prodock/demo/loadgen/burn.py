"""Trace-driven load generator: runs INSIDE a container and really consumes CPU and memory.

Every tick (one simulated minute, compressed to a few real seconds) it looks up its row in
demo/trace.csv and
  - burns that many CPU cores: N worker processes, each busy for a fraction of every 50 ms
    (a 2.4-core target = two workers at 100% + one at 40%), and
  - holds that much memory: 16 MB blocks of real, touched pages.
If the container's CPU limit is lower than the target, the kernel's CFS scheduler throttles
these workers: that is real throttling, counted by the kernel in cpu.stat. If the memory limit
is lower than the target, the kernel OOM-kills the container and Docker restarts it; it then
rejoins the shared clock at the current tick.

All containers and ProDock read the same demo/clock.json, so they agree on which tick it is.
Only the Python standard library is used, so the stock python:slim image is enough.
"""
import csv
import json
import math
import multiprocessing as mp
import os
import random
import time

DEMO = os.environ.get("DEMO_DIR", "/demo")
TRACE = os.environ["TRACE"]
PERIOD = 0.05                      # duty-cycle period of each worker (seconds)
BLOCK = 16 * 1024 * 1024           # memory is held in 16 MB blocks


def worker(i, share):
    """Burn min(1, max(0, share - i)) of one core, forever."""
    random.seed(i)
    time.sleep(random.random() * PERIOD)          # spread workers out within the period
    while True:
        t = time.perf_counter()
        f = min(1.0, max(0.0, share.value - i))
        if f > 0.0:
            end = t + f * PERIOD
            while time.perf_counter() < end:
                pass
        rest = t + PERIOD - time.perf_counter()
        if rest > 0:
            time.sleep(rest)


def cgroup_mem():
    """Bytes this container's cgroup is charged for, minus reclaimable page cache."""
    try:
        cur = int(open("/sys/fs/cgroup/memory.current").read())
        stat = dict(l.split() for l in open("/sys/fs/cgroup/memory.stat"))
        return cur - int(stat.get("inactive_file", 0))
    except (OSError, ValueError):
        return None


def main():
    clock = json.load(open(os.path.join(DEMO, "clock.json")))
    rows = {}
    with open(os.path.join(DEMO, "trace.csv")) as f:
        for r in csv.DictReader(f):
            if r["container"] == TRACE:
                rows[int(r["tick"])] = (float(r["cpu"]), float(r["mem"]))
    last = max(rows)
    n_workers = math.ceil(max(c for c, _ in rows.values()))
    share = mp.Value("d", 0.0, lock=False)
    for i in range(n_workers):
        mp.Process(target=worker, args=(i, share), daemon=True).start()

    start, tick_s, k0 = clock["start_epoch"], clock["tick_seconds"], clock["start_tick"]
    time.sleep(0.5)
    base = cgroup_mem() or 60 * 1024 ** 2           # what Python + workers use on their own
    blocks = []
    print(f"loadgen {TRACE}: {len(rows)} ticks, {n_workers} workers, base {base / 2**20:.0f} MB",
          flush=True)
    while time.time() < start:
        time.sleep(0.1)

    while True:
        k = k0 + int((time.time() - start) // tick_s)
        cpu, mem = rows.get(min(k, last), (0.0, 0.0))
        if k > last:
            cpu = 0.0                                # trace finished: go idle, keep running
        share.value = cpu

        cur = cgroup_mem()
        if cur is not None:                          # slowly track the non-block baseline
            base = 0.9 * base + 0.1 * max(0, cur - len(blocks) * BLOCK)
        want = max(0, round((mem * 1024 ** 3 - base) / BLOCK))
        while len(blocks) < want:
            blocks.append(b"\x01" * BLOCK)           # bytes filled with 1s: pages really touched
        del blocks[want:]

        print(f"tick {k} cpu={cpu:.2f} mem={mem:.2f}GB", flush=True)
        time.sleep(max(0.0, start + (k - k0 + 1) * tick_s - time.time()))


if __name__ == "__main__":
    main()
