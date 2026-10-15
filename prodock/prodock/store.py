"""Operator path: every decision is written to SQLite so it can be audited and plotted."""
import sqlite3


class DecisionLog:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS decisions(
            ts REAL, container TEXT, policy TEXT, obs_cpu REAL, obs_mem REAL,
            fc_cpu REAL, fc_mem REAL, burst REAL, limit_cpu REAL, limit_mem REAL,
            reason TEXT, command TEXT)""")

    def write(self, **r):
        self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            r["ts"], r["container"], r["policy"], r["obs_cpu"], r["obs_mem"], r["fc_cpu"],
            r["fc_mem"], r["burst"], r["limit_cpu"], r["limit_mem"], r.get("reason", ""),
            r.get("command", "")))

    def commit(self):
        self.db.commit()


LIVE_COLS = [
    ("wall_ts", "REAL"), ("sim_tick", "INTEGER"), ("container", "TEXT"), ("policy", "TEXT"),
    ("used_cpu", "REAL"), ("demand_cpu", "REAL"), ("mem", "REAL"),
    ("limit_cpu", "REAL"), ("limit_mem", "REAL"),        # limits Docker reports were in force
    ("thr_ratio", "REAL"), ("died", "INTEGER"),
    ("true_cpu", "REAL"), ("true_mem", "REAL"),          # from the trace, used ONLY for scoring
    ("fc_cpu", "REAL"), ("fc_mem", "REAL"), ("burst", "REAL"), ("burst_thr", "REAL"),
    ("new_cpu", "REAL"), ("new_mem", "REAL"), ("reason", "TEXT"), ("command", "TEXT"),
]


class LiveLog:
    """One row per container per tick of a live run. The dashboard reads it while it is written."""

    def __init__(self, path, fresh=True):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")        # lets the dashboard read concurrently
        if fresh:
            self.db.execute("DROP TABLE IF EXISTS live")
        cols = ", ".join(f"{n} {t}" for n, t in LIVE_COLS)
        self.db.execute(f"CREATE TABLE IF NOT EXISTS live({cols})")
        self.db.commit()

    def write(self, **r):
        names = [n for n, _ in LIVE_COLS]
        self.db.execute(f"INSERT INTO live VALUES ({','.join('?' * len(names))})",
                        [r.get(n) for n in names])

    def commit(self):
        self.db.commit()


def read_live(path):
    """All live rows as a list of dicts (empty if the run has not started yet)."""
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM live ORDER BY sim_tick, container")]
        db.close()
        return rows
    except sqlite3.Error:
        return []


def scoreboard(rows, slo_tolerance=0.02, kernel_threshold=0.10):
    """Per-container live results, counted from the first tick ProDock made a decision.

    SLO miss   : true demand (trace) > CPU limit in force * 1.02   (same rule as the simulation)
    kernel thr.: >= 10% of the kernel's CFS periods in that tick were throttled (cpu.stat)
    wasted     : sum of (limit - true demand) over ticks, in core-minutes (1 tick = 1 sim minute)
    """
    started = [r["sim_tick"] for r in rows if r["policy"] == "prodock" and r["fc_cpu"] is not None]
    if not started:
        return {}
    t0 = min(started)
    out = {}
    for r in rows:
        if r["sim_tick"] < t0 or r["true_cpu"] is None:
            continue
        s = out.setdefault(r["container"], dict(policy=r["policy"], ticks=0, slo=0, kernel=0,
                                                reserved=0.0, demand=0.0, wasted=0.0,
                                                resizes=0, deaths=0))
        lim = r["limit_cpu"] or 0.0
        s["ticks"] += 1
        s["slo"] += r["true_cpu"] > lim * (1 + slo_tolerance)
        s["kernel"] += (r["thr_ratio"] or 0) >= kernel_threshold
        s["reserved"] += lim
        s["demand"] += r["true_cpu"]
        s["wasted"] += max(0.0, lim - r["true_cpu"])
        s["resizes"] += (r["command"] or "").startswith("[applied]")
        s["deaths"] += r["died"] or 0
    for s in out.values():
        n = max(1, s["ticks"])
        s["slo_pct"] = 100 * s["slo"] / n
        s["reserved"] /= n
        s["demand"] /= n
    return out
