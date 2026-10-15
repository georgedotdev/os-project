"""L1 Collector and L4 Executor for a REAL Docker host.

Collector reads `docker stats` for each container through the official `docker` SDK:
  cpu usage (cores), throttled time (so we can estimate TRUE demand when capped),
  memory working set, network traffic, and the kernel's CFS throttling counters
  (cpu.stat: nr_periods / nr_throttled), which tell us how often the limit actually bit.
It asks for a single fast snapshot (one_shot) and works out CPU use itself from the
difference between this snapshot and the previous one, so each tick takes well under a second.

Executor runs `docker update --cpus X --memory Y` through the docker command line.
(The Python SDK cannot set --cpus, and mixing its CPU-quota option with the --cpus value that
docker compose sets makes Docker refuse the update.)
Dry-run is ON by default: it prints the command instead of running it.
"""
import shutil
import subprocess
import time


class DockerCollector:
    def __init__(self, names=None, label=None):
        import docker  # imported here so the simulation works without Docker installed
        self.client = docker.from_env()
        self.names = names
        self.label = label
        self._prev = {}       # container -> (time, cpu_total_ns, system_ns, throttled_ns, rx, periods, thr_periods)
        self._restarts = {}   # container -> RestartCount last seen (a jump means it died, e.g. OOM kill)

    def containers(self):
        if self.names:
            return [self.client.containers.get(n) for n in self.names]
        filt = {"label": self.label} if self.label else {}
        return self.client.containers.list(filters=filt)

    def _raw(self, c):
        try:
            return c.stats(stream=False, one_shot=True)
        except TypeError:                     # very old SDK without one_shot
            return c.stats(stream=False)

    def sample(self, c):
        """Returns a dict, or None on the very first call for a container (nothing to compare yet)."""
        s = self._raw(c)
        now = time.time()
        cpu = s.get("cpu_stats", {})
        total = cpu.get("cpu_usage", {}).get("total_usage", 0)
        system = cpu.get("system_cpu_usage", 0)
        td = cpu.get("throttling_data", {}) or {}
        thr = td.get("throttled_time", 0)
        periods, thr_periods = td.get("periods", 0), td.get("throttled_periods", 0)
        ncpu = cpu.get("online_cpus") or len(cpu.get("cpu_usage", {}).get("percpu_usage") or [1])
        rx = sum(v.get("rx_bytes", 0) for v in (s.get("networks") or {}).values())
        mem = s.get("memory_stats", {})
        st = mem.get("stats", {}) or {}
        usage = mem.get("usage", 0) - st.get("inactive_file", st.get("total_inactive_file", 0))

        restarts = c.attrs.get("RestartCount", 0) or 0
        died = restarts > self._restarts.get(c.name, restarts)
        self._restarts[c.name] = restarts

        prev = self._prev.get(c.name)
        self._prev[c.name] = (now, total, system, thr, rx, periods, thr_periods)
        if prev is None:
            return None
        t0, total0, system0, thr0, rx0, periods0, thr_periods0 = prev
        dt = max(1e-3, now - t0)
        sys_delta = system - system0
        if sys_delta > 0:
            cores = (total - total0) / sys_delta * ncpu
        else:                                  # fall back to wall-clock time
            cores = (total - total0) / 1e9 / dt
        cores = max(0.0, cores)                # counters reset when a container restarts
        throttled_cores = max(0, thr - thr0) / 1e9 / dt
        dp = periods - periods0
        host = c.attrs.get("HostConfig", {})
        limit_cpu = (host.get("NanoCpus") or 0) / 1e9
        if not limit_cpu and host.get("CpuQuota", 0) > 0:
            limit_cpu = host["CpuQuota"] / (host.get("CpuPeriod") or 100000)
        return {
            "container": c.name,
            "status": c.status,
            "used": cores,                              # what it actually got
            "cpu": cores + throttled_cores,             # demand estimate in cores
            "mem": max(usage, 0) / 1024 ** 3,           # GB
            "net": max(0.0, (rx - rx0) / dt / 1024),    # KB/s
            "thr_ratio": (thr_periods - thr_periods0) / dp if dp > 0 else 0.0,
            "died": died,
            "limit_cpu": limit_cpu,
            "limit_mem": (host.get("Memory") or 0) / 1024 ** 3,
            "ncpu": ncpu,
        }


class DockerExecutor:
    def __init__(self, dry_run=True, max_cpus=None):
        self.dry_run = dry_run
        self.max_cpus = max_cpus
        self.docker = shutil.which("docker") or "docker"

    def apply(self, name, cpus, mem_gb):
        if self.max_cpus:
            cpus = min(cpus, self.max_cpus)
        mib = max(6, int(mem_gb * 1024))       # Docker's minimum memory limit is 6 MB
        args = ["update", "--cpus", f"{cpus:.2f}", "--memory", f"{mib}m",
                "--memory-swap", f"{mib}m", name]
        cmd = "docker " + " ".join(args)
        if self.dry_run:
            return f"[dry-run] {cmd}"
        r = subprocess.run([self.docker] + args, capture_output=True, text=True)
        if r.returncode == 0:
            return f"[applied] {cmd}"
        # The kernel can refuse a memory limit (e.g. below what the container holds right now).
        # Still apply the CPU half so the container is not left starved.
        r2 = subprocess.run([self.docker, "update", "--cpus", f"{cpus:.2f}", name],
                            capture_output=True, text=True)
        return (f"[refused] {cmd}  ({r.stderr.strip()[:120]})"
                f"  cpu-only: {'applied' if r2.returncode == 0 else 'refused'}")
