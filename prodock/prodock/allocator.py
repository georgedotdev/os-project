"""L3 Allocator: turns forecasts into CPU / memory limits.

Four policies share one interface so they can be compared fairly:
  StaticPolicy     - what most people do: a fixed limit set once by hand
  ReactivePolicy   - threshold rules (like Kubernetes HPA / VPA style): react after the fact
  PredictivePolicy - ConLAR-style: LSTM forecast + fixed headroom, applied every tick
  ProDockPolicy    - ours: forecast + burst-aware headroom + hysteresis + cooldown
decide(...) is called once per tick with information up to the previous tick.
"""
from collections import deque

from .config import Config


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class StaticPolicy:
    name = "Static"

    def __init__(self, cpu, mem):
        self.cpu, self.mem = cpu, mem

    def decide(self, obs_cpu, obs_mem, fc_cpu, fc_mem, burst, thr):
        return self.cpu, self.mem


class ReactivePolicy:
    name = "Reactive"

    def __init__(self, cfg: Config, cpu_cap, mem_cap, start):
        self.cfg, self.cpu_cap, self.mem_cap = cfg, cpu_cap, mem_cap
        self.cpu, self.mem = start

    def decide(self, obs_cpu, obs_mem, fc_cpu, fc_mem, burst, thr):
        u = obs_cpu / self.cpu
        if u > 0.80:
            self.cpu *= 1.5
        elif u < 0.30:
            self.cpu *= 0.75
        um = obs_mem / self.mem
        if um > 0.85:
            self.mem *= 1.3
        elif um < 0.40:
            self.mem = max(self.mem * 0.8, obs_mem * 1.1)
        self.cpu = _clip(self.cpu, self.cfg.min_cpu, self.cpu_cap)
        self.mem = _clip(self.mem, self.cfg.min_mem_gb, self.mem_cap)
        return self.cpu, self.mem


class PredictivePolicy:
    name = "Predictive (ConLAR-style)"

    def __init__(self, cfg: Config, cpu_cap, mem_cap, headroom=None):
        self.cfg, self.cpu_cap, self.mem_cap = cfg, cpu_cap, mem_cap
        self.h = cfg.base_headroom if headroom is None else headroom

    def decide(self, obs_cpu, obs_mem, fc_cpu, fc_mem, burst, thr):
        c = _clip(fc_cpu * (1 + self.h), self.cfg.min_cpu, self.cpu_cap)
        m = _clip(fc_mem * (1 + self.h), self.cfg.min_mem_gb, self.mem_cap)
        return c, m


class ProDockPolicy:
    name = "ProDock (ours)"

    def __init__(self, cfg: Config, cpu_cap, mem_cap, start, base_headroom=None):
        self.cfg, self.cpu_cap, self.mem_cap = cfg, cpu_cap, mem_cap
        self.cpu, self.mem = start
        self.base = cfg.base_headroom if base_headroom is None else base_headroom
        self.since_change = 10 ** 6
        self.last_reason = ""
        self.mem_hist = deque(maxlen=cfg.mem_lookback)   # memory cannot be throttled, only killed

    def headroom(self, burst, thr):
        if burst <= thr:
            return self.base
        level = (burst - thr) / max(1e-9, 1 - thr)          # 0..1, how far past the line
        return self.base + self.cfg.burst_headroom * (0.5 + 0.5 * level)

    def decide(self, obs_cpu, obs_mem, fc_cpu, fc_mem, burst, thr):
        cfg = self.cfg
        h = self.headroom(burst, thr)
        # never plan below what the container is using right now
        want_c = max(fc_cpu, obs_cpu) * (1 + h)
        self.mem_hist.append(obs_mem)
        want_m = max(fc_mem, max(self.mem_hist)) * cfg.mem_safety * (1 + (h - self.base))
        want_c = _clip(want_c, cfg.min_cpu, self.cpu_cap)
        want_m = _clip(want_m, cfg.min_mem_gb, self.mem_cap)

        changed = False
        reason = "hold"
        # scale UP immediately (being late is what causes SLO misses)
        if want_c > self.cpu:
            self.cpu = min(self.cpu_cap, want_c * cfg.up_buffer)
            changed, reason = True, ("burst-up" if burst > thr else "up")
        # scale DOWN only when clearly lower and after a quiet period (no thrashing)
        elif want_c < cfg.scale_down_ratio * self.cpu and self.since_change >= cfg.cooldown_ticks:
            self.cpu, changed, reason = want_c, True, "down"
        if want_m > self.mem:
            self.mem, changed = min(self.mem_cap, want_m * cfg.up_buffer), True
        elif want_m < cfg.scale_down_ratio * self.mem and self.since_change >= cfg.cooldown_ticks:
            self.mem, changed = want_m, True

        self.since_change = 0 if changed else self.since_change + 1
        self.last_reason = reason
        return self.cpu, self.mem
