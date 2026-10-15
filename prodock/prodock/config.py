"""All tunable numbers live here so they are easy to find and explain."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- host (the machine running Docker) ----
    host_cpus: float = 8.0          # cores
    host_mem_gb: float = 16.0       # GB

    # ---- data / time ----
    tick_seconds: int = 60          # simulation: one row per minute
    days: int = 7
    train_days: int = 4
    val_days: int = 1               # remaining days are the test set
    seed: int = 7

    # ---- L2 feature extractor ----
    roll: int = 10                  # rolling window for mean / std features

    # ---- Stage 2 LSTM ----
    window: int = 30                # rows the LSTM looks at (sliding buffer)
    horizon: int = 3                # predict the PEAK over the next 3 ticks
    hidden: int = 32
    epochs: int = 25
    batch: int = 128
    lr: float = 3e-3
    patience: int = 5

    # ---- L3 allocator (ProDock policy) ----
    base_headroom: float = 0.15     # +15% on top of the forecast
    burst_headroom: float = 0.35    # extra headroom when Stage 1 says "burst"
    min_cpu: float = 0.10
    min_mem_gb: float = 0.128
    scale_down_ratio: float = 0.85  # only shrink if new target < 85% of current limit
    cooldown_ticks: int = 5         # wait this long after a change before shrinking
    mem_safety: float = 1.20        # memory never set below 120% of forecast/usage
    mem_lookback: int = 180         # ...or below 120% of the highest usage in the last 3 h
    up_buffer: float = 1.05         # when growing, add 5% extra so small creeps don't resize again

    # ---- SLO definition used in evaluation ----
    slo_tolerance: float = 0.02     # demand > limit * 1.02 counts as throttled

    containers: list = field(default_factory=lambda: [
        "web-frontend", "api-service", "batch-worker", "cache-db"])

    @property
    def ticks_per_day(self) -> int:
        return 86400 // self.tick_seconds
