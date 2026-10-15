"""Glue: L2 features -> Stage 1 burst score (29th-column style relay) -> Stage 2 LSTM.

Train once on historical traces, then call `forecast_series` (offline) or `forecast_live`
(one step, online) to get peak CPU / memory predictions in real units.
"""
import pickle
import numpy as np

from . import features as fx
from .burst import BurstDetector
from .lstm import LSTMRegressor


class Pipeline:
    def __init__(self, cfg, use_burst=True):
        self.cfg = cfg
        self.use_burst = use_burst
        self.scales = {}
        self.burst = BurstDetector(seed=cfg.seed)
        self.model = None

    # ---- helpers ----
    def rows(self, name, d):
        """L2 rows (+ burst column) for one container dataframe slice."""
        F = fx.build(d.cpu.values, d.mem.values, d.net.values, d.minute_of_day.values,
                     self.scales[name], self.cfg.roll, self.cfg.ticks_per_day)
        b = self.burst.score(name, F)
        if self.use_burst:
            F = np.column_stack([F, b])
        return F, b

    def fit(self, df, verbose=True):
        cfg = self.cfg
        tpd = cfg.ticks_per_day
        tr_end = cfg.train_days * tpd
        va_end = (cfg.train_days + cfg.val_days) * tpd
        Xs, Ys, Xv, Yv = [], [], [], []
        for name in cfg.containers:
            d = df[df.container == name].reset_index(drop=True)
            tr = d.iloc[:tr_end]
            self.scales[name] = (float(tr.cpu.max()), float(tr.mem.max()), float(tr.net.max()))
            F0 = fx.build(tr.cpu.values, tr.mem.values, tr.net.values, tr.minute_of_day.values,
                          self.scales[name], cfg.roll, tpd)
            self.burst.fit(name, F0)
            F, _ = self.rows(name, d)
            Y = fx.targets(d.cpu.values, d.mem.values, self.scales[name], cfg.horizon)
            X1, Y1, _ = fx.windows(F, Y, cfg.window, 0, tr_end - cfg.horizon)
            X2, Y2, _ = fx.windows(F, Y, cfg.window, tr_end, va_end - cfg.horizon)
            Xs.append(X1); Ys.append(Y1); Xv.append(X2); Yv.append(Y2)
        X, Y = np.concatenate(Xs), np.concatenate(Ys)
        Xv, Yv = np.concatenate(Xv), np.concatenate(Yv)
        if verbose:
            print(f"  training windows: {X.shape}  validation windows: {Xv.shape}")
        self.model = LSTMRegressor(X.shape[2], cfg.hidden, 2, seed=cfg.seed)
        self.history = self.model.fit(X, Y, Xv, Yv, cfg.epochs, cfg.batch, cfg.lr,
                                      cfg.patience, cfg.seed, verbose)
        return self

    def forecast_series(self, name, d, start, stop):
        """Forecast for every t in [start, stop): uses rows <= t, predicts peak of t+1..t+H."""
        F, b = self.rows(name, d)
        w = self.cfg.window
        ts = np.arange(max(start, w - 1), stop)
        X = np.stack([F[t - w + 1: t + 1] for t in ts])
        P = np.concatenate([self.model.predict(X[i:i + 512]) for i in range(0, len(X), 512)])
        cs, ms, _ = self.scales[name]
        return ts, np.clip(P[:, 0], 0, None) * cs, np.clip(P[:, 1], 0, None) * ms, b[ts]

    def forecast_live(self, name, cpu, mem, net, minute_of_day):
        """Online: pass the recent raw history (>= window + roll samples) for one container."""
        F = fx.build(np.asarray(cpu), np.asarray(mem), np.asarray(net),
                     np.asarray(minute_of_day), self.scales[name], self.cfg.roll,
                     self.cfg.ticks_per_day)
        b = self.burst.score(name, F)
        if self.use_burst:
            F = np.column_stack([F, b])
        x = F[-self.cfg.window:][None]
        p = self.model.predict(x)[0]
        cs, ms, _ = self.scales[name]
        return max(0.0, p[0] * cs), max(0.0, p[1] * ms), float(b[-1])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"cfg": self.cfg, "use_burst": self.use_burst, "scales": self.scales,
                         "burst": self.burst, "model": self.model.p,
                         "shape": (self.model.D, self.model.H)}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            s = pickle.load(f)
        p = cls(s["cfg"], s["use_burst"])
        p.scales, p.burst = s["scales"], s["burst"]
        D, H = s["shape"]
        p.model = LSTMRegressor(D, H, 2)
        p.model.p = s["model"]
        return p
