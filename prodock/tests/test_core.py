"""Run with:  python -m pytest -q   (or just: python tests/test_core.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from prodock.lstm import LSTMRegressor
from prodock.config import Config
from prodock import workload, features
from prodock.allocator import ProDockPolicy, StaticPolicy


def test_lstm_gradients_match_numerical():
    rng = np.random.default_rng(1)
    m = LSTMRegressor(4, 5, 2, seed=1)
    X, Y = rng.normal(size=(3, 6, 4)), rng.normal(size=(3, 2))
    _, g = m.loss_and_grads(X, Y)
    for k in m.p:
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in m.p[k].shape)
            old = m.p[k][idx]
            m.p[k][idx] = old + 1e-5; lp = np.mean((m.predict(X) - Y) ** 2)
            m.p[k][idx] = old - 1e-5; lm = np.mean((m.predict(X) - Y) ** 2)
            m.p[k][idx] = old
            num = (lp - lm) / 2e-5
            assert abs(num - g[k][idx]) < 1e-6 + 1e-4 * abs(num), (k, num, g[k][idx])


def test_features_never_look_ahead():
    cfg = Config(days=1)
    df = workload.generate(cfg)
    d = df[df.container == "web-frontend"]
    sc = (1.0, 1.0, 1.0)
    full = features.build(d.cpu.values, d.mem.values, d.net.values, d.minute_of_day.values, sc)
    part = features.build(d.cpu.values[:500], d.mem.values[:500], d.net.values[:500],
                          d.minute_of_day.values[:500], sc)
    assert np.allclose(full[:500], part)


def test_allocator_limits_are_sane():
    cfg = Config()
    p = ProDockPolicy(cfg, cpu_cap=4.0, mem_cap=4.0, start=(1.0, 1.0))
    c, m = p.decide(obs_cpu=0.5, obs_mem=0.5, fc_cpu=10.0, fc_mem=10.0, burst=1.0, thr=0.5)
    assert c <= 4.0 and m <= 4.0
    c, m = p.decide(obs_cpu=0.0, obs_mem=0.0, fc_cpu=0.0, fc_mem=0.0, burst=0.0, thr=0.5)
    assert c >= cfg.min_cpu and m >= cfg.min_mem_gb
    s = StaticPolicy(2.0, 3.0)
    assert s.decide(0, 0, 0, 0, 0, 0) == (2.0, 3.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
