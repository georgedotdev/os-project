"""Stage 1: memoryless burst detector (idea from ExtremoNet, Wang & Wang 2024).

An Isolation Forest learns what "normal" rows look like for each container. Rows that are easy
to isolate (few random splits needed) are unusual, e.g. the first minutes of a flash crowd.
Its score is NOT used as a vote. It is appended to the row as an extra column and handed to
the LSTM, which learns what an unusual row tends to lead to.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

# columns of the L2 row that describe "shape of change"
BURST_INPUT = [0, 2, 3, 5, 6]   # cpu, net, cpu_delta, cpu_roll_std, mem_delta


class BurstDetector:
    def __init__(self, seed=0, contamination=0.03):
        self.seed = seed
        self.contamination = contamination
        self.models = {}   # one forest per container
        self.norm = {}     # (p50, p99.5, threshold)

    def fit(self, name, F_train):
        X = F_train[:, BURST_INPUT]
        m = IsolationForest(n_estimators=150, contamination=self.contamination,
                            random_state=self.seed).fit(X)
        raw = -m.score_samples(X)
        lo, hi = np.percentile(raw, 50), np.percentile(raw, 99.5)
        self.models[name] = m
        s = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
        self.norm[name] = (lo, hi, float(np.percentile(s, 97)))
        return self

    def score(self, name, F):
        lo, hi, _ = self.norm[name]
        raw = -self.models[name].score_samples(F[:, BURST_INPUT])
        return np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)

    def threshold(self, name):
        return self.norm[name][2]
