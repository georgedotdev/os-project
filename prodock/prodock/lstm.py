"""Stage 2: a small LSTM written in plain NumPy.

No PyTorch/TensorFlow needed, so it runs on any laptop and every line of the maths is visible.
Architecture: input (T x D) -> one LSTM layer (H units) -> last hidden state -> linear -> 2 outputs
(peak CPU, peak memory over the next few ticks). Trained with MSE + Adam, backprop through time.
"""
import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTMRegressor:
    def __init__(self, n_in, n_hidden=32, n_out=2, seed=0):
        rng = np.random.default_rng(seed)
        H, D = n_hidden, n_in
        lim = np.sqrt(6 / (D + H + 4 * H))
        self.p = {
            "W": rng.uniform(-lim, lim, (4 * H, D + H)),
            "b": np.zeros(4 * H),
            "Wy": rng.uniform(-np.sqrt(6 / (H + n_out)), np.sqrt(6 / (H + n_out)), (n_out, H)),
            "by": np.zeros(n_out),
        }
        self.p["b"][H:2 * H] = 1.0     # forget-gate bias = 1 (standard trick)
        self.H, self.D = H, D
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---------- forward ----------
    def forward(self, X, keep=False):
        B, T, _ = X.shape
        H = self.H
        W, b = self.p["W"], self.p["b"]
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = []
        for t in range(T):
            xh = np.concatenate([X[:, t, :], h], axis=1)
            z = xh @ W.T + b
            i, f, o = _sig(z[:, :H]), _sig(z[:, H:2 * H]), _sig(z[:, 2 * H:3 * H])
            g = np.tanh(z[:, 3 * H:])
            c_prev = c
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            if keep:
                cache.append((xh, i, f, o, g, c_prev, tc))
        y = h @ self.p["Wy"].T + self.p["by"]
        if keep:
            self._cache = (cache, h)
        return y

    predict = forward

    # ---------- backward ----------
    def backward(self, dy):
        cache, h_last = self._cache
        H, D = self.H, self.D
        W = self.p["W"]
        g_ = {k: np.zeros_like(v) for k, v in self.p.items()}
        g_["Wy"] = dy.T @ h_last
        g_["by"] = dy.sum(0)
        dh = dy @ self.p["Wy"]
        dc = np.zeros_like(dh)
        for xh, i, f, o, g, c_prev, tc in reversed(cache):
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di, dg, df = dc * g, dc * i, dc * c_prev
            dc = dc * f
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 do * o * (1 - o), dg * (1 - g ** 2)], axis=1)
            g_["W"] += dz.T @ xh
            g_["b"] += dz.sum(0)
            dh = (dz @ W)[:, D:]
        return g_

    def loss_and_grads(self, X, Y):
        pred = self.forward(X, keep=True)
        diff = pred - Y
        loss = float(np.mean(diff ** 2))
        grads = self.backward(2 * diff / diff.size)
        return loss, grads

    # ---------- Adam ----------
    def step(self, grads, lr, clip=5.0, b1=0.9, b2=0.999, eps=1e-8):
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in self.p:
            g = grads[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, Y, Xv, Yv, epochs=20, batch=128, lr=3e-3, patience=5, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        hist = {"train": [], "val": []}
        best, best_p, bad = np.inf, None, 0
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            tl = []
            for s in range(0, len(X), batch):
                j = idx[s:s + batch]
                loss, grads = self.loss_and_grads(X[j], Y[j])
                self.step(grads, lr)
                tl.append(loss)
            vl = float(np.mean((self.predict(Xv) - Yv) ** 2))
            hist["train"].append(float(np.mean(tl)))
            hist["val"].append(vl)
            if verbose:
                print(f"  epoch {ep + 1:2d}  train MSE {hist['train'][-1]:.5f}  val MSE {vl:.5f}")
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_p = {k: v.copy() for k, v in self.p.items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print("  early stop")
                    break
        if best_p is not None:
            self.p = best_p
        return hist

    def save(self, path):
        np.savez(path, H=self.H, D=self.D, **self.p)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(int(z["D"]), int(z["H"]), z["by"].shape[0])
        for k in m.p:
            m.p[k] = z[k]
        return m
