"""
Train P(guess | state) as a full-vocab sampled-softmax (conditional logit).

Each decision is a group: the played word (index 0) + sampled negatives. The net maps
each candidate's 9 features -> a scalar logit; softmax over the group; cross-entropy
against the played word. This yields a score whose softmax over the whole guess
vocabulary approximates P(a human plays this word | state).

Saves data/model.pt (weights + normalisation).
"""
import json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats
FEATS, LOG1P, LOGF = feats.FEATS, feats.LOG1P, feats.LOGF


def transform(df):
    X = np.zeros((len(df), len(FEATS)), np.float32)
    for j, f in enumerate(FEATS):
        v = df[f].to_numpy(np.float64)
        if f in LOG1P:
            v = np.log1p(v)
        elif f in LOGF:
            v = np.log(np.clip(v, 1e-12, None))
        X[:, j] = v
    return X


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):                       # x: (..., d) -> (...)
        return self.net(x).squeeze(-1)


def main():
    df = pd.read_parquet(os.path.join(ROOT, "data", "train.parquet"))
    gsize = df.groupby("group").size()
    assert gsize.nunique() == 1, "groups must be equal-sized"
    K = int(gsize.iloc[0])                      # 1 positive + N_NEG
    assert (df.groupby("group")["played"].first() == 1).all(), "played must be row 0"

    X = transform(df).reshape(-1, K, len(FEATS))   # (G, K, F), played at index 0
    G = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(G)
    val = perm[: G // 5]; tr = perm[G // 5:]

    flat = X[tr].reshape(-1, len(FEATS))
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xn = (X - mu) / sd

    # NCE / self-normalising objective: train exp(logit) to be ~normalised over the
    # vocabulary, so at inference P(play W|state) = exp(logit) directly (no partition Z).
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from wordlists import ANSWERS
    V = len(ANSWERS)                                  # vocabulary size (uniform noise q=1/V)
    n_noise = K - 1
    log_kq = math.log(n_noise / V)                    # shift for uniform noise

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net(len(FEATS)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr = torch.tensor(Xn[tr], device=dev); Xva = torch.tensor(Xn[val], device=dev)

    def nce_loss(x):                                  # x: (B, K, F); index 0 = played
        s = net(x) - log_kq                           # (B, K) NCE-shifted scores
        pos = torch.nn.functional.logsigmoid(s[:, 0])
        neg = torch.nn.functional.logsigmoid(-s[:, 1:]).sum(1)
        return -(pos + neg).mean()

    best, best_acc = None, -1
    for epoch in range(80):
        net.train(); idx = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), 2048):
            b = idx[i:i + 2048]
            opt.zero_grad()
            loss = nce_loss(Xtr[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            logits = net(Xva)
            acc = (logits.argmax(1) == 0).float().mean().item()      # played on top of its group
            selfZ = torch.exp(net(Xva)).mean().item()                # ~ (played+noise) mass proxy
        if acc > best_acc:
            best_acc = acc; best = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if epoch % 10 == 0 or epoch == 79:
            print(f"epoch {epoch:2d}  val pick-played {acc:.3f}  (chance {1/K:.3f})  mean exp(logit) {selfZ:.4f}")

    torch.save({"state": best, "mu": mu, "sd": sd, "feats": FEATS,
                "log1p": list(LOG1P), "logf": list(LOGF)},
               os.path.join(ROOT, "data", "model.pt"))
    print(f"saved model (best val pick-played {best_acc:.3f})")


if __name__ == "__main__":
    main()
