"""
Pairwise (RankNet) MLP on hard in-pool negatives: learn a per-word score f(word,state)
so f(played) > f(other consistent word). Loss = -logsigmoid(f(played) - f(neg)).
At inference: rank words by f (softmax over the candidate set for probabilities).

Saves data/model_rank.pt.
"""
import json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats
from nn import Net

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS, LOG1P, LOGF = feats.FEATS, feats.LOG1P, feats.LOGF


def logtransform(Xraw):
    X = Xraw.astype(np.float32).copy()
    for j, f in enumerate(FEATS):
        if f in LOG1P:
            X[:, j] = np.log1p(X[:, j])
        elif f in LOGF:
            X[:, j] = np.log(np.clip(X[:, j], 1e-12, None))
    return X


def main():
    df = pd.read_parquet(os.path.join(ROOT, "data", "train_hard.parquet"))
    g = df["group"].to_numpy()
    starts = np.r_[0, np.where(np.diff(g) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(g)]
    grp = list(zip(starts.tolist(), ends.tolist()))          # (start=played row, end)
    Xraw = df[FEATS].to_numpy(np.float32)
    Xlog = logtransform(Xraw)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(grp))
    val_g = set(perm[: len(grp) // 5].tolist())
    tr_g = [grp[i] for i in range(len(grp)) if i not in val_g]
    va_g = [grp[i] for i in range(len(grp)) if i in val_g]

    mu = Xlog[np.r_[[s for s, _ in tr_g]]].mean(0)           # stats from played rows (stable)
    sd = Xlog.std(0) + 1e-6
    Xn = ((Xlog - mu) / sd).astype(np.float32)

    def pairs(groups):
        a, b = [], []                                        # played row, neg row
        for s, e in groups:
            for r in range(s + 1, e):
                a.append(s); b.append(r)
        return np.array(a), np.array(b)
    pa, pb = pairs(tr_g)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(Xn, device=dev)
    net = Net(len(FEATS)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def val_pickplayed():
        net.eval()
        with torch.no_grad():
            s = net(X).cpu().numpy()
        hit = sum(1 for st, e in va_g if s[st:e].argmax() == 0)
        return hit / len(va_g)

    best, best_acc = None, -1
    pa_t = torch.tensor(pa, device=dev); pb_t = torch.tensor(pb, device=dev)
    n = len(pa); bs = 8192
    for epoch in range(60):
        net.train(); idx = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            opt.zero_grad()
            diff = net(X[pa_t[j]]) - net(X[pb_t[j]])
            loss = -torch.nn.functional.logsigmoid(diff).mean()
            loss.backward(); opt.step()
        acc = val_pickplayed()
        if acc > best_acc:
            best_acc = acc; best = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if epoch % 10 == 0 or epoch == 59:
            print(f"epoch {epoch:2d}  val pick-played {acc:.3f}")
    torch.save({"state": best, "mu": mu, "sd": sd, "feats": FEATS,
                "log1p": list(LOG1P), "logf": list(LOGF)},
               os.path.join(ROOT, "data", "model_rank.pt"))
    print(f"saved model_rank.pt (best val pick-played {best_acc:.3f})")


if __name__ == "__main__":
    main()
