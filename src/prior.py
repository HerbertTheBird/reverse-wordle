"""
Trained answer prior: P(a word is the day's answer) as a function of how many people
know it (pknown). Fit as logistic regression -- positives = observed daily answers,
negatives = the rest of the guess vocabulary. Saves data/prior.json.

log-prior used at inference = log P(is-answer | pknown(A))  (offset cancels in softmax).
"""
import json, os, sys, collections, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wordlists import ANSWERS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PJSON = os.path.join(ROOT, "data", "prior.json")


def _pk():
    import pandas as pd
    df = pd.read_excel(os.path.join(ROOT, "data", "English_Word_Prevalences.xlsx"))
    df["Word"] = df["Word"].astype(str).str.lower()
    return {r.Word: r.Pknown for r in df.itertuples() if isinstance(r.Word, str)}


def feats(pk):
    """Basis functions of pknown (a small flexible function of it)."""
    x = np.clip(np.asarray(pk, float), 1e-4, 1.0)
    return np.stack([x, x * x, np.log(x)], axis=-1)


def fit():
    PK = _pk()
    games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "games.jsonl"))]
    by = collections.defaultdict(list)
    for g in games:
        by[g["thread_id"]].append(g)
    answers = {collections.Counter(x["answer"] for x in v).most_common(1)[0][0] for v in by.values()}
    ans_set = set(answers)

    words = ANSWERS
    y = np.array([w in ans_set for w in words], np.float32)
    X = feats([PK.get(w, 0.02) for w in words]).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd

    Xt = torch.tensor(Xn); yt = torch.tensor(y)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1))   # balance the rare positives
    for _ in range(2000):
        opt.zero_grad()
        logit = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt, pos_weight=pos_w)
        loss.backward(); opt.step()

    obj = {"w": w.detach().tolist(), "b": float(b.detach()), "mu": mu.tolist(), "sd": sd.tolist()}
    json.dump(obj, open(PJSON, "w"))
    print(f"fit prior on {int(y.sum())} answers / {len(words)} words")
    for pk in [0.05, 0.2, 0.5, 0.8, 0.95, 0.99]:
        print(f"  pknown={pk:.2f} -> log-prior {log_prior(np.array([pk]), obj)[0]:+.2f}  "
              f"P(is-answer)={math.exp(min(log_prior(np.array([pk]), obj)[0],0)):.4f}")
    return obj


_CACHE = None


def load():
    global _CACHE
    if _CACHE is None:
        _CACHE = json.load(open(PJSON))
    return _CACHE


def log_prior(pknown, obj=None):
    obj = obj or load()
    X = (feats(pknown) - np.array(obj["mu"])) / np.array(obj["sd"])
    logit = X @ np.array(obj["w"]) + obj["b"]
    return -np.log1p(np.exp(-logit))            # log sigmoid = log P(is-answer | pknown)


if __name__ == "__main__":
    fit()
