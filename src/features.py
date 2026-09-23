"""
Build the NN training table (grouped, for full-vocab NCE), using the shared feats module.

Each non-opener decision -> one group: played word (row 0) + N_NEG vocab negatives.
Features (66) come from feats.feature_matrix so training == inference exactly.
"""
import json, math, os, random, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wordlists import ANSWERS, ANSWER_SET, IDX, scores_over_pool, STARTERS, STARTER_FLOOR
import feats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(0)
NALL = len(ANSWERS)
N_NEG = 255
MAX_DEC = 40000                       # ~10.2M rows

_c = json.load(open(os.path.join(ROOT, "data", "eguess_curve.json")))
_CX, _CY = np.array(_c["x"]), np.array(_c["y"])
_nsj = json.load(open(os.path.join(ROOT, "data", "nonstart-freq.json")))


def eg(size):
    s = np.asarray(size, float)
    out = np.where(s <= 1, 1.0, np.interp(np.log2(np.clip(s, 1, None)), _CX, _CY))
    return out if out.ndim else float(out)


def load_prevalence():
    df = pd.read_excel(os.path.join(ROOT, "data", "English_Word_Prevalences.xlsx"))
    df["Word"] = df["Word"].astype(str).str.lower()
    df = df[df["Word"].str.fullmatch(r"[a-z]{5}")]
    return ({r.Word: r.Pknown for r in df.itertuples()},
            {r.Word: r.FreqZipfUS for r in df.itertuples()})


PK, ZF = load_prevalence()
PKV = np.array([PK.get(w, 0.02) for w in ANSWERS])
ZFV = np.array([ZF.get(w, 1.0) for w in ANSWERS])              # Zipf frequency
SFV = np.array([STARTERS.get(w, STARTER_FLOOR) for w in ANSWERS])
NSV = np.array([_nsj["freq"].get(w, _nsj["floor"]) for w in ANSWERS])


def collect(games):
    for g in games:
        if g["answer"] not in ANSWER_SET:
            continue
        ai = IDX[g["answer"]]; pool = np.arange(NALL); hist = []
        for i, gu in enumerate(g["guesses"]):
            patt = int(gu["pattern"], 3)
            if i >= 1:
                yield ai, pool, IDX.get(gu["word"], -1), i, list(hist)
            hist.append((gu["word"], patt))
            pool = pool[scores_over_pool(gu["word"])[pool] == patt]


def main(limit=None):
    games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "games.jsonl"))]
    if limit:
        games = games[:limit]
    total = sum(max(0, len(g["guesses"]) - 1) for g in games if g["answer"] in ANSWER_SET)
    keep = min(1.0, MAX_DEC / max(total, 1))
    print(f"~{total} decisions; keeping ~{keep:.1%}")
    Ffloat, Fint, groups, played = [], [], [], []
    d = 0
    for ai, pool, pk_played, gused, hist in collect(games):
        if pk_played < 0 or random.random() > keep:
            continue
        negs = set()
        while len(negs) < N_NEG:
            c = random.randrange(NALL)
            if c != pk_played:
                negs.add(c)
        cand = np.array([pk_played] + list(negs))
        state = feats.parse_history(hist)
        F = feats.feature_matrix(cand, pool, ai, eg(len(pool)), gused, state, PKV, SFV, NSV, ZFV, eg)
        Ffloat.append(F[:, :14].astype(np.float32))
        Fint.append(F[:, 14:].astype(np.int8))
        groups.extend([d] * len(cand))
        played.extend([1] + [0] * len(negs))
        d += 1
        if d % 5000 == 0:
            print(f"  {d} decisions kept")
    Ffloat = np.vstack(Ffloat); Fint = np.vstack(Fint)
    df = pd.DataFrame({"group": np.array(groups, np.int32), "played": np.array(played, np.int8)})
    for j, name in enumerate(feats.FEATS[:14]):
        df[name] = Ffloat[:, j]
    for j, name in enumerate(feats.FEATS[14:]):
        df[name] = Fint[:, j]
    df.to_parquet(os.path.join(ROOT, "data", "train.parquet"))
    json.dump(feats.FEATS, open(os.path.join(ROOT, "data", "feature_cols.json"), "w"))
    print(f"wrote {len(df)} rows / {d} decisions, {N_NEG} negatives, {len(feats.FEATS)} features")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
