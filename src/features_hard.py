"""
Hard-negative training table: each decision = the played word (row 0) + up to N_NEG
OTHER still-possible answers (sampled from the pool). Since every candidate is a
possible answer, the model must learn WHICH consistent word a human prefers -- the
real inference task -- instead of the trivial 'is it an answer'.

Output: data/train_hard.parquet  (group, played, + 72 features; variable group sizes)
"""
import json, os, random, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats
import features as FT                                   # reuse collect(), eg, PKV/SFV/NSV/ZFV

ROOT = FT.ROOT
random.seed(0)
N_NEG = 15
MAX_DEC = 60000


def main(limit=None):
    games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "games.jsonl"))]
    if limit:
        games = games[:limit]
    total = sum(max(0, len(g["guesses"]) - 1) for g in games if g["answer"] in FT.ANSWER_SET)
    keep = min(1.0, MAX_DEC / max(total, 1))
    print(f"~{total} decisions; keeping ~{keep:.1%}")
    Ffloat, Fint, groups, played = [], [], [], []
    d = 0
    for ai, pool, pk_played, gused, hist in FT.collect(games):
        if pk_played < 0 or random.random() > keep:
            continue
        poollist = pool.tolist()
        if pk_played not in poollist or len(poollist) < 3:
            continue                                     # need the played word to be a live answer
        others = [k for k in poollist if k != pk_played]
        negs = random.sample(others, min(N_NEG, len(others)))
        cand = np.array([pk_played] + negs)              # played at index 0
        state = feats.parse_history(hist)
        F = feats.feature_matrix(cand, pool, ai, FT.eg(len(pool)), gused, state,
                                 FT.PKV, FT.SFV, FT.NSV, FT.ZFV, FT.eg)
        Ffloat.append(F[:, :20].astype(np.float32))
        Fint.append(F[:, 20:].astype(np.int8))
        groups.extend([d] * len(cand))
        played.extend([1] + [0] * len(negs))
        d += 1
        if d % 5000 == 0:
            print(f"  {d} decisions kept", flush=True)
    Ffloat = np.vstack(Ffloat); Fint = np.vstack(Fint)
    df = pd.DataFrame({"group": np.array(groups, np.int32), "played": np.array(played, np.int8)})
    for j, name in enumerate(feats.FEATS[:20]):
        df[name] = Ffloat[:, j]
    for j, name in enumerate(feats.FEATS[20:]):
        df[name] = Fint[:, j]
    df.to_parquet(os.path.join(ROOT, "data", "train_hard.parquet"))
    print(f"wrote {len(df)} rows / {d} decisions (in-pool negatives, <= {N_NEG} each)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
