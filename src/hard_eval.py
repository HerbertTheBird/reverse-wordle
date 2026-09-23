"""
Hard-negative evaluation: can the model pick the actually-played word out of the
OTHER still-possible answers (pool), not random vocab? This is what inference needs.
Compares the NCE MLP vs the LightGBM ranker on the same hard groups.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json, random
import numpy as np
import lightgbm as lgb                            # load LightGBM's libomp FIRST
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats, features as FT
from wordlists import ANSWERS, IDX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 12


def main(n=1500, seed=1):
    bst = lgb.Booster(model_file=os.path.join(ROOT, "data", "model_lgb.txt"))
    import torch                                   # torch/OMP after LightGBM
    import infer
    globals()["torch"] = torch
    infer.init()                                  # loads NET, MU, SD, arrays
    games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "games.jsonl"))]
    rng = random.Random(seed)
    rng.shuffle(games)
    decisions = list(FT.collect(games[:4000]))    # (ai, pool, played, gused, hist)
    rng.shuffle(decisions)

    mlp_hit = lgb_hit = tot = 0
    for ai, pool, played, gused, hist in decisions:
        if played < 0 or played not in set(pool.tolist()) or len(pool) < 3:
            continue                              # need played to be a still-possible answer
        others = [k for k in pool.tolist() if k != played]
        negs = rng.sample(others, min(K, len(others)))
        lst = [played] + negs
        rng.shuffle(lst)                          # played NOT always index 0 (avoid tie bias)
        played_pos = lst.index(played)
        cand = np.array(lst)
        state = feats.parse_history(hist)
        F = feats.feature_matrix(cand, pool, ai, FT.eg(len(pool)), gused, state,
                                 FT.PKV, FT.SFV, FT.NSV, FT.ZFV, FT.eg)
        with torch.no_grad():
            mlp = infer.NET(torch.tensor(feats.transform(F, infer.MU, infer.SD))).numpy()
        lg = bst.predict(F)
        mlp_hit += int(mlp.argmax() == played_pos)
        lgb_hit += int(lg.argmax() == played_pos)
        tot += 1
        if tot >= n:
            break
    print(f"hard-negative pick-played ({tot} decisions, played vs {K} pool words):")
    print(f"  MLP      : {mlp_hit/tot:.3f}")
    print(f"  LightGBM : {lgb_hit/tot:.3f}")
    print(f"  chance   : {1/(K+1):.3f}")


if __name__ == "__main__":
    main()
