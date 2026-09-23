"""
K-free evaluation: rank the actually-played word against ALL candidates, not a small
sampled group.
  (A) full-pool  : rank among EVERY still-possible answer (the hard, relevant test)
  (B) big-random : rank among the played word + BIG random sample of the vocabulary
Reports top-1 / top-5 / mean-reciprocal-rank for the NN and the LightGBM-hard ranker.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json, random
import numpy as np
import lightgbm as lgb                                    # before torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats, features as FT
from wordlists import ANSWERS

ROOT = FT.ROOT
NALL = len(ANSWERS)


def ranks_of(played_pos, scores):
    s = scores[played_pos]
    return int((scores > s).sum()) + 1


def main(n=800, big=3000, seed=2):
    bst = lgb.Booster(model_file=os.path.join(ROOT, "data", "model_lgb_hard.txt"))
    import torch, infer
    infer.init()
    games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "games.jsonl"))]
    rng = random.Random(seed); rng.shuffle(games)
    decs = [d for d in FT.collect(games[:4000])]
    rng.shuffle(decs)

    stat = {m: {"pool_t1": 0, "pool_t5": 0, "pool_mrr": 0.0, "rand_t1": 0}
            for m in ("NN", "LGB")}
    tot = 0
    for ai, pool, played, gused, hist in decs:
        if played < 0 or played not in set(pool.tolist()) or len(pool) < 5:
            continue
        state = feats.parse_history(hist)
        # (A) full pool
        Fp = feats.feature_matrix(pool, pool, ai, FT.eg(len(pool)), gused, state,
                                  FT.PKV, FT.SFV, FT.NSV, FT.ZFV, FT.eg)
        pp = int(np.where(pool == played)[0][0])
        with torch.no_grad():
            nn_s = infer.NET(torch.tensor(feats.transform(Fp, infer.MU, infer.SD))).numpy()
        lgb_s = bst.predict(Fp)
        for m, s in (("NN", nn_s), ("LGB", lgb_s)):
            r = ranks_of(pp, s)
            stat[m]["pool_t1"] += r == 1
            stat[m]["pool_t5"] += r <= 5
            stat[m]["pool_mrr"] += 1.0 / r
        # (B) played + big random vocab sample
        neg = rng.sample(range(NALL), big)
        cand = np.array([played] + [k for k in neg if k != played])
        Fr = feats.feature_matrix(cand, pool, ai, FT.eg(len(pool)), gused, state,
                                  FT.PKV, FT.SFV, FT.NSV, FT.ZFV, FT.eg)
        with torch.no_grad():
            nn_r = infer.NET(torch.tensor(feats.transform(Fr, infer.MU, infer.SD))).numpy()
        lgb_r = bst.predict(Fr)
        stat["NN"]["rand_t1"] += nn_r.argmax() == 0
        stat["LGB"]["rand_t1"] += lgb_r.argmax() == 0
        tot += 1
        if tot % 100 == 0 or tot >= n:
            print(f"\n[{tot}] pool_top1 / pool_top5 / pool_MRR / rand_top1(of {big+1}):", flush=True)
            for m in ("NN", "LGB"):
                s = stat[m]
                print(f"  {m:4} {s['pool_t1']/tot:.3f}  {s['pool_t5']/tot:.3f}  "
                      f"{s['pool_mrr']/tot:.3f}  {s['rand_t1']/tot:.3f}", flush=True)
        if tot >= n:
            break


if __name__ == "__main__":
    import sys as _s
    main(int(_s.argv[1]) if len(_s.argv)>1 else 400, int(_s.argv[2]) if len(_s.argv)>2 else 1200)
