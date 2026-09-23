"""
LightGBM LambdaMART ranker on a grouped parquet (played row is first in each group).
Handles variable group sizes. Reports held-out pick-played (played is the top-scored
candidate in its group).

Usage:  python src/train_lgb.py [train_hard.parquet]  -> data/model_lgb_hard.txt
        python src/train_lgb.py                        -> data/model_lgb.txt (random-neg)
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(fname="train.parquet"):
    df = pd.read_parquet(os.path.join(ROOT, "data", fname))
    F = len(feats.FEATS)
    g = df["group"].to_numpy()
    starts = np.r_[0, np.where(np.diff(g) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(g)]
    grp = list(zip(starts.tolist(), ends.tolist()))          # (start=played row, end)
    X = df[feats.FEATS].to_numpy(np.float32)
    y = df["played"].to_numpy(np.int8)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(grp))
    valset = set(perm[: len(grp) // 5].tolist())
    tr = [grp[i] for i in range(len(grp)) if i not in valset]
    va = [grp[i] for i in range(len(grp)) if i in valset]

    def build(groups):
        rows = np.concatenate([np.arange(s, e) for s, e in groups])
        sizes = np.array([e - s for s, e in groups])
        return X[rows], y[rows], sizes
    Xtr, ytr, gtr = build(tr)
    Xva, yva, gva = build(va)

    dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=feats.FEATS)
    dva = lgb.Dataset(Xva, label=yva, group=gva, reference=dtr)
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  label_gain=[0, 1], verbosity=-1)
    bst = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva],
                    callbacks=[lgb.log_evaluation(100), lgb.early_stopping(80)])

    sv = bst.predict(Xva)
    off = np.r_[0, np.cumsum(gva)]
    hits = sum(1 for i in range(len(gva)) if sv[off[i]:off[i + 1]].argmax() == 0)  # played=idx0
    acc = hits / len(gva)
    out = "model_lgb_hard.txt" if "hard" in fname else "model_lgb.txt"
    bst.save_model(os.path.join(ROOT, "data", out))
    print(f"\n{fname}: LightGBM pick-played (val) = {acc:.3f}  over {len(gva)} groups  -> {out}")
    for name, gain in sorted(zip(feats.FEATS, bst.feature_importance("gain")),
                             key=lambda x: -x[1])[:15]:
        print(f"  {name:16} {gain:12.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train.parquet")
