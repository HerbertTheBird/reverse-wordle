"""Measure the cheap_score rank of the TRUE answer within the achievable set,
per K. The minimum safe prune_to = the worst (largest) such rank observed:
any prune_to >= that keeps the real answer for every day tested.

Usage: python src/measure_prune.py --days 50 --grids 5,10,20 --seed 7
"""
import argparse, sys, os, json, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=50)
    ap.add_argument("--grids", default="5,10,20")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    KS = [int(x) for x in a.grids.split(",")]

    infer.init()
    games = [json.loads(l) for l in open(os.path.join(infer.ROOT, "data", "games.jsonl"))]
    by = collections.defaultdict(list)
    for g in games:
        by[g["thread_id"]].append(g)
    days = []
    for tid, gs in by.items():
        ans = collections.Counter(x["answer"] for x in gs).most_common(1)[0][0]
        gs = [x for x in gs if x["answer"] == ans]
        if len(gs) >= max(KS):
            days.append((tid, ans, gs))
    rng = random.Random(a.seed); rng.shuffle(days); days = days[: a.days]

    stat = {k: {"rank": [], "legal": [], "missing": 0} for k in KS}
    for tid, ans, gs in days:
        sample = rng.sample(gs, max(KS))
        ai = infer.IDX[ans]
        for k in KS:
            grids = [[int(r, 3) for r in [gu["pattern"] for gu in g["guesses"]]] for g in sample[:k]]
            cand = np.arange(infer.N)
            cand = cand[infer.achievable_mask(grids)[cand]]
            stat[k]["legal"].append(len(cand))
            pos = np.where(cand == ai)[0]
            if len(pos) == 0:
                stat[k]["missing"] += 1                       # answer not even achievable (should be 0)
                continue
            cs = infer.cheap_score(grids, cand)
            rank = int((cs > cs[pos[0]]).sum()) + 1           # 1-indexed rank by cheap_score
            stat[k]["rank"].append(rank)

    print(f"\n=== cheap_score rank of TRUE answer ({len(days)} days) ===")
    print(f"{'K':>3} {'legal(med/max)':>16} {'rank max':>9} {'p99':>6} {'p95':>6} {'median':>7} {'missing':>8}")
    for k in KS:
        r = np.array(stat[k]["rank"]); L = np.array(stat[k]["legal"])
        if len(r) == 0:
            print(f"{k:3}  (no data)"); continue
        print(f"{k:3} {int(np.median(L)):7}/{int(L.max()):<8} {r.max():9} "
              f"{int(np.percentile(r,99)):6} {int(np.percentile(r,95)):6} "
              f"{int(np.median(r)):7} {stat[k]['missing']:8}")
    allr = np.array([x for k in KS for x in stat[k]["rank"]])
    print(f"\nMIN SAFE prune_to (all K, all days) = {allr.max()}  "
          f"(p99 across all = {int(np.percentile(allr,99))})")


if __name__ == "__main__":
    main()
