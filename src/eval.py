"""Evaluate few-player accuracy: top-1 / top-5 / mean-rank at K players."""
import json, os, sys, random, collections, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_days(min_games=10):
    games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "games.jsonl"))]
    byday = collections.defaultdict(list)
    for g in games:
        byday[g["thread_id"]].append(g)
    days = []
    for d, gs in byday.items():
        ans = collections.Counter(x["answer"] for x in gs).most_common(1)[0][0]
        gs = [x for x in gs if x["answer"] == ans]
        if len(gs) >= min_games:
            days.append((d, ans, gs))
    return days


def grid_codes(game):
    return [int(gu["pattern"], 3) for gu in game["guesses"]]


def main(K=5, n_days=40, repeats=2, seed=0):
    infer.init()
    days = load_days()
    rng = random.Random(seed)
    rng.shuffle(days)
    days = days[:n_days]
    top1 = top5 = 0
    ranks = []
    t0 = time.time()
    for di, (d, ans, gs) in enumerate(days):
        for r in range(repeats):
            sample = rng.sample(gs, min(K, len(gs)))
            grids = [grid_codes(g) for g in sample]
            ranked, cand_idx, total = infer.infer(grids, prune=300, top=5)
            words = [w for w, _ in ranked]
            hit1 = words[0] == ans
            hit5 = ans in words
            # rank of true answer among candidates
            order = np.argsort(-total)
            pos = np.where(cand_idx[order] == infer.IDX[ans])[0]
            rank = int(pos[0]) + 1 if len(pos) else 9999
            top1 += hit1; top5 += hit5; ranks.append(rank)
        print(f"[{di+1}/{len(days)}] {ans}: last top5={words}", flush=True)
    n = len(days) * repeats
    print(f"\nK={K} players | days={len(days)} x{repeats}  ({time.time()-t0:.0f}s)")
    print(f"  top-1 accuracy: {top1/n:.1%}")
    print(f"  top-5 accuracy: {top5/n:.1%}")
    print(f"  median rank   : {int(np.median(ranks))}   mean rank: {np.mean(ranks):.1f}")


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    main(K, nd)
