"""
Offline benchmark for the crowd -> answer inference.

Pools K random players' colour grids per day (no starter word, colours only) and checks
whether the model's top guess is the real answer. Reports top-1 / top-3 / top-5 / median
& mean rank, with a live progress bar + ETA.

Examples:
  python src/benchmark.py --k 15 --days 30
  python src/benchmark.py --k 5 --days 100 --beam 6 --prune 250 --seed 7
  python src/benchmark.py --k 10 --days 20 --verbose
"""
import argparse, sys, os, json, collections, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def bar(done, total, elapsed, extra, w=28):
    frac = done / total
    f = int(w * frac)
    eta = (elapsed / done) * (total - done) if done else 0
    sys.stdout.write(f"\r[{'#'*f}{'.'*(w-f)}] {done}/{total}  "
                     f"elapsed {fmt(elapsed)}  eta {fmt(eta)}  {extra}")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def load_days(min_games):
    games = [json.loads(l) for l in open(os.path.join(infer.ROOT, "data", "games.jsonl"))]
    by = collections.defaultdict(list)
    for g in games:
        by[g["thread_id"]].append(g)
    days = []
    for gs in by.values():
        ans = collections.Counter(x["answer"] for x in gs).most_common(1)[0][0]
        gs = [x for x in gs if x["answer"] == ans]
        if len(gs) >= min_games:
            days.append((ans, gs))
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15, help="players per day")
    ap.add_argument("--days", type=int, default=20, help="number of days to test")
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--prune", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true", help="print each day's result")
    a = ap.parse_args()

    print(f"loading model ...", flush=True)
    infer.init()
    days = load_days(a.k)
    if not days:
        print(f"no days with >= {a.k} games"); return
    rng = random.Random(a.seed)
    rng.shuffle(days)
    days = days[: a.days]
    print(f"benchmarking {len(days)} days, K={a.k} players (beam {a.beam}, prune {a.prune})\n")

    top1 = top3 = top5 = 0
    ranks = []
    t0 = time.time()
    for i, (ans, gs) in enumerate(days, 1):
        sample = rng.sample(gs, a.k)
        grids = [[gu["pattern"] for gu in g["guesses"]] for g in sample]
        ranked, cand, total, n_legal = infer.solve(grids, top=5, beam=a.beam, prune_to=a.prune)
        words = [w for w, _ in ranked]
        order = np.argsort(-total)
        pos = np.where(cand[order] == infer.IDX[ans])[0]
        rank = int(pos[0]) + 1 if len(pos) else 9999
        top1 += words[0] == ans; top3 += ans in words[:3]; top5 += ans in words
        ranks.append(rank)
        if a.verbose:
            sys.stdout.write("\r" + " " * 90 + "\r")
            mark = "HIT" if words[0] == ans else f"{words[0]} (ans rank {rank})"
            print(f"  {ans:6} -> {mark}   top3={words[:3]}")
        bar(i, len(days), time.time() - t0,
            f"top1 {top1}/{i} ({100*top1//i}%)")

    n = len(days)
    r = np.array(ranks)
    print(f"\n=== results: {n} days, K={a.k} players ===")
    print(f"  top-1 (one try): {top1}/{n} = {100*top1/n:.1f}%")
    print(f"  top-3          : {top3}/{n} = {100*top3/n:.1f}%")
    print(f"  top-5          : {top5}/{n} = {100*top5/n:.1f}%")
    print(f"  median rank    : {int(np.median(r))}")
    print(f"  mean rank      : {r[r < 9999].mean():.1f}  (found in candidate set: {(r<9999).sum()}/{n})")
    print(f"  total time     : {fmt(time.time()-t0)}  ({(time.time()-t0)/n:.0f}s/day)")


if __name__ == "__main__":
    main()
