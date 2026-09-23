"""
Scaling study: answer-inference accuracy vs number of grids (players).
Runs K in {1,2,4,8,16} on the same `--days` days (nested: same 16-player sample per day,
take the first K), reporting top-1 and mean P(true answer) per K.

Every solve is appended to data/scaling_results.jsonl so partial results survive an
interruption; re-running skips (day,K) pairs already recorded.

Usage: python src/scaling.py --days 50 [--seed 7 --beam 5 --prune 150]
"""
import argparse, sys, os, json, collections, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer

OUT = None


def fmt(s):
    m, s = divmod(int(s), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--prune", type=int, default=150, help="-1 = no pruning (infinite)")
    ap.add_argument("--grids", default="1,2,4,8,16", help="comma list of K values")
    ap.add_argument("--out", default="scaling_results.jsonl")
    a = ap.parse_args()
    KS = [int(x) for x in a.grids.split(",")]
    prune_to = 10**9 if a.prune < 0 else a.prune
    OUT = os.path.join(infer.ROOT, "data", a.out)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["day"], r["k"]))
    print(f"loading model ... ({len(done)} solves already recorded)", flush=True)
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
    total = len(days) * len(KS)
    print(f"{len(days)} days x {KS} grids = {total} solves\n", flush=True)

    t0 = time.time(); n = 0
    for tid, ans, gs in days:
        sample = rng.sample(gs, max(KS))
        for k in KS:
            n += 1
            if (tid, k) in done:
                continue
            grids = [[gu["pattern"] for gu in g["guesses"]] for g in sample[:k]]
            out = infer.solve(grids, top=1, beam=a.beam, prune_to=prune_to)
            _, cand, tot, _ = out
            p = np.exp(tot - tot.max()); p /= p.sum()
            pos = np.where(cand == infer.IDX[ans])[0]
            pc = float(p[pos[0]]) if len(pos) else 0.0
            hit = int(out[0][0][0] == ans)
            with open(OUT, "a") as f:
                f.write(json.dumps({"day": tid, "ans": ans, "k": k, "p": pc, "hit": hit}) + "\n")
            eta = (time.time() - t0) / n * (total - n)
            print(f"  [{n}/{total}] {ans:6} K={k:2}: P={pc:.3f} {'HIT' if hit else '   '}  eta {fmt(eta)}", flush=True)

    # summary from the results file
    agg = collections.defaultdict(lambda: [0, 0.0, 0])   # k -> [hits, psum, count]
    for l in open(OUT):
        r = json.loads(l)
        if r["k"] in KS:
            agg[r["k"]][0] += r["hit"]; agg[r["k"]][1] += r["p"]; agg[r["k"]][2] += 1
    print(f"\n=== scaling ({fmt(time.time()-t0)}) ===")
    print(f"{'grids':>5}  {'top-1':>12}  {'mean P(answer)':>14}")
    for k in KS:
        h, ps, c = agg[k]
        if c:
            print(f"{k:5}  {h}/{c} = {100*h/c:4.0f}%  {ps/c:14.3f}")


if __name__ == "__main__":
    main()
