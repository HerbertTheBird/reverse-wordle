"""
Probabilistic benchmark: for each day, how much probability does the model put on the
TRUE answer? Runs each day with 10 grids and with 20 grids (40 solves for 20 days).

Reports, per grid-count: sum & mean P(true answer), and how often the top guess is right.
Prints per-day so partial results survive, with a progress bar + ETA.

Usage: python src/bench_prob.py --days 20 [--seed 7] [--beam 6 --prune 250]
"""
import argparse, sys, os, json, collections, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer


def fmt(s):
    m, s = divmod(int(s), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def p_correct(ranked_out, ans):
    """Softmax prob assigned to the true answer (0 if it fell outside the candidate set)."""
    _, cand, total, _ = ranked_out
    p = np.exp(total - total.max()); p /= p.sum()
    pos = np.where(cand == infer.IDX[ans])[0]
    return float(p[pos[0]]) if len(pos) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--prune", type=int, default=250)
    a = ap.parse_args()
    KS = (10, 20)

    print("loading model ...", flush=True)
    infer.init()
    games = [json.loads(l) for l in open(os.path.join(infer.ROOT, "data", "games.jsonl"))]
    by = collections.defaultdict(list)
    for g in games:
        by[g["thread_id"]].append(g)
    days = []
    for gs in by.values():
        ans = collections.Counter(x["answer"] for x in gs).most_common(1)[0][0]
        gs = [x for x in gs if x["answer"] == ans]
        if len(gs) >= max(KS):
            days.append((ans, gs))
    rng = random.Random(a.seed); rng.shuffle(days); days = days[: a.days]
    print(f"{len(days)} days; solving with {KS} grids each ({len(days)*len(KS)} solves)\n", flush=True)

    psum = {k: 0.0 for k in KS}
    hits = {k: 0 for k in KS}
    t0 = time.time(); solves = 0; tot_solves = len(days) * len(KS)
    for i, (ans, gs) in enumerate(days, 1):
        sample = rng.sample(gs, max(KS))
        line = f"[{i}/{len(days)}] {ans:6}"
        for k in KS:
            grids = [[gu["pattern"] for gu in g["guesses"]] for g in sample[:k]]
            out = infer.solve(grids, top=1, beam=a.beam, prune_to=a.prune)
            pc = p_correct(out, ans); hit = out[0][0][0] == ans
            psum[k] += pc; hits[k] += hit
            solves += 1
            line += f"  | K{k}: P={pc:.3f} {'HIT' if hit else '   '}"
        eta = (time.time() - t0) / solves * (tot_solves - solves)
        print(line + f"  (eta {fmt(eta)})", flush=True)

    print(f"\n=== {len(days)} days ({fmt(time.time()-t0)}) ===")
    for k in KS:
        n = len(days)
        print(f"K={k:2} grids:  sum P(answer) = {psum[k]:.2f} / {n}   "
              f"mean P = {psum[k]/n:.3f}   top-1 = {hits[k]}/{n} ({100*hits[k]/n:.0f}%)")


if __name__ == "__main__":
    main()
