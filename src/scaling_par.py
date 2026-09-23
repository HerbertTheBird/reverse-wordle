"""
Parallel scaling study (multiprocess). Distributes independent (day, K) solves across
worker processes -- one model per worker, single-threaded to avoid BLAS oversubscription.
Resumable: appends each solve to the results file and skips already-recorded (day, K).

Usage:
  python src/scaling_par.py --grids 5,10,20 --days 50 --beam 25 --prune -1 --workers 4
"""
import argparse, sys, os, json, collections, random, time, math
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INF = None


def init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import infer
    infer.init()
    import torch
    torch.set_num_threads(1)
    infer.engine.save = lambda: None            # no shared-cache writes across workers
    global _INF
    _INF = infer


def solve_task(task):
    import numpy as np
    tid, ans, grids, k, beam, prune = task
    out = _INF.solve(grids, top=1, beam=beam, prune_to=prune)
    _, cand, total, n_legal = out
    p = np.exp(total - total.max()); p /= p.sum()
    pos = np.where(cand == _INF.IDX[ans])[0]
    pc = float(p[pos[0]]) if len(pos) else 0.0
    hit = int(out[0][0][0] == ans)
    return {"day": tid, "ans": ans, "k": k, "p": pc, "hit": hit, "legal": int(n_legal)}


def fmt(s):
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return (f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--beam", type=int, default=25)
    ap.add_argument("--prune", type=int, default=-1, help="-1 = infinite")
    ap.add_argument("--grids", default="5,10,20")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="scaling_par.jsonl")
    a = ap.parse_args()
    KS = [int(x) for x in a.grids.split(",")]
    prune = 10**9 if a.prune < 0 else a.prune
    out = os.path.join(ROOT, "data", a.out)

    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["day"], r["k"]))

    games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "games.jsonl"))]
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

    tasks = []
    for tid, ans, gs in days:
        sample = rng.sample(gs, max(KS))
        grids_full = [[gu["pattern"] for gu in g["guesses"]] for g in sample]
        for k in KS:
            if (tid, k) not in done:
                tasks.append((tid, ans, grids_full[:k], k, a.beam, prune))
    print(f"{len(days)} days x {KS} = {len(days)*len(KS)} solves; {len(tasks)} to run "
          f"({len(done)} done); {a.workers} workers, beam {a.beam}, prune {'inf' if a.prune<0 else a.prune}",
          flush=True)

    t0 = time.time(); n = 0
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(solve_task, tasks):
            f.write(json.dumps(r) + "\n"); f.flush()
            n += 1
            eta = (time.time() - t0) / n * (len(tasks) - n)
            print(f"  [{n}/{len(tasks)}] {r['ans']:6} K={r['k']:2} P={r['p']:.3f} "
                  f"{'HIT' if r['hit'] else '   '}  eta {fmt(eta)}", flush=True)

    agg = collections.defaultdict(lambda: [0, 0.0, 0])
    for l in open(out):
        r = json.loads(l)
        if r["k"] in KS:
            agg[r["k"]][0] += r["hit"]; agg[r["k"]][1] += r["p"]; agg[r["k"]][2] += 1
    print(f"\n=== scaling ({fmt(time.time()-t0)}) ===")
    print(f"{'grids':>5} {'top-1':>14} {'meanP':>8}")
    for k in KS:
        h, ps, c = agg[k]
        if c:
            print(f"{k:5} {h}/{c}={100*h/c:4.0f}%   {ps/c:.3f}")


if __name__ == "__main__":
    main()
