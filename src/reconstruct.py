"""
Reconstruct the most likely GAME someone played, given the answer, the colour grid,
and optionally the starter word. Uses a BEAM SEARCH over whole word-paths (wider beam
= explores more games = better, slower), not a greedy per-row pick.

For each row the played word must be consistent with that row's colours against the
answer. Row 1 is weighted by empirical opener frequency (or the given starter);
rows 2+ by P(play W | state) from the model, with the pool / used-letters / greens /
yellows evolving along each path.

Run:  .venv/bin/python src/reconstruct.py
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wordlists import ANSWERS, ANSWER_SET, IDX, scores_over_pool, PM
import infer
import feats

EMOJI = {"0": "⬛", "1": "\U0001f7e8", "2": "\U0001f7e9"}
CONS_CAP = 400            # max consistent words scored per row (top by play frequency)


def _bar(done, total, w=40):
    f = int(w * done / max(total, 1))
    sys.stdout.write(f"\r  [{'#' * f}{'.' * (w - f)}] {done}/{total}")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


def reconstruct(answer, patterns, starter=None, beam=200, progress=None):
    ci = IDX[answer]
    # a beam entry = [logprob, pool(indices), history(tuple), words(list)]
    beams = [[0.0, np.arange(len(ANSWERS)), (), []]]
    total_steps = len(patterns) * beam
    done = 0
    for i, p in enumerate(patterns):
        cons = np.where(PM[:, ci] == p)[0]                # words consistent with this row
        if len(cons) == 0:
            return None, beams                            # impossible row for this answer
        if len(cons) > CONS_CAP:
            cons = cons[np.argsort(-(infer.NSV[cons] + infer.SFV[cons]))[:CONS_CAP]]
        new = []
        for lp, pool, hist, words in beams:
            if i == 0 and starter:
                cand, probs = [IDX[starter]], [1.0] if PM[IDX[starter], ci] == p else []
                if not probs:
                    cand = []
            elif i == 0:
                wts = infer.SFV[cons]; probs = wts / wts.sum(); cand = cons
            else:
                state = feats.parse_history(list(hist))
                cand = cons[np.isin(cons, pool)]         # consistent AND still-possible
                if len(cand) == 0:
                    cand = cons                           # player probed off-pool: fall back
                lg = infer.rank_score(cand, pool, ci, infer.eg(len(pool)), i, state)
                e = np.exp(lg - lg.max()); probs = e / e.sum()
            order = np.argsort(-np.asarray(probs))[:beam]
            for j in order:
                k = int(cand[j]); pr = float(probs[j])
                if pr <= 0:
                    continue
                sub = pool[scores_over_pool(ANSWERS[k])[pool] == p]
                new.append([lp + math.log(pr), sub, hist + ((ANSWERS[k], int(p)),),
                            words + [ANSWERS[k]]])
            done += 1
            if progress:
                progress(min(done, total_steps), total_steps)
        new.sort(key=lambda b: -b[0])
        beams = new[:beam]
        if not beams:
            return None, []
    if progress:
        progress(total_steps, total_steps)
    return beams[0], beams


def main():
    sys.stdout.write("loading model ... "); sys.stdout.flush()
    infer.init(); print("ready.\n")
    ans = ""
    while ans not in ANSWER_SET:
        ans = input("Answer (5-letter word): ").strip().lower()
    starter = input("Starter word (optional, blank to skip): ").strip().lower() or None
    if starter and starter not in ANSWER_SET:
        print("  starter not in word list; ignoring"); starter = None
    bw = input("Beam width (blank = 200; larger = better & slower): ").strip()
    beam = int(bw) if bw.isdigit() else 200
    print("Colour rows (0=grey 1=yellow 2=green), one per line, blank to finish:")
    patterns = []
    while True:
        line = input(f"  row {len(patterns)+1}: ").strip()
        if not line:
            break
        if len(line) == 5 and set(line) <= set("012"):
            patterns.append(int(line, 3))
        else:
            print("  need 5 chars of 0/1/2")

    print(f"\nsearching (beam {beam}) ...")
    best, beams = reconstruct(ans, patterns, starter, beam, progress=_bar)
    if best is None:
        print("no consistent game found for that answer + colours"); return
    print("\nMost likely game (single best path):")
    for w, p in zip(best[3], patterns):
        colors = "".join(EMOJI[c] for c in np.base_repr(p, 3).zfill(5))
        print(f"  {colors}  {w}")

    lp = np.array([b[0] for b in beams])
    pr = np.exp(lp - lp.max()); pr /= pr.sum()          # normalise over the beam

    # Most likely word at EACH spot independently: marginalise over the beam
    # (sum path probabilities grouped by the word played at that row).
    print("\nMost likely word in each spot (marginal over beam):")
    for i, p in enumerate(patterns):
        colors = "".join(EMOJI[c] for c in np.base_repr(p, 3).zfill(5))
        marg = {}
        for b, q in zip(beams, pr):
            w = b[3][i]
            marg[w] = marg.get(w, 0.0) + q
        top = sorted(marg.items(), key=lambda kv: -kv[1])[:3]
        best_w, best_q = top[0]
        alt = "   ".join(f"{w} {q*100:.0f}%" for w, q in top[1:])
        print(f"  {colors}  {best_w} ({best_q*100:.1f}%)" + (f"    alt: {alt}" if alt else ""))
    n = min(len(beams), 15)
    print(f"\nTop {n} candidate games (of {len(beams)} in beam):")
    for b, q in zip(beams[:n], pr[:n]):
        print(f"  {q*100:5.1f}%   {' -> '.join(b[3])}")


if __name__ == "__main__":
    main()
