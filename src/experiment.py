"""
Simulate a human-like player and measure average guesses to solve.

- First guess: sampled from the empirical opener dataset (BlueSky starter frequencies).
- Each later guess: sampled from the NN's P(play W | state) over ALL guess words
  (self-normalised exp(logit), renormalised over the vocabulary).
The pool / used-letters / greens / yellows evolve; stop when the guess == answer.

Reports mean/median guesses and the Wordle success rate (<= 6).
"""
import sys, os, collections, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wordlists import ANSWERS, IDX, scores_over_pool, STARTERS
import infer, feats

CAP = 15


def simulate(answer, rng, op_words, op_p):
    ci = IDX[answer]
    pool = np.arange(len(ANSWERS))
    history = []
    for t in range(CAP):
        if t == 0:
            w = op_words[rng.choice(len(op_words), p=op_p)]         # opener ~ dataset
        else:
            state = feats.parse_history(history)
            logits = infer.play_logits(np.arange(len(ANSWERS)), pool, ci,
                                       infer.eg(len(pool)), t, state)
            pr = np.exp(logits - logits.max()); pr /= pr.sum()      # distribution over all words
            w = ANSWERS[rng.choice(len(ANSWERS), p=pr)]
        p = int(scores_over_pool(w)[ci])
        history.append((w, p))
        pool = pool[scores_over_pool(w)[pool] == p]
        if p == 242:
            return t + 1
    return CAP + 1                                                  # did not solve in CAP


def main(n=400, seed=0):
    infer.init()
    ow = [(w, c) for w, c in STARTERS.items() if w in IDX]
    op_words = [w for w, _ in ow]
    op_p = np.array([c for _, c in ow], float); op_p /= op_p.sum()

    games = [json.loads(l) for l in open(os.path.join(infer.ROOT, "data", "games.jsonl"))]
    by = collections.defaultdict(list)
    for g in games:
        by[g["thread_id"]].append(g)
    answers = sorted({collections.Counter(x["answer"] for x in v).most_common(1)[0][0]
                      for v in by.values()})
    rng = np.random.default_rng(seed)
    answers = list(rng.choice(answers, size=min(n, len(answers)), replace=False))

    guesses = []
    for i, a in enumerate(answers):
        guesses.append(simulate(a, rng, op_words, op_p))
        if (i + 1) % 25 == 0:
            g = np.array(guesses)
            print(f"  {i+1}/{len(answers)}  running mean {g[g<=CAP].mean():.3f}", flush=True)
    g = np.array(guesses)
    solved = g <= 6
    print(f"\n=== {len(g)} simulated games ===")
    print(f"mean guesses (solved, cap {CAP}): {g[g<=CAP].mean():.3f}")
    print(f"median: {int(np.median(g))}   solved within 6: {solved.mean()*100:.1f}%   "
          f"failed(> {CAP}): {(g>CAP).mean()*100:.1f}%")
    dist = collections.Counter(int(x) for x in g)
    print("distribution:", {k: dist[k] for k in sorted(dist)})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
