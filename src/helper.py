"""
Single-player "what would a human play" helper.

You pick the answer up front. It shows the most common opening words, then after each
guess it computes the colours for you, narrows the possible answers, and lists the
10 words a human is most likely to play next -- P(play W | state) from the model.

Run:  .venv/bin/python src/helper.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer                                    # imports LightGBM before torch (libomp order)
from wordlists import ANSWERS, ANSWER_SET, IDX, scores_over_pool, STARTERS

EMOJI = {"0": "⬛", "1": "\U0001f7e8", "2": "\U0001f7e9"}


def show_starters(n=15):
    top = sorted(STARTERS.items(), key=lambda kv: -kv[1])[:n]
    print("\nMost common opening words:")
    for w, f in top:
        print(f"  {w}   {f*100:5.1f}%")


def most_likely_next(pool, ci, gused, history, n=10):
    """Top-n words a human is most likely to play next -- ranked among the still-possible
    answers (the ranker was trained to discriminate among consistent words)."""
    import feats
    state = feats.parse_history(history)
    s = infer.rank_score(pool, pool, ci, infer.eg(len(pool)), gused, state)
    p = np.exp(s - s.max()); p /= p.sum()
    order = np.argsort(-p)[:n]
    return [(ANSWERS[pool[j]], p[j]) for j in order]


def main():
    infer.init()
    answer = ""
    while answer not in ANSWER_SET:
        answer = input("Pick the answer (5-letter word): ").strip().lower()
        if answer not in ANSWER_SET:
            print("  not in the word list, try again")
    ci = IDX[answer]
    show_starters()

    pool = np.arange(len(ANSWERS))
    made = 0
    history = []
    print("\nType a guess each turn (or 'q' to quit).")
    while True:
        g = input(f"\nGuess #{made+1}: ").strip().lower()
        if g == "q":
            return
        if g not in ANSWER_SET:
            print("  not a valid word"); continue
        patt = int(scores_over_pool(g)[ci])                # colours vs the chosen answer
        pstr = np.base_repr(patt, 3).zfill(5)
        print("  " + "".join(EMOJI[c] for c in pstr) + f"   ({pstr})")
        pool = pool[scores_over_pool(g)[pool] == patt]
        history.append((g, patt))
        made += 1
        if patt == 242:
            print(f"\nSolved '{answer}' in {made}!")
            return
        print(f"  {len(pool)} possible answers left")
        print("  most likely next guess (human):")
        poolset = set(pool.tolist())
        for w, pr in most_likely_next(pool, ci, made, history):
            mark = "  <- possible answer" if IDX[w] in poolset else ""
            print(f"    {w}   {pr*100:6.2f}%{mark}")


if __name__ == "__main__":
    main()
