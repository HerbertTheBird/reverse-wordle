"""
Guess the day's answer from many players' games.

For each player you enter their colour patterns, and optionally the starter word
(if you know it -- it's a strong extra clue). It pools everyone, prints the top-20
most likely answers with their chances, and how many answers are still legal.

Run:  .venv/bin/python src/guess.py

Per-player line formats (blank line to finish):
  crane 00201 00100 22200 22222     <- starter known + colours
  00201 00100 22200 22222           <- colours only
(0 = grey, 1 = yellow, 2 = green)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wordlists import ANSWER_SET
import infer


def parse_line(line):
    starter, pats = None, []
    for t in line.split():
        t = t.lower()
        if len(t) == 5 and t.isalpha() and starter is None and not pats:
            starter = t
        elif len(t) == 5 and set(t) <= set("012"):
            pats.append(t)
    return starter, pats


def bar(done, total, w=40):
    f = int(w * done / total)
    sys.stdout.write(f"\r  [{'#' * f}{'.' * (w - f)}] {done}/{total} answers scored")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def main():
    sys.stdout.write("loading model ... "); sys.stdout.flush()
    infer.init()
    print("ready.\n")
    print("Enter each player's game (blank line to finish):")
    print("  'crane 00201 00100 22222'  (starter known)   or   '00201 00100 22222'  (colours only)\n")
    grids, starters = [], []
    while True:
        try:
            line = input(f"player {len(grids)+1}: ").strip()
        except EOFError:
            break
        if not line:
            break
        sw, pats = parse_line(line)
        if not pats:
            print("  need at least one colour pattern"); continue
        if sw and sw not in ANSWER_SET:
            print(f"  starter '{sw}' not in word list -> treating opener as unknown"); sw = None
        grids.append(pats); starters.append(sw)
    if not grids:
        return
    known = sum(s is not None for s in starters)
    print(f"\nsolving over {len(grids)} players ({known} with known starter) ...")
    ranked, cand, total, n_legal = infer.solve(grids, starters=starters, top=20,
                                                beam=25, prune_to=1000, progress=bar)
    print(f"\n{n_legal} legal answers consistent with all clues.\n")
    print("Top answer choices:")
    for i, (w, pr) in enumerate(ranked, 1):
        print(f"  {i:2}. {w}   {pr*100:5.1f}%")


if __name__ == "__main__":
    main()
