"""Shared word lists + scoring, mirroring the repo's Java engine (Bot.java)."""
import os, re, functools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = ROOT                       # wordlists (guess.txt/answer.txt) live at the repo root


def load_answers():
    """Candidate/answer pool = the official valid-guess list (guess.txt, 12,972).
    Covers 100% of scraped 2026 day-answers (the classic answer.txt misses ~31%),
    and is a clean subset of Scoredle's pool (drops ~1,883 junk words)."""
    p = os.path.join(REPO, "guess.txt")
    if os.path.exists(p):
        return open(p).read().split()
    return open(os.path.join(ROOT, "data", "scoredle_answers.txt")).read().split()


def load_curated():
    """The classic curated answer list (answer.txt) -- used only as a prior signal."""
    p = os.path.join(REPO, "answer.txt")
    return set(open(p).read().split()) if os.path.exists(p) else set()


def load_starters():
    """Empirical P(a game opens with this word), from BlueSky (~1.78M games).
    Returns (dict word->prob, floor for unseen openers)."""
    import csv
    p = os.path.join(ROOT, "data", "first-guess-freq.csv")
    if not os.path.exists(p):
        return {}, 1e-7
    rows = list(csv.DictReader(open(p)))
    tot = sum(int(r["count"]) for r in rows)
    d = {r["word"]: int(r["count"]) / tot for r in rows}
    return d, 0.5 / tot               # half-count floor for never-seen openers


ANSWERS = load_answers()
ANSWER_SET = set(ANSWERS)
CURATED = load_curated()          # classic answers: strong "answer-like" prior
STARTERS, STARTER_FLOOR = load_starters()


@functools.lru_cache(maxsize=1 << 20)
def score(guess, answer):
    """Wordle feedback as a 5-char string of '0/1/2' (grey/yellow/green)."""
    res = [0] * 5
    matched = [False] * 5
    for i in range(5):
        if guess[i] == answer[i]:
            res[i] = 2
            matched[i] = True
    for i in range(5):
        if res[i] == 2:
            continue
        for j in range(5):
            if not matched[j] and guess[i] == answer[j]:
                res[i] = 1
                matched[j] = True
                break
    return "".join(str(x) for x in res)


def filter_pool(pool, guess, pattern):
    """Keep answers consistent with (guess -> pattern)."""
    return [w for w in pool if score(guess, w) == pattern]


# ---- vectorised scoring over the whole pool (shared by features.py / infer.py) ----
import numpy as _np
import functools as _ft

IDX = {w: i for i, w in enumerate(ANSWERS)}
_ARR = _np.array([[ord(c) - 97 for c in w] for w in ANSWERS], dtype=_np.int8)
_N = len(ANSWERS)
_POW3 = _np.array([81, 27, 9, 3, 1], dtype=_np.int64)
_COUNTS = _np.zeros((_N, 26), _np.int16)
for _i in range(_N):
    for _c in _ARR[_i]:
        _COUNTS[_i, _c] += 1


def _compute_scores(word):
    """Pattern code of `word` (as guess) against every pool answer -> int64 array."""
    g = _np.array([ord(c) - 97 for c in word], dtype=_np.int8)
    green = _ARR == g
    code = green.astype(_np.int64) * 2
    avail = _COUNTS.copy()
    for i in range(5):
        avail[_np.arange(_N), _ARR[:, i]] -= green[:, i]
    for i in range(5):
        can = (~green[:, i]) & (avail[_np.arange(_N), g[i]] > 0)
        code[can, i] = 1
        avail[can, g[i]] -= 1
    return code @ _POW3


# Full 12,972 x 12,972 pattern matrix: PM[i, j] = score(guess ANSWERS[i], answer ANSWERS[j]).
# Turns every per-word scoring into an O(1) array lookup.
_PM = None
_pm_path = os.path.join(ROOT, "data", "pattern_matrix.npy")
if os.path.exists(_pm_path):
    _PM = _np.load(_pm_path)                      # (N, N) uint8, in RAM (~168MB)
PM = _PM                                          # public handle: PM[guess_i, answer_j]


@_ft.lru_cache(maxsize=300000)
def scores_over_pool(word):
    """Pattern of `word` (as guess) vs every answer. Row of the pattern matrix
    (falls back to direct computation for words outside the vocabulary)."""
    if _PM is not None and word in IDX:
        return _PM[IDX[word]]
    return _compute_scores(word)


def scores_vs_answer(ai):
    """Pattern of every guess word vs answer index `ai`. Column of the pattern matrix."""
    if _PM is not None:
        return _PM[:, ai]
    return _np.array([_compute_scores(w)[ai] for w in ANSWERS])


if __name__ == "__main__":
    print("answers:", len(ANSWERS), ANSWERS[:3], ANSWERS[-3:])
