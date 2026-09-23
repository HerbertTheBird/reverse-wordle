"""
Shared feature computation for P(guess | state) -- used by features.py (training),
helper.py, and infer.py so train and inference stay identical.

66 features:
  base(10): answers_now, guesses_now, pknown, answers_after, guesses_after,
            starter_freq, regret, exp_regret, guesses_used, nonstart_freq
  extra(4): count_new (fresh letters in W), is_answer (W still possible),
            yellow_pct (yellow letters re-placed in valid spots), greens_sat
  onehot:   used_00..used_25 (letters used in prior guesses),
            gw_00..gw_25      (letters in candidate word W)
"""
import math
import numpy as np
from wordlists import ANSWERS, IDX, scores_over_pool, PM

N = len(ANSWERS)
WLET = np.array([[ord(c) - 97 for c in w] for w in ANSWERS], dtype=np.int64)   # (N,5)
WHOT = np.zeros((N, 26), np.int8)
for _i in range(N):
    for _c in WLET[_i]:
        WHOT[_i, _c] = 1
_POW3 = [81, 27, 9, 3, 1]

BASE = ["answers_now", "guesses_now", "pknown", "answers_after", "guesses_after",
        "starter_freq", "regret", "exp_regret", "guesses_used", "nonstart_freq"]
EXTRA = ["count_new", "is_answer", "yellow_pct", "greens_sat",
         "repeats", "known_present", "known_absent",
         "entropy", "zipf", "is_repeat"]
USED = [f"used_{i:02d}" for i in range(26)]
GW = [f"gw_{i:02d}" for i in range(26)]
FEATS = BASE + EXTRA + USED + GW
LOG1P = {"answers_now", "answers_after"}
LOGF = {"starter_freq", "nonstart_freq"}


def parse_history(hist):
    """hist: list of (word_str, pattern_int). Returns the accumulated game state.
    yellow_known[L] = (max over guesses of #green+yellow of L in one guess) - #greens(L)
                    = instances of L known present but not yet pinned by a green."""
    used = np.zeros(26, np.int8)
    green = -np.ones(5, np.int64)
    known = np.zeros(26, np.int64)          # max multiplicity of L seen in any one guess
    excluded = np.zeros((26, 5), bool)
    for w, p in hist:
        digs = [(p // _POW3[i]) % 3 for i in range(5)]
        gy = np.zeros(26, np.int64)          # green+yellow count of each letter this guess
        for i, ch in enumerate(w):
            L = ord(ch) - 97
            used[L] = 1
            if digs[i] == 2:
                green[i] = L; gy[L] += 1
            elif digs[i] == 1:
                gy[L] += 1; excluded[L, i] = True
            else:
                excluded[L, i] = True
        known = np.maximum(known, gy)
    green_conf = np.zeros(26, np.int64)
    for i in range(5):
        if green[i] >= 0:
            green_conf[green[i]] += 1
    yellow_known = np.maximum(known - green_conf, 0)
    greenpos = green >= 0
    valid = (~excluded) & (~greenpos[None, :])           # position i valid for letter L
    present = known > 0                                  # letters known IN the word (green/yellow)
    grey_seen = excluded.any(1)                          # letters that appeared grey somewhere
    absent = grey_seen & (~present)                      # letters known NOT in the word
    prior = np.array([IDX[w] for w, _ in hist if w in IDX], dtype=np.int64)  # already-played words
    return {"used": used, "green": green, "greenpos": greenpos, "ycount": yellow_known,
            "valid": valid, "present": present, "absent": absent, "prior": prior}


def feature_matrix(cand_idx, pool, ci, gnow, gused, state, PKV, SFV, NSV, ZFV, eg):
    """(len(cand_idx), 66) feature matrix, FEATS order, for candidate answer index ci.
    Vectorised (identical values to the per-word reference)."""
    cand_idx = np.asarray(cand_idx)
    m = len(cand_idx)
    F = np.zeros((m, len(FEATS)), np.float32)
    anow = len(pool)
    used, green, greenpos = state["used"], state["green"], state["greenpos"]
    ycount, valid = state["ycount"], state["valid"]
    tot_y = int(ycount.sum())
    WL = WLET[cand_idx]                                   # (m,5)
    WH = WHOT[cand_idx]                                   # (m,26)

    # answers_after: for each candidate word, how many pool answers share its colour
    # vs the true answer. Fully vectorised via the pattern matrix.
    npool = len(pool)
    if PM is not None:
        sub = PM[np.ix_(cand_idx, pool)]                 # (m, |pool|) colours vs pool
        col = PM[cand_idx, ci]                           # (m,) colour vs the answer
        aaft = (sub == col[:, None]).sum(1).astype(float)
        # entropy of the guess's colour split over the pool (info gain)
        counts = np.zeros((m, 243))
        np.add.at(counts, (np.repeat(np.arange(m), npool), sub.ravel()), 1.0)
        p = counts / npool
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -(np.where(p > 0, p * np.log2(p), 0.0)).sum(1)
    else:
        aaft = np.array([(scores_over_pool(ANSWERS[k])[pool] ==
                          scores_over_pool(ANSWERS[k])[ci]).sum() for k in cand_idx], float)
        entropy = np.zeros(m)
    gaft = np.asarray(eg(aaft), float)                   # eg must accept arrays
    regret = gaft - (gnow - 1.0)

    greens_sat = ((WL == green[None, :]) & greenpos[None, :]).sum(1)
    count_new = (WH * (used == 0)[None, :]).sum(1)
    if tot_y:
        place = np.zeros((m, 26))
        for i in range(5):
            L = WL[:, i]
            np.add.at(place, (np.arange(m), L), valid[L, i].astype(float))
        ypct = np.minimum(place, ycount[None, :]).sum(1) / tot_y
    else:
        ypct = np.zeros(m)
    is_ans = np.isin(cand_idx, pool).astype(float)
    present, absent = state["present"], state["absent"]
    repeats = 5 - WH.sum(1)                               # duplicate letters in the guess
    known_present = (WH * present[None, :]).sum(1)        # guess letters known IN the word
    known_absent = (WH * absent[None, :]).sum(1)          # guess letters known NOT in the word
    is_repeat = np.isin(cand_idx, state["prior"]).astype(float)   # already played this game

    nb, ne = len(BASE), len(EXTRA)
    F[:, 0] = anow; F[:, 1] = gnow; F[:, 2] = PKV[cand_idx]; F[:, 3] = aaft
    F[:, 4] = gaft; F[:, 5] = SFV[cand_idx]; F[:, 6] = regret
    F[:, 7] = np.exp(np.minimum(regret, 10.0)); F[:, 8] = gused; F[:, 9] = NSV[cand_idx]
    F[:, 10] = count_new; F[:, 11] = is_ans; F[:, 12] = ypct; F[:, 13] = greens_sat
    F[:, 14] = repeats; F[:, 15] = known_present; F[:, 16] = known_absent
    F[:, 17] = entropy; F[:, 18] = ZFV[cand_idx]; F[:, 19] = is_repeat
    F[:, nb + ne:nb + ne + 26] = used
    F[:, nb + ne + 26:nb + ne + 52] = WH
    return F


def transform(F, mu, sd):
    X = F.astype(np.float32).copy()
    for j, f in enumerate(FEATS):
        if f in LOG1P:
            X[:, j] = np.log1p(X[:, j])
        elif f in LOGF:
            X[:, j] = np.log(np.clip(X[:, j], 1e-12, None))
    return (X - mu) / sd
