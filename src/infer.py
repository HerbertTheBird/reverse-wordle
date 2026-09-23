"""
Infer the day's answer from COLOUR-ONLY grids.

P(A | grids) ∝ π(A) · Π_players P(grid | A)          (players independent)
P(grid | A) via a BFS/beam forward over pool states (branches NOT collapsed):
  row 1     : branch over openers, weighted by empirical BlueSky starter frequency
  rows 2..k : branch over words producing the observed colour; P(play W | state) from
              the softmax NN, normalised over a bounded plausible choice set
              (remaining pool + common human words), with mass preserved on pruning.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, math, sys, functools
import numpy as np
try:
    import lightgbm as _lgb                              # import BEFORE torch (libomp order)
except Exception:
    _lgb = None
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wordlists import (ANSWERS, IDX, scores_over_pool, scores_vs_answer, PM, STARTERS, STARTER_FLOOR)
from nn import Net, FEATS, LOG1P, LOGF
import engine
import prior
import feats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = ANSWERS
N = len(POOL)
_ARR = np.array([[ord(c) - 97 for c in w] for w in POOL], dtype=np.int8)
_POW3 = np.array([81, 27, 9, 3, 1], dtype=np.int64)

NET = MU = SD = PKV = SFV = NSV = ZFV = None
BST = None              # LightGBM ranker (hard negatives), loaded if present
TEMP = 1.0              # softmax temperature for the tree's choice-set normalisation
OPENERS = None          # (indices, freqs) of common openers
PROBES = None           # candidate "words a human might type" for the denominator
EG_X = EG_Y = None      # calibrated E[guesses](size) curve
BEAM = 12
BUCKET_CAP = 64         # cap words scored per colour-bucket (self-norm probs concentrate on top-played)
DENOM = 100             # (unused now) legacy denominator size


def np_scores_answer(answer):
    """score(W, answer) for every guess W -> pattern-matrix column (O(1) lookup)."""
    return scores_vs_answer(IDX[answer])


def eg(size):
    s = np.asarray(size, float)
    out = np.where(s <= 1, 1.0, np.interp(np.log2(np.clip(s, 1, None)), EG_X, EG_Y))
    return out if out.ndim else float(out)


def eg_exact(idx):
    return 1.0 if len(idx) <= 1 else engine.eguess([POOL[k] for k in idx])


def init():
    global NET, MU, SD, PKV, SFV, NSV, ZFV, OPENERS, PROBES, EG_X, EG_Y
    ck = torch.load(os.path.join(ROOT, "data", "model.pt"), weights_only=False)
    NET = Net(len(ck["feats"])); NET.load_state_dict(ck["state"]); NET.eval()
    MU, SD = ck["mu"], ck["sd"]
    import pandas as pd
    dfp = pd.read_excel(os.path.join(ROOT, "data", "English_Word_Prevalences.xlsx"))
    dfp["Word"] = dfp["Word"].astype(str).str.lower()
    pk = {r.Word: r.Pknown for r in dfp.itertuples() if isinstance(r.Word, str)}
    zf = {r.Word: r.FreqZipfUS for r in dfp.itertuples() if isinstance(r.Word, str)}
    PKV = np.array([pk.get(w, 0.02) for w in POOL])
    ZFV = np.array([zf.get(w, 1.0) for w in POOL])
    SFV = np.array([STARTERS.get(w, STARTER_FLOOR) for w in POOL])
    _ns = json.load(open(os.path.join(ROOT, "data", "nonstart-freq.json")))
    NSV = np.array([_ns["freq"].get(w, _ns["floor"]) for w in POOL])
    OPENERS = (np.array([IDX[w] for w in STARTERS if w in IDX]),
               np.array([STARTERS[w] for w in STARTERS if w in IDX]))
    PROBES = np.argsort(-SFV)[:DENOM]                       # words people actually type
    # calibrated E[guesses](size) curve from the training table (falls back to a
    # precomputed npz of the same curve when train.parquet isn't present)
    _trp = os.path.join(ROOT, "data", "train.parquet")
    _egp = os.path.join(ROOT, "data", "eg_curve.npz")
    if os.path.exists(_trp):
        tr = pd.read_parquet(_trp)
        pairs = np.vstack([tr[["answers_now", "guesses_now"]].to_numpy(),
                           tr[["answers_after", "guesses_after"]].to_numpy()])
        x = np.log2(np.clip(pairs[:, 0], 1, None))
        o = np.argsort(x); ux, inv = np.unique(np.round(x[o], 3), return_inverse=True)
        uy = np.zeros_like(ux); np.add.at(uy, inv, pairs[o, 1][:, ]); uy /= np.bincount(inv)
        EG_X, EG_Y = ux, uy
    else:
        _c = np.load(_egp); EG_X, EG_Y = _c["x"], _c["y"]
    global BST
    p = os.path.join(ROOT, "data", "model_lgb_hard.txt")
    if _lgb is not None and os.path.exists(p):
        BST = _lgb.Booster(model_file=p)


def _transform(F):
    X = F.astype(np.float32).copy()
    for j, f in enumerate(FEATS):
        if f in LOG1P:
            X[:, j] = np.log1p(X[:, j])
        elif f in LOGF:
            X[:, j] = np.log(np.clip(X[:, j], 1e-12, None))
    return (X - MU) / SD


def play_logits(cand_idx, pool, ci, gnow, gused, state):
    """NN logits for candidate words at a state (uses the shared feature module)."""
    F = feats.feature_matrix(cand_idx, pool, ci, gnow, gused, state, PKV, SFV, NSV, ZFV, eg)
    with torch.no_grad():
        return NET(torch.tensor(feats.transform(F, MU, SD))).numpy()


def tree_score(cand_idx, pool, ci, gnow, gused, state):
    """LightGBM-hard ranker score per candidate (raw features). Higher = more likely
    a human plays it. Use for ranking among a FIXED candidate set (helper/reconstruct)."""
    F = feats.feature_matrix(cand_idx, pool, ci, gnow, gused, state, PKV, SFV, NSV, ZFV, eg)
    return BST.predict(F)


def rank_score(cand_idx, pool, ci, gnow, gused, state):
    """Best available scorer: the hard-negative tree if loaded, else the NN logits."""
    if BST is not None:
        return tree_score(cand_idx, pool, ci, gnow, gused, state)
    return play_logits(cand_idx, pool, ci, gnow, gused, state)


def score_forward(grids, cand_idx, beam=BEAM, starters=None, progress=None):
    ll = np.zeros(len(cand_idx))
    op_idx, op_freq = OPENERS
    for cpos, ci in enumerate(cand_idx):
        codesWA = np_scores_answer(POOL[ci])          # score(W, A) for all W
        total = 0.0
        for gi, grid in enumerate(grids):
            patterns = [p for p in grid if p != 242]  # drop uninformative all-green rows
            if not patterns:
                continue
            p0 = patterns[0]
            branches = {}                              # key -> [weight, pool, history]
            sw = starters[gi] if starters else None
            if sw:
                # ---- row 1: the opener word is KNOWN -> single certain branch ----
                pool = np.where(scores_over_pool(sw) == p0)[0]
                hist = ((sw, int(p0)),)
                branches[(pool.tobytes(), hist)] = [1.0, pool, hist]
            else:
                # ---- row 1: marginalise over the common openers by frequency ----
                mask = codesWA[op_idx] == p0
                if mask.any():
                    for k, f in zip(op_idx[mask], op_freq[mask]):
                        pool = np.where(scores_over_pool(POOL[k]) == p0)[0]
                        hist = ((POOL[k], int(p0)),)
                        key = (pool.tobytes(), hist)
                        if key in branches:
                            branches[key][0] += f
                        else:
                            branches[key] = [f, pool, hist]
                else:
                    pool = np.where(codesWA == p0)[0]
                    branches[(pool.tobytes(), ())] = [STARTER_FLOOR, pool, ()]
            branches = _beam(branches, beam)
            # ---- rows 2+ : branch over the words that produce the observed colour.
            # NN is self-normalised (NCE), so P(play W|state) = exp(logit) directly.
            for ri, p in enumerate(patterns[1:], start=1):
                matching = np.where(codesWA == p)[0]        # full-vocab bucket for colour p
                if len(matching) == 0:
                    branches = {}; break
                if len(matching) > BUCKET_CAP:              # keep the most-played words
                    matching = matching[np.argsort(-(NSV[matching] + SFV[matching]))[:BUCKET_CAP]]
                nb = {}
                for w, pool, hist in branches.values():
                    state = feats.parse_history(list(hist))
                    probs = np.exp(play_logits(matching, pool, ci, eg(len(pool)), ri, state))
                    for k, pr in zip(matching, probs):
                        sub = pool[scores_over_pool(POOL[k])[pool] == p]
                        nh = hist + ((POOL[k], int(p)),)
                        key = (sub.tobytes(), nh)
                        add = w * pr
                        if key in nb:
                            nb[key][0] += add
                        else:
                            nb[key] = [add, sub, nh]
                if not nb:
                    branches = {}; break
                branches = _beam(nb, beam)
            total += math.log(sum(b[0] for b in branches.values())) if branches else math.log(1e-9)
        ll[cpos] = total
        if progress:
            progress(cpos + 1, len(cand_idx))
    return ll


def tree_probs(pool, ci, gused, state):
    """P(play W | state) for a bounded choice set (pool + common words), via the tree
    ranker softmaxed over that set. Returns (choice_indices, probs)."""
    choice = np.unique(np.concatenate([pool, PROBES]))
    F = feats.feature_matrix(choice, pool, ci, eg(len(pool)), gused, state,
                             PKV, SFV, NSV, ZFV, eg)
    s = BST.predict(F)
    e = np.exp((s - s.max()) / TEMP)
    return choice, e / e.sum()


def score_forward_tree(grids, cand_idx, beam=BEAM, starters=None, progress=None):
    """Same forward as score_forward but rows 2+ score the choice set with the LightGBM
    ranker (choice-set softmax) instead of the self-normalised NN."""
    ll = np.zeros(len(cand_idx))
    op_idx, op_freq = OPENERS
    for cpos, ci in enumerate(cand_idx):
        codesWA = np_scores_answer(POOL[ci])
        total = 0.0
        for gi, grid in enumerate(grids):
            patterns = [p for p in grid if p != 242]
            if not patterns:
                continue
            p0 = patterns[0]
            branches = {}
            sw = starters[gi] if starters else None
            if sw:
                pool = np.where(scores_over_pool(sw) == p0)[0]; hist = ((sw, int(p0)),)
                branches[(pool.tobytes(), hist)] = [1.0, pool, hist]
            else:
                mask = codesWA[op_idx] == p0
                if mask.any():
                    for k, f in zip(op_idx[mask], op_freq[mask]):
                        pool = np.where(scores_over_pool(POOL[k]) == p0)[0]
                        hist = ((POOL[k], int(p0)),); key = (pool.tobytes(), hist)
                        if key in branches:
                            branches[key][0] += f
                        else:
                            branches[key] = [f, pool, hist]
                else:
                    pool = np.where(codesWA == p0)[0]
                    branches[(pool.tobytes(), ())] = [STARTER_FLOOR, pool, ()]
            branches = _beam(branches, beam)
            for ri, p in enumerate(patterns[1:], start=1):
                nb = {}
                for w, pool, hist in branches.values():
                    choice, cp = tree_probs(pool, ci, ri, feats.parse_history(list(hist)))
                    m = codesWA[choice] == p
                    for k, pr in zip(choice[m], cp[m]):
                        sub = pool[scores_over_pool(POOL[k])[pool] == p]
                        nh = hist + ((POOL[k], int(p)),); key = (sub.tobytes(), nh)
                        add = w * float(pr)
                        if key in nb:
                            nb[key][0] += add
                        else:
                            nb[key] = [add, sub, nh]
                if not nb:
                    branches = {}; break
                branches = _beam(nb, beam)
            total += math.log(sum(b[0] for b in branches.values())) if branches else math.log(1e-9)
        ll[cpos] = total
        if progress:
            progress(cpos + 1, len(cand_idx))
    return ll


def _beam(branches, beam):
    if len(branches) <= beam:
        return branches
    tot = sum(b[0] for b in branches.values())
    keep = sorted(branches.items(), key=lambda kv: -kv[1][0])[:beam]
    ktot = sum(b[0] for _, b in keep)
    scale = tot / ktot if ktot > 0 else 1.0            # preserve mass on prune
    return {k: [b[0] * scale, b[1], b[2]] for k, b in keep}


def achievable_mask(grids):
    """Boolean over all answers: answer survives iff every observed colour row is
    producible by some guess against it. Vectorised over the pattern matrix."""
    observed = {int(p) for g in grids for p in g if int(p) != 242}
    mask = np.ones(N, bool)
    for p in observed:
        mask &= (PM == p).any(axis=0)               # answers whose column contains pattern p
    return mask


def cheap_score(grids, cand):
    """Fast shortlist score: prior + row-1 opener log-likelihood (no branching)."""
    op_idx, op_freq = OPENERS
    s = prior.log_prior(PKV[cand])
    for pos, ci in enumerate(cand):
        codesWA = np_scores_answer(POOL[ci])
        for grid in grids:
            s[pos] += math.log(op_freq[codesWA[op_idx] == grid[0]].sum() + STARTER_FLOOR)
    return s


def solve(grids_str, starters=None, top=20, beam=6, prune_to=250, progress=None, use_tree=False):
    """grids_str: list of players' grids (each a list of '00201'-style colour strings).
    starters: optional list, one per player, each a known opener word or None.
    Returns (ranked [(word, prob)], cand_idx, total, n_legal)."""
    grids = [[int(r, 3) for r in g] for g in grids_str]
    cand = np.arange(N)
    if starters:                                         # hard-anchor players whose opener is known
        for gi, sw in enumerate(starters):
            if sw:
                row = scores_over_pool(sw)               # pattern of sw vs every answer
                cand = cand[row[cand] == grids[gi][0]]
    amask = achievable_mask(grids)
    cand = cand[amask[cand]]
    n_legal = len(cand)                                  # answers consistent with everything
    if len(cand) > prune_to:                             # cheap shortlist before the forward
        cand = cand[np.argsort(-cheap_score(grids, cand))[:prune_to]]
    fwd = score_forward_tree if (use_tree and BST is not None) else score_forward
    ll = fwd(grids, cand, beam, starters=starters, progress=progress)
    total = ll + prior.log_prior(PKV[cand])              # trained P(is-answer | pknown)
    p = np.exp(total - total.max()); p /= p.sum()
    order = np.argsort(-total)
    engine.save()
    return [(POOL[cand[j]], float(p[j])) for j in order[:top]], cand, total, n_legal


if __name__ == "__main__":
    init()
    grids = json.load(open(sys.argv[1]))
    for w, pr in solve(grids)[0]:
        print(f"{w}  {pr*100:.1f}%")
