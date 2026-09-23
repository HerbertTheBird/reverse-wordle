"""Rebuild the 12,972 x 12,972 pattern matrix (data/pattern_matrix.npy, uint8).
PM[i, j] = base-3 pattern code of guessing ANSWERS[i] against answer ANSWERS[j]."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wordlists as W

N = len(W.ANSWERS)
PM = np.empty((N, N), np.uint8)
t0 = time.time()
for i, w in enumerate(W.ANSWERS):
    PM[i] = W._compute_scores(w).astype(np.uint8)
    if i % 2000 == 0:
        print(f"  {i}/{N}  ({time.time()-t0:.0f}s)", flush=True)
out = os.path.join(W.ROOT, "data", "pattern_matrix.npy")
np.save(out, PM)
print(f"saved {out}  shape={PM.shape} dtype={PM.dtype}  ({time.time()-t0:.0f}s)")
