# Reverse Wordle

**Infer the day's Wordle answer from the color-only grids that many players post — without ever seeing the answer or the words anyone typed.**

Reverse Wordle models *how humans choose guesses*, then runs that model backwards. A grid of grey/yellow/green tiles is a lossy trace of a game; pooling many players' traces and asking "which hidden answer makes this crowd's guessing behavior most likely?" recovers the answer with high accuracy.

## The idea

Players are treated as independent. For a candidate answer `A`:

```
P(A | all grids) ∝ P(A) · Π_players Π_rows  Σ_{W : score(W, A) = observed row}  P(a human plays W)
```

For each color row, sum over every word that *could* have produced it against `A`, weighted by how likely a human is to actually play that word. `P(play W)` is a trained model of human guessing (the "psychology"). The answer with the highest posterior is the crowd's implied answer.

## Results

Color-only inference, 50 days each, `beam=25, prune=1000` (no answer peeking):

| players | top-1 accuracy | mean P(answer) |
|--------:|:--------------:|:--------------:|
|       5 |      74 %      |     0.68       |
|      10 |      92 %      |     0.90       |
|      20 |      94 %      |     0.93       |

From colors alone, pooling ~10 strangers identifies the exact daily answer **92 %** of the time.

## Layout

```
crowdle/
  guess.txt, answer.txt      # word lists (valid guesses / curated answers)
  src/                       # all code
  data/                      # models, priors, curves, corpus (big files gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/build_pm.py        # build the 12,972² pattern matrix (~40s, gitignored)
```

## Usage

- **`src/guess.py`** — crowd → answer. Paste many players' color grids (optionally known opener words); prints the top-20 answer candidates with probabilities and the legal-answer count. Uses `beam=25, prune_to=1000`.
- **`src/helper.py`** — single player: "here are my colors, what should I guess next?"
- **`src/reconstruct.py`** — given answer + colors, reconstruct the most likely game played.
- **`src/benchmark.py` / `src/bench_prob.py` / `src/scaling.py` / `src/scaling_par.py`** — offline evaluation (last one is a multiprocess parallel driver).

## How it works (pipeline)

1. **`scrape.py`** — pull r/wordle daily threads (Scoredle bot replies) via the Arctic Shift API.
2. **`parse.py`** — decode color grids + answers → `data/games.jsonl` (~204k games, ~1,196 answers).
3. **`feats.py`** — 72 features per (word, game-state): word prevalence (Brysbaert), opener frequency (BlueSky), optimal answers/expected-guesses remaining (Wordle engine), plus game-logic features. Shared by training and inference so they never drift.
4. **`nn.py`** — PyTorch MLP for `P(play W | state)`, trained with a self-normalizing NCE objective so `exp(logit) ≈ P` at inference (no partition function). `train_lgb.py` / `rank_nn.py` train ranking alternatives.
5. **`prior.py`** — trained logistic `P(is-answer | word prevalence)`.
6. **`infer.py`** — the Bayesian solve: achievability filter → cheap-score prune → branch-preserving beam forward → combine players → posterior. A precomputed 12,972² **pattern matrix** makes all scoring O(1).

## Speed notes

- `pattern_matrix.npy` turns per-word scoring into array lookups (biggest speed win); rebuild with `build_pm.py`.
- Self-normalized NN avoids normalization at inference.
- `prune_to` keeps the top-N candidates by a cheap row-1 score before the expensive forward; measured min-safe value is ~600 (so 1000 is a safe margin — the true answer is never dropped).
- `eg_curve.npz` is a precomputed E[guesses]-vs-size curve, so `infer` runs without the 149 MB training table or the Java engine.

## Java engine (optional)

`src/EGuess.java` computes optimal expected-guesses-remaining and was used offline to calibrate the size→E[guesses] curve (already cached in `data/eguess_curve.json` / `eg_curve.npz`). It's only needed to *regenerate* that curve; all inference is pure Python.
