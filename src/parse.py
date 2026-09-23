"""
Parse Scoredle result comments into structured games.

A Scoredle comment looks like:
    Scoredle 4/6*
    14,855
    <colors> >!WORD!< (answers_remaining_after)
    ...
    🟩🟩🟩🟩🟩 >!ANSWER!<

We keep only games where every guessed row exposes its WORD (in spoiler tags),
including the final all-green answer row, so we have full ground truth.

Colors: 🟩 green=2, 🟨 yellow=1, ⬜/⬛ grey=0.

Output: psych/data/games.jsonl  (one game per line)
  {thread_id, comment_id, score, answer, n_guesses, start_pool,
   guesses:[{word, pattern, answers_after}]}
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMENTS_DIR = os.path.join(ROOT, "raw", "comments")
OUT = os.path.join(ROOT, "data", "games.jsonl")

GREEN, YELLOW = "\U0001F7E9", "\U0001F7E8"
GREYS = {"⬜", "⬛"}  # white / black large square
COLOR = {GREEN: "2", YELLOW: "1"}
for g in GREYS:
    COLOR[g] = "0"
COLOR_CHARS = set(COLOR)

WORD_RE = re.compile(r">!\s*([A-Za-z]{5})\s*!<")
PAREN_RE = re.compile(r"\((\d[\d,]*)\)")


def parse_row(line):
    """Return (pattern, word|None, answers_after|None) if the line is a guess row."""
    colors = [ch for ch in line if ch in COLOR_CHARS]
    if len(colors) != 5:
        return None
    pattern = "".join(COLOR[ch] for ch in colors)
    wm = WORD_RE.search(line)
    word = wm.group(1).lower() if wm else None
    pm = PAREN_RE.search(line)
    answers_after = int(pm.group(1).replace(",", "")) if pm else None
    return pattern, word, answers_after


def parse_comment(body):
    if "Scoredle" not in body:
        return None
    rows = []
    for line in body.splitlines():
        r = parse_row(line)
        if r:
            rows.append(r)
    if len(rows) < 1:
        return None
    # final row must be the all-green answer
    if rows[-1][0] != "22222":
        return None
    # require every row to carry its word (full ground truth)
    if any(w is None for (_, w, _) in rows):
        return None
    answer = rows[-1][1]
    # sanity: non-final green rows shouldn't exist before the end
    guesses = [{"word": w, "pattern": p, "answers_after": a} for (p, w, a) in rows]
    m = re.search(r"Scoredle\s+(\d)\s*/\s*6", body)
    score = int(m.group(1)) if m else len(rows)
    return {"answer": answer, "n_guesses": len(rows), "guesses": guesses, "score": score}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = sorted(f for f in os.listdir(COMMENTS_DIR) if f.endswith(".jsonl")) \
        if os.path.isdir(COMMENTS_DIR) else []
    n_games = n_comments = 0
    with open(OUT, "w") as out:
        for fn in files:
            thread_id = fn[:-6]
            for line in open(os.path.join(COMMENTS_DIR, fn)):
                c = json.loads(line)
                n_comments += 1
                g = parse_comment(c.get("body", ""))
                if not g:
                    continue
                g["thread_id"] = thread_id
                g["comment_id"] = c["id"]
                out.write(json.dumps(g) + "\n")
                n_games += 1
    print(f"parsed {n_games} full games from {n_comments} comments -> {OUT}")


if __name__ == "__main__":
    main()
