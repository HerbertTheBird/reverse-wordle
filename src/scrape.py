"""
Scrape r/wordle daily threads (posted by u/Scoredle) and their comments via the
Arctic Shift API (https://arctic-shift.photon-reddit.com). No auth required.

Outputs (JSONL, resumable):
  psych/raw/threads.jsonl              one line per daily thread (id, title, created_utc)
  psych/raw/comments/<thread_id>.jsonl one line per comment on that thread

Usage:
  python src/scrape.py threads --max 1200
  python src/scrape.py comments            # fetch comments for every known thread
"""
import json, os, sys, time, urllib.parse, urllib.request

API = "https://arctic-shift.photon-reddit.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
COMMENTS_DIR = os.path.join(RAW, "comments")
THREADS_FILE = os.path.join(RAW, "threads.jsonl")


def get(path, params, tries=5):
    url = API + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wordle-psych-research"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            wait = 2 * (attempt + 1)
            print(f"  retry {attempt+1}/{tries} after {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def scrape_threads(max_threads):
    os.makedirs(RAW, exist_ok=True)
    seen = set()
    if os.path.exists(THREADS_FILE):
        for line in open(THREADS_FILE):
            seen.add(json.loads(line)["id"])
    before = None
    out = open(THREADS_FILE, "a")
    total = len(seen)
    while total < max_threads:
        params = {"author": "Scoredle", "subreddit": "wordle", "limit": 100, "sort": "desc"}
        if before:
            params["before"] = before
        data = get("/api/posts/search", params).get("data", [])
        if not data:
            print("no more threads")
            break
        new = 0
        for p in data:
            before = p["created_utc"] if before is None else min(before, p["created_utc"])
            if p["id"] in seen:
                continue
            # only keep the daily puzzle threads
            if "Daily Wordle" not in (p.get("title") or ""):
                continue
            rec = {"id": p["id"], "title": p.get("title"),
                   "created_utc": p["created_utc"], "permalink": p.get("permalink")}
            out.write(json.dumps(rec) + "\n")
            seen.add(p["id"]); new += 1; total += 1
        out.flush()
        print(f"threads: {total} (+{new}); before={before}")
        if new == 0 and len(data) < 100:
            break
        time.sleep(0.5)
    out.close()
    print(f"done, {total} threads in {THREADS_FILE}")


def scrape_comments_for(thread_id):
    os.makedirs(COMMENTS_DIR, exist_ok=True)
    path = os.path.join(COMMENTS_DIR, f"{thread_id}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0  # already done
    after = None
    seen = set()
    rows = []
    while True:
        params = {"link_id": thread_id, "limit": 100, "sort": "asc"}
        if after:
            params["after"] = after
        data = get("/api/comments/search", params).get("data", [])
        if not data:
            break
        progressed = False
        for c in data:
            after = c["created_utc"] if after is None else max(after, c["created_utc"])
            if c["id"] in seen:
                continue
            seen.add(c["id"]); progressed = True
            rows.append({"id": c["id"], "body": c.get("body", ""),
                         "score": c.get("score"), "created_utc": c["created_utc"]})
        if len(data) < 100 or not progressed:
            break
        time.sleep(0.2)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def scrape_all_comments():
    threads = [json.loads(l) for l in open(THREADS_FILE)]
    for i, t in enumerate(threads):
        n = scrape_comments_for(t["id"])
        if n:
            print(f"[{i+1}/{len(threads)}] {t['title']}: {n} comments")
        time.sleep(0.1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "threads"
    if cmd == "threads":
        mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 1200
        scrape_threads(mx)
    elif cmd == "comments":
        scrape_all_comments()
    else:
        print(__doc__)
