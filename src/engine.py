"""
Persistent cache over the repo's optimal-guesses engine (EGuess.java).

Computes E[optimal guesses to solve | candidate answer set] and caches results to
disk so the bot's numbers are computed once and reused across runs. A single long-
lived Java 'server' process answers queries without per-call JVM startup.
"""
import atexit, hashlib, os, pickle, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
CACHE_FILE = os.path.join(ROOT, "data", "eguess_cache.pkl")

_cache = {}
_dirty = 0
_proc = None
if os.path.exists(CACHE_FILE):
    try:
        _cache = pickle.load(open(CACHE_FILE, "rb"))
    except Exception:
        _cache = {}


def _key(words):
    return hashlib.sha1(" ".join(sorted(words)).encode()).digest()[:16]


def _server():
    global _proc
    if _proc is None:
        _proc = subprocess.Popen(["java", "-cp", SRC, "EGuess", "server"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    return _proc


def eguess(words):
    """E[optimal guesses] for a candidate answer set (list/tuple of words)."""
    n = len(words)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    k = _key(words)
    v = _cache.get(k)
    if v is not None:
        return v
    p = _server()
    p.stdin.write(" ".join(sorted(words)) + "\n")
    p.stdin.flush()
    v = float(p.stdout.readline())
    _cache[k] = v
    global _dirty
    _dirty += 1
    return v


def save():
    """Merge with whatever is on disk before writing, so concurrent processes
    (e.g. an eval and a feature build) don't clobber each other's cached results."""
    global _dirty, _cache
    if not _dirty:
        return
    if os.path.exists(CACHE_FILE):
        try:
            disk = pickle.load(open(CACHE_FILE, "rb"))
            disk.update(_cache)
            _cache = disk
        except Exception:
            pass
    tmp = CACHE_FILE + f".tmp{os.getpid()}"
    pickle.dump(_cache, open(tmp, "wb"))
    os.replace(tmp, CACHE_FILE)
    _dirty = 0


atexit.register(save)


def stats():
    return {"cached_sets": len(_cache)}
