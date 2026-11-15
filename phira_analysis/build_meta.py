"""Phira 全量元数据：keep-alive 每线程一条连接（关键，否则每页要 25s）。"""
import json, os, socket, threading, time, http.client, concurrent.futures as cf

_o = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _o(h, p, socket.AF_INET, t, pr, fl)

from paths import WORK  # 语料根目录由 paths.py 统一解析（PHIRA_WORK 优先）
OUT = os.path.join(WORK, "pages"); os.makedirs(OUT, exist_ok=True)
TL = threading.local()


def path(p): return os.path.join(OUT, f"{p:03d}.json")


def ok(p):
    f = path(p)
    if not os.path.exists(f) or os.path.getsize(f) < 20: return False
    try:
        d = json.load(open(f))
        return isinstance(d, list) and len(d) > 0
    except Exception: return False


def conn():
    c = getattr(TL, "c", None)
    if c is None:
        c = http.client.HTTPSConnection("api.phira.cn", timeout=90)
        TL.c = c
    return c


def get(p, tries=8):
    for i in range(tries):
        try:
            c = conn()
            c.request("GET", f"/chart?page={p}", headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            r = c.getresponse(); body = r.read()
            if r.status != 200: raise ValueError(f"HTTP {r.status}")
            return json.loads(body.decode())
        except Exception:
            try: TL.c.close()
            except Exception: pass
            TL.c = None
            time.sleep(min(2 + i * 1.5, 15))
    return None


def fetch(p):
    if ok(p): return p, "cached", 0
    body = get(p)
    if body is None: return p, "FAIL", 0
    rows = body.get("results", [])
    if not rows: return p, "empty", 0
    json.dump(rows, open(path(p), "w"), ensure_ascii=False)
    return p, "ok", len(rows)


probe = get(1)
count = probe["count"] if probe else 9760
LAST = (count + 29) // 30
pages = list(range(1, LAST + 1))
if probe and not ok(1):
    json.dump(probe["results"], open(path(1), "w"), ensure_ascii=False)
print(f"count={count} last={LAST} cached={sum(1 for p in pages if ok(p))}", flush=True)

todo = [p for p in pages if not ok(p)]
t0 = time.time()
fails = []
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for n, (p, s, k) in enumerate(ex.map(fetch, todo), 1):
        if s == "FAIL": fails.append(p); print(f"page {p} FAIL", flush=True)
        if n % 50 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} {time.time()-t0:.0f}s", flush=True)

rows = []
for p in pages:
    if ok(p): rows.extend(json.load(open(path(p))))
seen = {r["id"]: r for r in rows}
json.dump(list(seen.values()), open(os.path.join(WORK, "charts_meta.json"), "w"), ensure_ascii=False)
stats = {"api_count": count, "last_page": LAST, "pages_good": sum(1 for p in pages if ok(p)),
         "pages_failed": fails, "rows_fetched": len(rows), "unique_ids": len(seen),
         "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(stats, open(os.path.join(WORK, "meta_stats.json"), "w"), ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False), flush=True)
