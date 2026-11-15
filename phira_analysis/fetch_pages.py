"""Phira 全量元数据抓取：api.phira.cn/chart?page=N，每页30条，共326页。"""
import json, os, socket, time, urllib.request, concurrent.futures as cf
_o = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _o(h, p, socket.AF_INET, t, pr, fl)

from paths import WORK  # 语料根目录由 paths.py 统一解析（PHIRA_WORK 优先）
OUT = os.path.join(WORK, "pages"); os.makedirs(OUT, exist_ok=True)
BASE = "https://api.phira.cn/chart?page={}"
HDR = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def path(p): return os.path.join(OUT, f"{p:03d}.json")

def ok(p):
    f = path(p)
    if not os.path.exists(f) or os.path.getsize(f) < 50: return False
    try:
        d = json.load(open(f))
        return isinstance(d, list) and len(d) > 0
    except Exception: return False

def fetch(p):
    if ok(p): return p, "cached", 0
    for i in range(12):
        try:
            req = urllib.request.Request(BASE.format(p), headers=HDR)
            with urllib.request.urlopen(req, timeout=20) as r:
                body = json.loads(r.read().decode())
            rows = body.get("results", [])
            if not rows: return p, "empty", 0
            json.dump(rows, open(path(p), "w"), ensure_ascii=False)
            return p, "ok", len(rows)
        except Exception as e:
            err = str(e)[:50]; time.sleep(min(1 + i * 1.2, 10))
    return p, "FAIL:" + err, 0

# 先探总页数
total = 9760; pages = list(range(1, total // 30 + 2))
print(f"目标 {len(pages)} 页 (count={total})", flush=True)
done = 0
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for p, s, n in ex.map(fetch, pages):
        done += 1
        if s.startswith("FAIL"): print(f"page {p} {s}", flush=True)
        if done % 30 == 0 or done == len(pages): print(f"  {done}/{len(pages)}", flush=True)

rows = []
for p in pages:
    if ok(p): rows.extend(json.load(open(path(p))))
seen = {r["id"]: r for r in rows}
print(f"抓取完成：{len(rows)} 行 → 唯一 id {len(seen)}", flush=True)
json.dump(list(seen.values()), open(os.path.join(WORK, "charts_meta.json"), "w"), ensure_ascii=False)
print("已写 charts_meta.json", flush=True)
