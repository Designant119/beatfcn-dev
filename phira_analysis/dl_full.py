"""Phira 全量谱面 zip 下载（分片幂等，每线程 keep-alive 长连接）。

用法:
  python3 dl_full.py --ids .../shard_a_ids.txt --dest .../zips/a --workers 6 --tag a

- 复用旧子集 charts_by_rating（按 id 命中即硬链接，原目录只读）
- 已存在且 testzip 通过 + namelist 非空 → 跳过（幂等，中断重跑只补缺）
- 失败写 failed_<tag>.csv（id,name,reason），不静默跳过
"""
import argparse, csv, json, os, random, re, socket, threading, time, zipfile
import http.client, urllib.parse
import concurrent.futures as cf

_o = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _o(h, p, socket.AF_INET, t, pr, fl)

from paths import WORK, OLD  # 语料根/旧分档目录由 paths.py 统一解析（PHIRA_WORK / PHIRA_OLD 优先）
META = os.path.join(WORK, "charts_meta.json")
HDR = {"User-Agent": "Mozilla/5.0", "Accept": "*/*", "Connection": "keep-alive"}
TL = threading.local()


def safe(s, n=36):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(s)).strip()
    s = re.sub(r"_+", "_", s)
    return s[:n] or "untitled"


def fname(r):
    return f"{r['id']}_{safe(r['name'])}_{safe(r.get('level', ''), 8)}.zip"


def zip_ok(p):
    if not os.path.exists(p) or os.path.getsize(p) < 100:
        return False
    try:
        with zipfile.ZipFile(p) as z:
            if not z.namelist():
                return False
            return z.testzip() is None
    except Exception:
        return False


def old_index():
    idx = {}
    if not os.path.isdir(OLD):
        return idx
    for f in os.listdir(OLD):
        if f.endswith(".zip"):
            m = re.search(r"_(\d+)\.zip$", f)
            if m:
                idx.setdefault(int(m.group(1)), os.path.join(OLD, f))
    return idx


def link_or_copy(src, dst):
    tmp = dst + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        os.link(src, tmp)
    except Exception:
        import shutil
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def get_conn(host):
    c = getattr(TL, "conn", None)
    if c is not None and getattr(TL, "host", None) == host:
        return c
    if c is not None:
        try: c.close()
        except Exception: pass
    c = http.client.HTTPSConnection(host, timeout=120)
    TL.conn, TL.host = c, host
    return c


def drop_conn():
    try: TL.conn.close()
    except Exception: pass
    TL.conn = None


def download(url, dst, max_tries=5):
    err = "?"
    for i in range(max_tries):
        code = None
        try:
            # 手动跟 30x（重定向到 CDN 分片）
            target = url
            r = None
            for hop in range(4):
                u = urllib.parse.urlsplit(target)
                c = get_conn(u.netloc)
                c.request("GET", u.path + ("?" + u.query if u.query else ""), headers=HDR)
                r = c.getresponse()
                if r.status in (301, 302, 303, 307, 308):
                    loc = r.getheader("Location")
                    r.read()
                    if not loc:
                        raise ValueError(f"HTTP {r.status} no Location")
                    target = urllib.parse.urljoin(target, loc)
                    continue
                break
            if r.status != 200:
                raise ValueError(f"HTTP {r.status}")
            n = int(r.getheader("Content-Length") or 0)
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
            if n and os.path.getsize(tmp) != n:
                raise ValueError(f"short read {os.path.getsize(tmp)}/{n}")
            with zipfile.ZipFile(tmp) as z:
                if not z.namelist():
                    raise ValueError("zip namelist empty")
                if z.testzip() is not None:
                    raise ValueError("zip crc bad")
            os.replace(tmp, dst)
            return True, os.path.getsize(dst), ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            code = None
            m = re.search(r"HTTP (\d+)", err)
            if m:
                code = int(m.group(1))
            drop_conn()          # 连接可能被带坏，丢掉重建
        if code in (429, 503, 502, 504):
            time.sleep(min(15 * (i + 1), 75) + random.uniform(0, 8))
        else:
            time.sleep(min(1.5 ** i, 12) + random.uniform(0, 1.5))
    return False, 0, err[:100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    os.makedirs(a.dest, exist_ok=True)
    meta = {r["id"]: r for r in json.load(open(META))}
    ids = [int(x) for x in open(a.ids) if x.strip()]
    old = old_index()
    targets = [meta[i] for i in ids if i in meta]
    print(f"[{a.tag}] ids={len(ids)} meta_hit={len(targets)} old_index={len(old)}", flush=True)

    st = {"tag": a.tag, "ids": len(ids), "targets": len(targets), "skip_done": 0,
          "reuse_old": 0, "reuse_bad": 0, "downloaded": 0, "failed": 0,
          "bytes_new": 0, "bytes_total": 0, "no_old_no_dl": 0}
    failed = []

    def work(r):
        dst = os.path.join(a.dest, fname(r))
        if zip_ok(dst):
            return r, "skip", os.path.getsize(dst), ""
        src = old.get(r["id"])
        if src and zip_ok(src):
            try:
                link_or_copy(src, dst)
                return r, "reuse", os.path.getsize(dst), ""
            except Exception as e:
                return r, "fail", 0, f"reuse:{type(e).__name__}:{e}"[:100]
        ok, size, err = download(r["file"], dst)
        return r, ("ok" if ok else "fail"), size, err

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, r) for r in targets]
        done = 0
        for f in cf.as_completed(futs):
            r, s, size, err = f.result()
            done += 1
            st["skipped" if False else s + "_done"] = st.get(s + "_done", 0) + 1
            if s == "ok":
                st["downloaded"] += 1; st["bytes_new"] += size
            elif s == "reuse":
                st["reuse_old"] += 1
            elif s == "fail":
                st["failed"] += 1; failed.append((r["id"], r["name"], err))
            st["bytes_total"] += size
            if done % 250 == 0 or done == len(targets):
                el = time.time() - t0
                print(f"[{a.tag}] {done}/{len(targets)} 复用{st['reuse_old']} 新下{st['downloaded']} "
                      f"失败{st['failed']} {st['bytes_total']/2**30:.1f}GiB "
                      f"{done/max(el,1):.2f}首/s 剩{st.get('eta','')}", flush=True)

    st["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(WORK, f"failed_{a.tag}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "name", "reason"]); w.writerows(failed)
    json.dump(st, open(os.path.join(WORK, f"stats_{a.tag}.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(st, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
