"""只读补时长：给缺 Length 字段的包用 ffprobe 实测音频秒数。

绝不写任何 zip；结果落 probe_durations.json，供变体分析做时长门控。
用法：python probe_durations.py [--all]
"""
import json, os, re, subprocess, tempfile, zipfile, sys, time

WORK = "/home/user/Doubao/chats/38453554213740802/phira_full"
META = os.path.join(WORK, "charts_meta.json")
ZIPS = os.path.join(WORK, "zips")
OUT = os.path.join(WORK, "probe_durations.json")
AUDIO_EXT = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")


def id2zip_map():
    m = {}
    for tag in ("a", "b"):
        d = os.path.join(ZIPS, tag)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".zip"):
                mm = re.match(r"(\d+)_", f)
                if mm:
                    m.setdefault(int(mm.group(1)), os.path.join(d, f))
    return m


def info_length(zp):
    """info.txt/info.yml 里真正的 Length 字段（行首锚定，避开 lineLength）。"""
    try:
        with zipfile.ZipFile(zp) as z:
            text = "".join(
                z.read(n).decode("utf-8", "replace")
                for n in z.namelist()
                if n.lower().endswith((".txt", ".yml", ".yaml"))
            )
        m = re.search(r"(?m)^\s*Length\s*:\s*([\d.]+)", text, re.I)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def probe(zp):
    """ffprobe 读容器时长。音频解压到临时文件再测（pipe 不可 seek，mp3/ogg 测不出）。"""
    tmp = None
    try:
        with zipfile.ZipFile(zp) as z:
            aud = [n for n in z.namelist() if n.lower().endswith(AUDIO_EXT)]
            if not aud:
                return None, None
            name = max(aud, key=lambda n: z.getinfo(n).file_size)
            data = z.read(name)
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(name)[1] or ".bin")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        p = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", "-i", tmp,
             "-show_entries", "format=duration", "-of", "csv=p=0"],
            capture_output=True, timeout=60,
        )
        if p.returncode != 0:
            return None, name
        return float(p.stdout.decode().strip()), name
    except Exception:
        return None, None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def main():
    targets_only_multi = "--all" not in sys.argv
    m = id2zip_map()
    meta = {int(r["id"]): r for r in json.load(open(META))}

    if targets_only_multi:
        rep = json.load(open(os.path.join(WORK, "variant_report.json")))
        ids = set()
        for key in ("auto_merge", "blocked_variant", "blocked_duration"):
            for g in rep[key]:
                ids.update(g["members"])
    else:
        ids = set(m)

    existing = {}
    if os.path.isfile(OUT):
        existing = json.load(open(OUT))

    todo = []
    for cid in sorted(ids):
        if cid in existing and existing[cid].get("seconds"):
            continue
        zp = m.get(cid)
        if not zp:
            continue
        L = info_length(zp)
        if L:
            existing[cid] = dict(seconds=L, source="info_length")
        else:
            todo.append(cid)

    print(f"目标 {len(ids)} 包 | 已有 Length {len(ids)-len(todo)} | 需 ffprobe {len(todo)}", flush=True)
    t0 = time.time()
    fail = 0
    for i, cid in enumerate(todo):
        sec, name = probe(m[cid])
        if sec:
            existing[cid] = dict(seconds=round(sec, 2), source="ffprobe", audio=name)
        else:
            fail += 1
            existing[cid] = dict(seconds=None, source="probe_failed")
        if (i + 1) % 200 == 0:
            json.dump(existing, open(OUT, "w"), ensure_ascii=False)
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)}  失败 {fail}  {el:.0f}s  {len(todo)/(el+1e-9):.2f} 包/s", flush=True)
    json.dump(existing, open(OUT, "w"), ensure_ascii=False)
    okn = sum(1 for v in existing.values() if v.get("seconds"))
    print(f"完成：{okn}/{len(existing)} 有可用时长，探测失败 {fail}，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
