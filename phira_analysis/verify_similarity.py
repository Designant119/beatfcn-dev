"""对"仅靠名字+作曲+时长推断"的组做真实音频相似度核验（只读）。

做法：解压音频 → ffmpeg 解成 8kHz 单声道 PCM → 按 100ms 切帧算 RMS 包络
→ 归一化后两两算 Pearson 相关。同一录音不同编码：包络几乎重合（>0.98）；
不同演奏/不同版本：明显下降。

输出 verify_similarity.json：每组的组内最小相关系数，作为能否合并的依据。
"""
import json, os, re, subprocess, sys, tempfile, zipfile, time
import numpy as np

WORK = "/home/user/Doubao/chats/38453554213740802/phira_full"
ZIPS = os.path.join(WORK, "zips")
AUDIO_EXT = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")
SR, HOP_MS = 8000, 100
HOP = SR * HOP_MS // 1000


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


def envelope(zp, cache):
    """返回 RMS 包络（float32 数组）。带缓存，避免同一包重复解码。"""
    if zp in cache:
        return cache[zp]
    tmp = None
    try:
        with zipfile.ZipFile(zp) as z:
            aud = [n for n in z.namelist() if n.lower().endswith(AUDIO_EXT)]
            if not aud:
                cache[zp] = None
                return None
            name = max(aud, key=lambda n: z.getinfo(n).file_size)
            data = z.read(name)
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(name)[1] or ".bin")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-i", tmp,
             "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
            capture_output=True, timeout=180,
        )
        if p.returncode != 0 or not p.stdout:
            cache[zp] = None
            return None
        pcm = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        frames = len(pcm) // HOP
        if frames < 8:
            cache[zp] = None
            return None
        rms = np.sqrt((pcm[: frames * HOP].reshape(frames, HOP) ** 2).mean(axis=1) + 1e-12)
        cache[zp] = rms
        return rms
    except Exception:
        cache[zp] = None
        return None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def corr(a, b, max_shift=5):
    """截断对齐 + 小幅偏移搜索，取最大相关系数。max_shift 单位=帧(100ms)。"""
    def z(x):
        x = x - x.mean()
        return x / (x.std() + 1e-9)

    best = -1.0
    n = min(len(a), len(b))
    if n < 8:
        return -1.0
    for s in range(0, max_shift + 1):
        if s and len(a) > len(b):
            a2, b2 = a[s:], b
        elif s:
            a2, b2 = a, b[s:]
        else:
            a2, b2 = a, b
        m = min(len(a2), len(b2))
        if m < 8:
            continue
        c = float((z(a2[:m]) * z(b2[:m])).mean())
        best = max(best, c)
    return best


def main():
    rep = json.load(open(os.path.join(WORK, "variant_report.json")))
    probe = json.load(open(os.path.join(WORK, "probe_durations.json")))
    base = json.load(open(os.path.join(WORK, "baseline_zips.json")))
    m = id2zip_map()

    # 只核验"CRC 不一致且时长极差 >=1%"的组
    targets = []
    for g in rep["auto_merge"]:
        ids = g["members"]
        tags = set()
        for i in ids:
            a = base[str(i)]["audio"]
            if a:
                big = max(a, key=lambda x: x[2])
                tags.add((big[1], big[2]))
        if len(tags) == 1:
            continue
        durs = [probe.get(str(i), {}).get("seconds") for i in ids]
        # 核所有 CRC 不一致的组；已完成的组由 done 缓存自动跳过
        targets.append(g)

    print(f"待核验组 {len(targets)}，涉及 {sum(len(g['members']) for g in targets)} 包", flush=True)
    out_path = os.path.join(WORK, "verify_similarity.json")
    done = json.load(open(out_path)) if os.path.isfile(out_path) else {}
    cache = {}
    t0 = time.time()

    for gi, g in enumerate(targets):
        key = ",".join(map(str, g["members"]))
        if key in done:
            continue
        envs = []
        for cid in g["members"]:
            zp = m.get(cid)
            if not zp:
                continue
            e = envelope(zp, cache)
            if e is not None:
                envs.append((cid, e))
        if len(envs) < 2:
            done[key] = dict(name=g["name"], members=g["members"], min_corr=None, note="decode_failed")
            continue
        pairs = []
        for i in range(len(envs)):
            for j in range(i + 1, len(envs)):
                pairs.append(corr(envs[i][1], envs[j][1]))
        worst = min(pairs) if pairs else None
        # 以包数最多的那个皮层决定"代表簇"（用于给出合并建议）
        done[key] = dict(
            name=g["name"], members=g["members"],
            min_corr=round(worst, 4) if worst is not None else None,
            mean_corr=round(float(np.mean(pairs)), 4) if pairs else None,
            verdict=("same" if worst is not None and worst >= 0.95 else
                     "diff" if worst is not None and worst < 0.80 else "unclear"),
        )
        if (gi + 1) % 20 == 0:
            json.dump(done, open(out_path, "w"), ensure_ascii=False)
            el = time.time() - t0
            print(f"  {gi+1}/{len(targets)}  {el:.0f}s  {(gi+1)/(el+1e-9):.2f} 组/s", flush=True)

    json.dump(done, open(out_path, "w"), ensure_ascii=False)
    v = [x for x in done.values() if x.get("verdict")]
    print(f"完成 {len(done)} 组 / 用时 {time.time()-t0:.0f}s")
    print(f"  同一录音(corr>=0.95): {sum(1 for x in v if x['verdict']=='same')}")
    print(f"  不同音频(corr<0.80): {sum(1 for x in v if x['verdict']=='diff')}")
    print(f"  存疑(0.80~0.95):     {sum(1 for x in v if x['verdict']=='unclear')}")


if __name__ == "__main__":
    main()
