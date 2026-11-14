"""音频同一性验证：解码后比对频谱指纹，判断两个包是不是同一份录音。

不写任何源文件，音频解压到临时文件，解码完即删。
输出 audio_match.json，供合并决策使用。

判据：把每条音轨解码成 8kHz 单声道，按 100ms 分帧算 24 个频带的对数能量，
再对帧序列做长度归一 + 余弦相似度；同一录音的不同编码应当高度一致，
不同录音/翻唱/变速会明显掉下来。
"""
import json, os, re, subprocess, tempfile, zipfile
import numpy as np

WORK = "/home/user/Doubao/chats/38453554213740802/phira_full"
ZIPS = os.path.join(WORK, "zips")
AUDIO_EXT = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")
SR = 8000
WIN = 800      # 100ms
HOP = 400      # 50ms
NBANDS = 24
NFIX = 1200    # 时间归一后的帧数


def band_edges(n_fft=1024, sr=SR, nbands=NBANDS):
    """近似 mel 分带边界。"""
    def hz2mel(f): return 2595 * np.log10(1 + f / 700.0)
    def mel2hz(m): return 700 * (10 ** (m / 2595.0) - 1)
    mel = np.linspace(hz2mel(50), hz2mel(sr / 2), nbands + 2)
    hz = mel2hz(mel)
    return np.clip((hz / (sr / 2) * (n_fft // 2)).astype(int), 0, n_fft // 2 - 1)


EDGES = band_edges()


def decode(zp):
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
            ["ffmpeg", "-v", "error", "-i", tmp, "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
            capture_output=True, timeout=120,
        )
        if p.returncode != 0 or not p.stdout:
            return None, name
        return np.frombuffer(p.stdout, dtype=np.float32), name
    except Exception:
        return None, None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def fingerprint(samples):
    """返回 [NFIX, NBANDS] 的对数频带能量，L2 归一化后用于余弦比较。"""
    if samples is None or len(samples) < SR:
        return None
    n = 1 + max(0, (len(samples) - WIN) // HOP)
    if n < 10:
        return None
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = samples[idx] * np.hanning(WIN)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=1024)) ** 2
    bands = np.stack([spec[:, EDGES[i]:max(EDGES[i] + 1, EDGES[i + 1])].mean(axis=1)
                      for i in range(NBANDS)], axis=1)
    feats = np.log1p(bands * 1e4)
    # 时间归一化到固定帧数
    src = np.linspace(0, 1, feats.shape[0])
    dst = np.linspace(0, 1, NFIX)
    feats = np.stack([np.interp(dst, src, feats[:, b]) for b in range(NBANDS)], axis=1)
    # 去均值 + L2
    feats = feats - feats.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    return feats / norm


def similarity(f1, f2):
    if f1 is None or f2 is None:
        return None
    cos = np.sum(f1 * f2, axis=1)
    return float(np.median(cos))


def main():
    import sys
    rep = json.load(open(os.path.join(WORK, "variant_report.json")))
    base = json.load(open(os.path.join(WORK, "baseline_zips.json")))
    id2zip = {}
    for tag in ("a", "b"):
        d = os.path.join(ZIPS, tag)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".zip"):
                m = re.match(r"(\d+)_", f)
                if m:
                    id2zip.setdefault(int(m.group(1)), os.path.join(d, f))

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    out_path = os.path.join(WORK, "audio_match.json")
    done = json.load(open(out_path)) if os.path.isfile(out_path) else {}

    fp_cache = {}
    def getfp(cid):
        if cid in fp_cache:
            return fp_cache[cid]
        zp = id2zip.get(cid)
        fp = None
        if zp:
            fp = fingerprint(decode(zp)[0])
        fp_cache[cid] = fp
        return fp

    groups = rep["auto_merge"]
    if limit:
        groups = groups[:limit]

    results = []
    for gi, g in enumerate(groups):
        ids = g["members"]
        key = str(g["members"][0])
        if key in done:
            continue
        # 选代表：时长最长
        probe = json.load(open(os.path.join(WORK, "probe_durations.json")))
        keeper = max(ids, key=lambda i: (probe.get(str(i), {}).get("seconds") or 0))
        fk = getfp(keeper)
        sims = []
        for cid in ids:
            if cid == keeper:
                continue
            sims.append([cid, similarity(fk, getfp(cid))])
        rec = dict(name=g["name"], keeper=keeper, sims=sims,
                   min_sim=min([s for _, s in sims if s is not None], default=None))
        results.append(rec)
        done[key] = rec
        if (gi + 1) % 50 == 0:
            json.dump(done, open(out_path, "w"), ensure_ascii=False)
            print(f"  {gi+1}/{len(groups)}", flush=True)

    json.dump(done, open(out_path, "w"), ensure_ascii=False)
    vals = [r["min_sim"] for r in results if r["min_sim"] is not None]
    if vals:
        print(f"\n完成 {len(results)} 组 | 相似度 中位 {np.median(vals):.3f} "
              f"| p10 {np.percentile(vals,10):.3f} | p90 {np.percentile(vals,90):.3f}")
        for lo, hi in [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (-1, 0.5)]:
            n = sum(1 for v in vals if lo <= v < hi)
            print(f"  {lo:.1f}~{hi:.1f}: {n} 组")


if __name__ == "__main__":
    main()
