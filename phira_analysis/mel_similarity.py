"""用 log-mel 频谱复核音频是否同一录音（只读，不改任何文件）。

比 RMS 包络更进一步：RMS 只看"响度起伏"，mel 看的是"每个时间点上各频带的能量分布"，
也就是音色 + 旋律的粗轮廓。同一录音不同编码 → 频谱形状几乎重合；
不同曲/不同编曲 → 帧级频谱余弦相似度明显掉下来。

输出 mel_similarity.json。
"""
import json, os, re, subprocess, tempfile, zipfile, time
import numpy as np

WORK = "/home/user/Doubao/chats/38453554213740802/phira_full"
ZIPS = os.path.join(WORK, "zips")
AUDIO_EXT = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")
SR = 16000
N_FFT = 512          # 32ms 窗
HOP = 160            # 10ms
N_MEL = 64


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


def mel_bank(sr=SR, n_fft=N_FFT, n_mels=N_MEL):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        c = max(c, l + 1)
        r = max(r, c + 1)
        for k in range(l, min(c, fb.shape[1])):
            fb[i, k] = (k - l) / (c - l)
        for k in range(c, min(r, fb.shape[1])):
            fb[i, k] = (r - k) / (r - c)
    return fb


FB = mel_bank()


def logmel(zp, cache):
    """返回 [T, N_MEL] 的 log-mel（行内 L2 归一化）。"""
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
             "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
            capture_output=True, timeout=240,
        )
        if p.returncode != 0 or not p.stdout:
            cache[zp] = None
            return None
        pcm = np.frombuffer(p.stdout, dtype=np.float32)
        n = 1 + (len(pcm) - N_FFT) // HOP
        if n < 8:
            cache[zp] = None
            return None
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
        win = np.hanning(N_FFT).astype(np.float32)
        frames = pcm[idx] * win[None, :]
        mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
        mel = mag @ FB.T
        lm = np.log(mel + 1e-6)
        lm = lm - lm.mean(axis=1, keepdims=True)
        lm = lm / (np.linalg.norm(lm, axis=1, keepdims=True) + 1e-9)
        cache[zp] = lm.astype(np.float32)
        return cache[zp]
    except Exception:
        cache[zp] = None
        return None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def mel_sim(a, b, max_shift=50):
    """帧级 log-mel 余弦相似度的中位数（±0.5s 偏移搜索取最优）。"""
    best = -1.0
    for s in range(0, max_shift + 1):
        if s and len(a) >= len(b):
            a2, b2 = a[s:], b
        elif s:
            a2, b2 = a, b[s:]
        else:
            a2, b2 = a, b
        n = min(len(a2), len(b2))
        if n < 8:
            continue
        cos = (a2[:n] * b2[:n]).sum(axis=1)
        best = max(best, float(np.median(cos)))
    return best


def main():
    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--selftest":
        c = {}
        rows = [(f"zips/{'a' if True else 'b'}", n) for n in (20626, 32167, 51050, 11745)]
        env = {}
        for _, cid in rows:
            g = __import__("glob").glob(f"zips/*/{cid}_*.zip")
            if g:
                env[cid] = logmel(g[0], c)
        def show(x, y, label):
            if env.get(x) is not None and env.get(y) is not None:
                print(f"  {label}: {mel_sim(env[x], env[y]):.4f}")
        show(20626, 32167, "同 169.6s（应高）")
        show(20626, 51050, "170s vs 298s（应低）")
        show(20626, 11745, "170s vs 340s（应低）")
        return

    targets_path = os.path.join(WORK, "mel_targets.json")
    if os.path.isfile(targets_path):
        targets = json.load(open(targets_path))
    else:
        targets = []
    m = id2zip_map()
    out_path = os.path.join(WORK, "mel_similarity.json")
    done = json.load(open(out_path)) if os.path.isfile(out_path) else {}
    cache = {}
    t0 = time.time()
    for gi, g in enumerate(targets):
        key = ",".join(map(str, g["members"]))
        if key in done:
            continue
        feats = []
        for cid in g["members"]:
            zp = m.get(cid)
            if not zp:
                continue
            f = logmel(zp, cache)
            if f is not None:
                feats.append((cid, f))
        if len(feats) < 2:
            done[key] = dict(name=g["name"], members=g["members"], min_sim=None, note="decode_failed")
            continue
        sims = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                sims.append(mel_sim(feats[i][1], feats[j][1]))
        worst = min(sims)
        done[key] = dict(name=g["name"], members=g["members"],
                         min_sim=round(worst, 4),
                         mean_sim=round(float(np.mean(sims)), 4),
                         verdict=("same" if worst >= 0.90 else "diff" if worst < 0.70 else "unclear"))
        if (gi + 1) % 10 == 0:
            json.dump(done, open(out_path, "w"), ensure_ascii=False)
            el = time.time() - t0
            print(f"  {gi+1}/{len(targets)}  {el:.0f}s", flush=True)
    json.dump(done, open(out_path, "w"), ensure_ascii=False)
    v = [x for x in done.values() if x.get("verdict")]
    print(f"完成 {len(done)} 组 / {time.time()-t0:.0f}s")
    for k in ("same", "unclear", "diff"):
        print(f"  {k}: {sum(1 for x in v if x['verdict']==k)}")


if __name__ == "__main__":
    main()
