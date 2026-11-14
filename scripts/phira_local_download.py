#!/usr/bin/env python3
r"""
Phira 全量谱面本地下载器
========================
用法（Windows 终端）:
    python phira_local_download.py --meta charts_meta.json --dest "E:\phira_charts" --workers 10

下完之后按「同一首歌」去重音频（推荐，省空间也不伤谱面）:
    python phira_local_download.py --meta charts_meta.json --dest "E:\phira_charts" --dedupe-audio

常用参数:
    --workers 10        并发数（默认 10，被限速就降到 6）
    --min-rating 0.0    只要评分高于此值的
    --min-rc 0          只要 ratingCount 不低于此值的
    --only-ranked       只要 ranked=True 的
    --dedupe-audio      生成音频别名表（特征层去重），谱面与 zip 全部原样保留
    --dedupe-dry-run    只统计能省多少，不动文件
    --levels IN,AT      只下这两个难度（写谱只写高难时用这个）
    --list-only         只出清单 selection.json，不下载

去重规则:
    歌名 name 归一化 + 作曲者 composer 拆成作者集合 -> 两首视为同一首歌
    同一首歌、同一时长簇内只认一份音轨（取体积最大的那个包），其余记为别名
    绝不修改任何 zip：音频不删、谱面 / 封面 / info 全部原样保留
    原因：仓库 ingest 按 info.yml 的 music 字段取音频，音频条目一删就会
          被判 audio_missing，整条谱面进不了训练集
"""
import argparse
import json
import os
import re
import socket
import sys
import time
import unicodedata
import zipfile
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
AUDIO_EXT = (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac", ".opus")
# 实测有包的音频没有扩展名（如 id 10256，文件名就是一串随机字符），
# 所以先认 info.yml 的 music 字段，扩展名只当兜底。
_MUSIC = re.compile(r"(?m)^\s*music\s*:\s*(\S.*?)\s*$", re.I)
CONFIG_EXT = (".json", ".yml", ".yaml", ".txt", ".md")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
_KEEP = re.compile(r"[^\w぀-ヿ㐀-䶿一-鿿가-힯]")
# 署名里的噪声：括号注释（常是罗马字/英文译名）、对战式 "vs"、各种连接符
_BRACKETS = re.compile(r"[\(（\[【「].*?[\)）\]】」]")
_SPLIT_AUTHOR = re.compile(
    r"\s*(?:/|／|、|&|＆|,|，|;|；|\+|vs\.?|versus|feat\.?|ft\.?)\s*", re.IGNORECASE
)


def force_ipv4():
    """部分网络 IPv6 不通，强制走 IPv4，否则每个请求先挂超时。"""
    orig = socket.getaddrinfo

    def gai(host, port, family=0, type=0, proto=0, flags=0):
        return orig(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = gai


def safe(s, n=40):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(s or "")).strip()
    s = re.sub(r"_+", "_", s)
    return s[:n] or "untitled"


def fname(r):
    return f"{float(r.get('rating') or 0) * 10:.1f}_{safe(r['name'])}_{safe(r.get('level', ''))}_{r['id']}.zip"


def norm_text(s):
    """全角转半角、casefold、去掉非字母数字假名汉字。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return _KEEP.sub("", re.sub(r"\s+", "", s)).replace("_", "")


def song_key(r, aggressive=False):
    """同一首歌的判重键：歌名 + 作曲者集合（顺序无关）。

    aggressive=True 时改用「字符集合」代替歌名原文，这样
    `水没都市 (Submerged City)` 与 `Submerged City_『水没都市』`
    这种中英换序的写法也能合并 —— 代价是 `NEVER`/`Nerve` 这类
    字符碰巧相同的不同曲会被误合（用户表态宁杀不放，可接受）。
    """
    name = norm_text(r.get("name") or r.get("title"))
    # 先 NFKC（全角＆→&）、去括号注释，再按连接符拆作者
    raw = _BRACKETS.sub(" ", unicodedata.normalize("NFKC", str(r.get("composer") or "")))
    composers = tuple(sorted(
        c for c in (norm_text(x) for x in _SPLIT_AUTHOR.split(raw)) if c
    ))
    if aggressive:
        return ("".join(sorted(name)), composers)
    return (name, composers)


def level_key(r):
    """难度键：类型字母 + 等级数字，例如 "IN  Lv.16" -> ("IN", 16.0)。"""
    raw = unicodedata.normalize("NFKC", str(r.get("level") or "")).upper()
    kind = re.match(r"\s*([A-Z]{1,3})", raw)
    num = re.search(r"(\d+(?:\.\d+)?)", raw)
    return ((kind.group(1) if kind else ""), float(num.group(1)) if num else None)


def pick_one_per_level(rows, aggressive=False):
    """同一首歌的每个难度只保留一个包（评分最高的那个），用于省下载流量。"""
    groups = defaultdict(list)
    for r in rows:
        groups[(song_key(r, aggressive), level_key(r))].append(r)
    kept = [
        max(v, key=lambda r: ((r.get("rating") or 0), (r.get("ratingCount") or 0)))
        for v in groups.values()
    ]
    return kept, len(rows) - len(kept)


def valid_zip(path):
    try:
        with zipfile.ZipFile(path) as z:
            if not z.namelist():
                return False
            z.testzip()
        return True
    except Exception:
        return False


def download(r, dest, tries=5):
    path = os.path.join(dest, fname(r))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        if valid_zip(path):
            return r["id"], "cached", os.path.getsize(path), None
        os.remove(path)

    err = "?"
    url = r.get("file")
    if not url:
        return r["id"], "FAIL", 0, "元数据里没有 file 字段"

    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if not data:
                raise ValueError("空文件")
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            if not valid_zip(tmp):
                os.remove(tmp)
                raise ValueError("zip 校验失败")
            os.replace(tmp, path)
            return r["id"], "ok", len(data), None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code == 429:
                time.sleep(5 + i * 5)
            elif e.code >= 500:
                time.sleep(2 + i * 3)
            else:
                break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:100]
            time.sleep(min(1 + i * 2, 10))
    return r["id"], "FAIL", 0, err


def audio_info(path, hash_audio=False):
    """包内音轨：(music 文件名, 字节数, sha256 或 None)。

    查找顺序 ① info.yml 的 music 字段 ② 音频扩展名 ③ 包内最大的非配置非图片文件。
    ②③ 是兜底 —— 有音频不带扩展名，只按后缀筛会漏。
    """
    name, size = None, 0
    try:
        with zipfile.ZipFile(path) as z:
            infos = {i.filename.lower().rsplit("/", 1)[-1]: i for i in z.infolist()}
            for entry in z.namelist():
                if not entry.lower().endswith((".yml", ".yaml")):
                    continue
                try:
                    text = z.read(entry).decode("utf-8", "replace")
                except Exception:
                    continue
                m = _MUSIC.search(text)
                if m:
                    target = m.group(1).strip().strip("\"'")
                    hit = infos.get(target.lower().rsplit("/", 1)[-1])
                    if hit is not None and hit.file_size > 0:
                        name, size = hit.filename, hit.file_size
                        break
            if size <= 0:
                hits = [i for i in z.infolist() if i.filename.lower().endswith(AUDIO_EXT)]
                if hits:
                    biggest = max(hits, key=lambda i: i.file_size)
                    name, size = biggest.filename, biggest.file_size
            if size <= 0:
                others = [i for i in z.infolist()
                          if not i.filename.lower().endswith(CONFIG_EXT + IMAGE_EXT)]
                if others:
                    biggest = max(others, key=lambda i: i.file_size)
                    name, size = biggest.filename, biggest.file_size
            if size <= 0:
                return None, 0, None
            if not hash_audio:
                return name, size, None
            import hashlib
            digest = hashlib.sha256()
            with z.open(name) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            return name, size, digest.hexdigest()
    except Exception:
        return None, 0, None


def audio_size(path):
    """包内音轨字节数；找不到返回 0。"""
    return audio_info(path)[1]


def strip_audio(path):
    """【已废弃，保留仅为兼容旧调用】绝不修改源 zip。

    实测结论：仓库 ingest 按 info.yml 的 music 字段取音频，音频条目一删，
    整条谱面会记 failed_row('audio_missing')，谱面直接进不了训练集。
    因此去重只在特征层做别名映射，见 dedupe_audio()。
    """
    raise RuntimeError(
        "strip_audio 已废弃：剥离音频会让 ingest 判定 audio_missing，"
        "请改用 --dedupe-audio 生成别名表（不改任何 zip）"
    )


def audio_duration(path):
    """包内音频时长（秒）。

    注意：info.txt 的 Length 字段只有约 40% 的包有；且正则必须行首锚定，
    否则会匹配到 info.yml 的 lineLength（谱面线长，常见值 6.0）。
    读不到返回 None，调用方按"未知"处理，不参与合并。
    """
    try:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith((".txt", ".yml", ".yaml")):
                    text = z.read(n).decode("utf-8", "replace")
                    m = re.search(r"(?m)^\s*Length\s*:\s*([\d.]+)", text, re.I)
                    if m:
                        return float(m.group(1))
    except Exception:
        pass
    return None


DUR_TOL = 0.06  # 时长相差超过 6% 视为不同版本，绝不共用音频


def dedupe_audio(rows, dest, dry_run=False, aggressive=False, hash_audio=True):
    """按 (歌名 + 作曲者) 归组，组内再做时长分簇，每簇只认一份音轨。

    同名同作曲不代表同一份音轨：实测 `Latent Kingdom` 的 9 个包里有 298 秒的
    完整版和 166 秒的短版，混用会让谱面时间轴整体错位。所以组内按时长贪心
    聚类，只有时长相差 6% 以内的才共用音轨；时长读不出来的包一律不参与去重。

    只写索引文件，绝不改动任何 zip。产物：
      audio_alias.json   谱面 id -> 组内代表谱面 id（特征层去重用）
      chart_audio.json   谱面 id -> {music, bytes, sha256}（接训练管线用）
      song_groups.json / _dedupe_report.json  明细与统计
    """
    groups = defaultdict(list)
    cache = {}
    for r in rows:
        p = os.path.join(dest, fname(r))
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            groups[song_key(r, aggressive)].append((r, p))
            if p not in cache:
                cache[p] = audio_info(p, hash_audio)

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\n按（歌名+作曲者）归组：{len(groups)} 组，其中 {len(multi)} 组包含多个包")

    total_saved = 0
    kept_list, merged_list, split_versions = [], [], []

    for key, members in groups.items():
        # --- 组内按时长分簇 -------------------------------------------------
        timed, unknown = [], []
        for m in members:
            d = audio_duration(m[1])
            if d is None:
                unknown.append(m)          # 保持 (row, path)，不要再包一层时长
            else:
                timed.append((d, m))
        timed.sort(key=lambda x: x[0])

        clusters = []
        for dur, m in timed:
            if clusters and abs(dur - clusters[-1][0]) <= DUR_TOL * max(clusters[-1][0], 1.0):
                clusters[-1][1].append(m)
            else:
                clusters.append([dur, [m]])
        if len(clusters) > 1:
            split_versions.append(dict(
                name=key[0], composer=list(key[1]),
                versions=[{"seconds": round(c[0], 1), "ids": [x[0]["id"] for x in c[1]]}
                          for c in clusters],
            ))

        # --- 每个簇内部去重：算过 sha256 就按字节精确分组 ------------------
        for dur, cluster in clusters:
            if hash_audio:
                buckets = defaultdict(list)
                for m in cluster:
                    buckets[cache.get(m[1], (None, 0, None))[2] or ("nosha:" + m[1])].append(m)
            else:
                buckets = {"unverified": cluster}
            for bucket in buckets.values():
                keeper = max(bucket, key=lambda m: cache.get(m[1], (None, 0, None))[1])
                kept_list.append(dict(
                    group_id="|".join(filter(None, [key[0], ",".join(key[1])]))[:120],
                    name=keeper[0].get("name"),
                    composer=keeper[0].get("composer"),
                    seconds=round(dur, 1),
                    members=[m[0]["id"] for m in bucket],
                    kept=keeper[0]["id"],
                    audio_bytes=cache.get(keeper[1], (None, 0, None))[1],
                ))
                for r, p in bucket:
                    if p == keeper[1]:
                        continue
                    saved = cache.get(p, (None, 0, None))[1]   # 仅统计，绝不改动源 zip
                    total_saved += saved
                    merged_list.append(dict(
                        id=r["id"], group=keeper[0]["id"], seconds=round(dur, 1),
                        audio_would_save_bytes=saved,
                        bytes_before=os.path.getsize(p),
                    ))
                if len(merged_list) and len(merged_list) % 100 == 0:
                    print(f"  已归并 {len(merged_list)} 个重复音轨，可省 {total_saved / 1e9:.2f} GB", flush=True)

        # 时长未知的包：不合并，各自保留自己的音频
        for m in unknown:
            kept_list.append(dict(
                group_id="|".join(filter(None, [key[0], ",".join(key[1])]))[:120],
                name=m[0].get("name"), composer=m[0].get("composer"),
                seconds=None, members=[m[0]["id"]], kept=m[0]["id"],
                audio_bytes=cache.get(m[1], (None, 0, None))[1],
            ))

    report = dict(
        groups=len(groups),
        groups_with_multiple=len(multi),
        alias_pairs=len(merged_list),
        verified_by_sha256=bool(hash_audio),
        audio_bytes_would_save=total_saved,
        note=("只在特征层生成别名映射，未修改任何 zip；剥离音频会让 ingest 判 audio_missing。"
              + ("合并仅发生在音频字节完全一致（sha256 相同）的包之间。"
                 if hash_audio else
                 "未算 sha256：仅按时长分簇，不同录音可能被误合，接训练前必须复核。")),
        split_versions=split_versions,
        kept=kept_list,
        merged=merged_list,
    )
    alias = {"verified_by_sha256": bool(hash_audio),
             "alias_of": {str(m["id"]): str(m["group"]) for m in merged_list},
             "groups": kept_list}
    with open(os.path.join(dest, "audio_alias.json"), "w", encoding="utf-8") as f:
        json.dump(alias, f, ensure_ascii=False, indent=2)

    # 训练管线按 audio_sha256 索引，这里把谱面 id 对到音轨指纹上
    chart_audio = {}
    for members in groups.values():
        for r, p in members:
            name, size, sha = cache.get(p, (None, 0, None))
            row = dict(music=name, bytes=size)
            if sha:
                row["sha256"] = sha
            chart_audio[str(r["id"])] = row
    with open(os.path.join(dest, "chart_audio.json"), "w", encoding="utf-8") as f:
        json.dump(chart_audio, f, ensure_ascii=False, indent=2)
    with open(os.path.join(dest, "song_groups.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(dest, "_dedupe_report.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in report.items() if k not in ("kept", "merged")},
                  f, ensure_ascii=False, indent=2)

    print(f"\n可省音频 {total_saved / 1e9:.2f} GB（仅统计，未改动任何 zip）")
    print(f"别名对数 {len(merged_list)}，涉及 {len(multi)} 组")
    print(f"别名表：{os.path.join(dest, 'audio_alias.json')}")
    print(f"谱面→音轨索引：{os.path.join(dest, 'chart_audio.json')}")
    print(f"分组明细：{os.path.join(dest, 'song_groups.json')}")
    return total_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="phira_charts_meta.json",
                    help="谱面元数据；支持 .json 数组 / .jsonl 一行一条，默认同目录 phira_charts_meta.json")
    ap.add_argument("--dest", "--out", dest="dest", required=True,
                    help="下载到哪个目录（--out 是旧文档的兼容别名）")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--min-rating", type=float, default=0.0)
    ap.add_argument("--min-rc", type=int, default=0)
    ap.add_argument("--levels", default="",
                    help="只保留这些难度，逗号分隔（如 IN,AT）。空=全要。"
                         "兼容粘连标签：IN15 / AT16 分别算 IN / AT")
    ap.add_argument("--list-only", action="store_true",
                    help="只输出符合条件的选择清单 selection.json，不下载")
    ap.add_argument("--only-ranked", action="store_true")
    ap.add_argument("--no-ipv4", action="store_true", help="不强制 IPv4（默认强制）")
    ap.add_argument("--dedupe-audio", action="store_true",
                    help="生成音频别名表 audio_alias.json + 谱面→音轨索引 chart_audio.json"
                         "（只在特征层去重，绝不修改 zip）")
    ap.add_argument("--dedupe-only", action="store_true", help="只生成索引，不下载")
    ap.add_argument("--no-hash-audio", action="store_true",
                    help="跳过音轨 sha256 计算（默认为每个音轨算，接训练管线要用）")
    ap.add_argument("--dedupe-dry-run", action="store_true", help="（已默认只读，保留兼容）")
    ap.add_argument("--one-per-song", action="store_true",
                    help="每首歌只下一个包（省流量，但会丢掉同歌的其它谱面）")
    ap.add_argument("--unique-levels", action="store_true",
                    help="下载前先按(歌名+作曲+难度)去重，每个难度只下一个包，省流量")
    ap.add_argument("--aggressive", action="store_true",
                    help="按字符集合判歌名（能抓中英换序，也会误合字符巧合的不同曲）")
    args = ap.parse_args()

    if not args.no_ipv4:
        force_ipv4()

    with open(args.meta, encoding="utf-8") as f:
        raw = f.read().strip()
    lines = [x for x in raw.splitlines() if x.strip()]
    try:                                     # 先整体当一个 JSON
        obj = json.loads(raw)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = obj.get("results") or obj.get("charts") or [obj]
        else:
            rows = []
    except json.JSONDecodeError:             # 整体不通就按 jsonl 逐行
        rows = [json.loads(x) for x in lines]
    if isinstance(rows, dict):
        rows = rows.get("results") or rows.get("charts") or []

    sel = [
        r for r in rows
        if (r.get("rating") or 0) >= args.min_rating
        and (r.get("ratingCount") or 0) >= args.min_rc
        and (not args.only_ranked or r.get("ranked"))
    ]
    raw_n = len(sel)
    if args.levels:
        allow = {x.strip().upper() for x in args.levels.split(",") if x.strip()}
        before = len(sel)
        sel = [r for r in sel if level_key(r)[0] in allow]
        print(f"难度筛选 {sorted(allow)}：{before} → {len(sel)}，排除 {before - len(sel)} 个")

    if args.one_per_song:
        keep = {}
        for r in sel:
            k = song_key(r, args.aggressive)
            if k not in keep or (r.get("rating") or 0) > (keep[k].get("rating") or 0):
                keep[k] = r
        sel = list(keep.values())
        print(f"每歌一包：{raw_n} 个谱面收敛到 {len(sel)} 首（丢掉 {raw_n - len(sel)} 个其它谱面）")

    if args.unique_levels:
        before = len(sel)
        sel, dropped = pick_one_per_level(sel, args.aggressive)
        print(f"按（歌名+作曲+难度）预过滤：{before} → {len(sel)}，砍掉 {dropped} 个重复难度包")

    os.makedirs(args.dest, exist_ok=True)

    if args.list_only:
        out = os.path.join(args.dest, "selection.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sel, f, ensure_ascii=False, indent=2)
        print(f"清单 {len(sel)} 条 -> {out}（未下载）")
        return

    if not args.dedupe_only:
        print(f"元数据 {len(rows)} 条 | 符合条件 {len(sel)} 个 | 输出到 {args.dest}")
        print(f"并发 {args.workers}，Ctrl+C 可随时中断，重跑会接着下\n", flush=True)

        failed, ok, cached, total_bytes = [], 0, 0, 0
        started = time.time()
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(download, r, args.dest): r for r in sel}
                for n, fut in enumerate(as_completed(futs), 1):
                    rid, status, size, err = fut.result()
                    if status == "ok":
                        ok += 1
                        total_bytes += size
                    elif status == "cached":
                        cached += 1
                        total_bytes += size
                    else:
                        failed.append((rid, err))
                    if n % 100 == 0 or n == len(sel):
                        elapsed = time.time() - started
                        speed = total_bytes / 1e6 / max(elapsed, 1)
                        print(
                            f"[{n}/{len(sel)}] 新下 {ok} | 已有 {cached} | 失败 {len(failed)} "
                            f"| {total_bytes / 1e9:.1f} GB | {speed:.1f} MB/s | 用时 {elapsed / 60:.0f} 分",
                            flush=True,
                        )
        except KeyboardInterrupt:
            print("\n已中断。重跑同一条命令会跳过下好的文件。")

        elapsed = time.time() - started
        report = {
            "selected": len(sel),
            "ok": ok,
            "cached": cached,
            "failed": len(failed),
            "total_bytes": total_bytes,
            "seconds": round(elapsed, 1),
            "failed_list": [{"id": i, "error": e} for i, e in failed],
        }
        with open(os.path.join(args.dest, "_download_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n完成：新下 {ok}，跳过 {cached}，失败 {len(failed)}，共 {total_bytes / 1e9:.2f} GB")
        if failed:
            print(f"失败清单在 _download_report.json，重跑一遍通常能补上")

    if args.dedupe_audio or args.dedupe_only:
        dedupe_audio(sel, args.dest, dry_run=args.dedupe_dry_run,
                     aggressive=args.aggressive, hash_audio=not args.no_hash_audio)


if __name__ == "__main__":
    main()
