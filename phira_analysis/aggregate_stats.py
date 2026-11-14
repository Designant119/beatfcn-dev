"""汇总统计：读下载分片统计 + 元数据 + 去重统计，产出 summary.json / buckets.csv。"""
import collections, json, os, statistics, zipfile, re

WORK = "/home/user/Doubao/chats/38453554213740802/phira_full"
ZIPS = os.path.join(WORK, "zips")


def load(p, default=None):
    try:
        return json.load(open(os.path.join(WORK, p)))
    except Exception:
        return default


def main():
    meta = load("charts_meta.json", [])
    meta_stats = load("meta_stats.json", {})
    sa = load("stats_a.json", {})
    sb = load("stats_b.json", {})
    dedup = load("dedup_stats.json", {})
    groups = load("song_groups.json", [])

    # 实际落地：扫 zips/a,b
    landed = {}
    bad = []
    audio_cnt = collections.Counter()
    for tag in ("a", "b"):
        d = os.path.join(ZIPS, tag)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".zip"):
                continue
            m = re.match(r"(\d+)_", f)
            if not m:
                continue
            cid = int(m.group(1))
            p = os.path.join(d, f)
            try:
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
                    if not names or z.testzip() is not None:
                        raise ValueError("bad zip")
                    has_audio = any(n.lower().endswith((".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus"))
                                    for n in names)
                    audio_cnt["with_audio" if has_audio else "ref_only"] += 1
                landed[cid] = os.path.getsize(p)
            except Exception as e:
                bad.append({"id": cid, "path": p, "reason": str(e)[:60]})

    total_bytes = sum(landed.values())
    sizes = sorted(landed.values())

    # 分档
    rating_b = collections.Counter()
    diff_b = collections.Counter()
    for r in meta:
        rt = r.get("rating") or 0
        if rt < 0.5: rating_b["<0.5"] += 1
        elif rt < 0.7: rating_b["0.5-0.7"] += 1
        elif rt < 0.8: rating_b["0.7-0.8"] += 1
        elif rt < 0.9: rating_b["0.8-0.9"] += 1
        elif rt < 0.95: rating_b["0.9-0.95"] += 1
        else: rating_b["0.95-1.0"] += 1
        df = float(r.get("difficulty") or 0)
        key = "<10" if df < 10 else (f"{int(df)}" if df < 17 else "17+")
        diff_b[key] += 1

    group_sizes = collections.Counter(min(len(g.get("members") or []), 3) for g in groups if g.get("kept"))
    multi = sum(1 for g in groups if len(g.get("members") or []) > 1)

    summary = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta_stats,
        "zips": {
            "expected": len(meta),
            "landed_valid": len(landed),
            "invalid": bad[:50],
            "invalid_count": len(bad),
            "total_bytes": total_bytes,
            "total_gib": round(total_bytes / 2**30, 2),
            "avg_bytes_per_chart": round(total_bytes / max(len(landed), 1)),
            "median_bytes": int(statistics.median(sizes)) if sizes else 0,
            "with_audio": audio_cnt["with_audio"],
            "audio_ref_only": audio_cnt["ref_only"],
        },
        "shards": {"a": sa, "b": sb},
        "rating_buckets": dict(rating_b),
        "difficulty_buckets": dict(diff_b),
        "audio_dedup": dedup,
        "song_group_dist": {"1": group_sizes[1], "2": group_sizes[2], "3plus": group_sizes[3],
                            "groups_with_multiple_charts": multi},
    }
    json.dump(summary, open(os.path.join(WORK, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
