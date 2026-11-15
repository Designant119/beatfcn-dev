"""特征层音频别名映射：zip 零改动，只产出 audio_alias.json（canonical -> members）。

分层证据（只归并，不删任何文件）：
  L1 crc_exact      组内音频条目 CRC+size 完全一致
  L2 duration_lt1pct 时长相对差 < 1%
  L3 corr>=0.95     verify_similarity.json 判定 same
unclear / diff / 未核验 → 一律不映射。
音轨键用 "crc8_size"（源包基线里的音频条目指纹），sha256 列留空待补。
"""
import json, os, collections

from paths import WORK  # 语料根目录由 paths.py 统一解析（PHIRA_WORK 优先）


def L(n):
    p = os.path.join(WORK, n)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rep = L("variant_report.json")
    base = L("baseline_zips.json")
    probe = L("probe_durations.json")
    sim = L("verify_similarity.json") or {}

    groups, no_merge = [], []
    cnt = collections.Counter()
    for g in rep["auto_merge"]:
        ids = [int(x) for x in g["members"]]
        tk = {}
        for i in ids:
            a = base[str(i)]["audio"]
            if a:
                big = max(a, key=lambda x: x[2])
                tk[i] = f"{big[1]}_{big[2]}"
        if not tk:
            no_merge.append((g["name"], ids, "no_audio")); continue
        uniq = set(tk.values())
        durs = [probe.get(str(i), {}).get("seconds") for i in ids]
        if len(uniq) == 1:
            ev = "crc_exact"
        elif all(durs) and (max(durs) - min(durs)) / min(durs) < 0.01:
            ev = "duration_lt1pct"
        else:
            key = ",".join(str(i) for i in ids)
            v = (sim.get(key) or {}).get("verdict")
            if v == "same":
                ev = "corr>=0.95"
            else:
                no_merge.append((g["name"], ids, f"sim_{v or 'unverified'}")); cnt[f"hold_{v or 'unverified'}"] += 1
                continue
        # canonical = 组内出现次数最多的音轨键（并列取体积最大）
        freq = collections.Counter(tk.values())
        size_of = {t: int(t.split("_")[1]) for t in freq}
        canon = max(freq, key=lambda t: (freq[t], size_of[t]))
        cnt[ev] += 1
        groups.append({"name": g["name"], "canonical": canon, "members": ids,
                       "member_tracks": {str(k): v for k, v in tk.items()},
                       "evidence": ev,
                       "collapsed_tracks": sorted(uniq - {canon})})

    alias = {}
    for g in groups:
        for t in g["collapsed_tracks"]:
            alias[t] = g["canonical"]

    total_tracks = len({f"{max(v['audio'], key=lambda x: x[2])[1]}_{max(v['audio'], key=lambda x: x[2])[2]}"
                        for v in base.values() if v["audio"]})
    removed = len(set(alias))
    canonical = total_tracks - removed

    out = {
        "schema_version": 1,
        "note": "特征层别名：MERT 只对 canonical 提一次，训练取特征前先映射。zip 未改动。track key = <crc8>_<bytes>；sha256 列待补（需对源包音频做一次哈希）。",
        "alias_of": alias,
        "groups": groups,
        "stats": {
            "tracks_total_crc_size": total_tracks,
            "tracks_non_canonical": removed,
            "tracks_canonical": canonical,
            "tracks_reduction_pct": round(removed / max(total_tracks, 1) * 100, 2),
            "groups_merged": len(groups),
            "groups_merged_by_evidence": dict(cnt),
            "groups_not_merged": len(no_merge),
        },
    }
    json.dump(out, open(os.path.join(WORK, "audio_alias.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({"stats": out["stats"], "not_merged": no_merge},
              open(os.path.join(WORK, "audio_alias_stats.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out["stats"], ensure_ascii=False, indent=2))
    print("not_merged sample:", json.dumps(no_merge[:5], ensure_ascii=False))


if __name__ == "__main__":
    main()
