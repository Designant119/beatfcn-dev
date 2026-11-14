"""三层证据核验 → 三档落地清单（只读，不碰 zips，不建 stripped/）。

分层口径：
  L1：组内各包最大音频条目 CRC+size 完全一致（字节级重复）。
  L2：CRC 不一致，但组内时长极差 <1%（未送音频相关性核验，靠名字+作曲+时长证据）。
  L3：CRC 不一致且时长极差 ≥1%，已进 verify_similarity.json 做包络相关性核验：
       verdict=same(>=0.95) / diff(<0.80) / unclear(0.80~0.95) / decode_failed。

三档：
  merge_ok      = L1 + L2 + L3(same)
  merge_blocked= 外部版本拦截组 + L3(diff)
  merge_review  = L3(unclear) + decode_failed（人工复核，禁止自动合并）
"""
import csv, json, os, zipfile, collections, time

WORK = "/home/user/Doubao/chats/38453554213740802/phira_full"
PLAN = os.path.join(WORK, "plan")
AUDIO_EXT = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")


def load(p):
    return json.load(open(os.path.join(WORK, p), encoding="utf-8"))


def main():
    rep = load("variant_report.json")
    probe = {int(k): v for k, v in load("probe_durations.json").items()}
    base = load("baseline_zips.json")
    sim_raw = load("verify_similarity.json")
    # 脚本里 key 用报告原始顺序拼接，这里按成员集合建索引，避免顺序不一致
    sim = {tuple(sorted(v["members"])): v for v in sim_raw.values()}
    dry = load("dry_run_report.json")

    shard_a = {int(x) for x in open(os.path.join(WORK, "shard_a_ids.txt")) if x.strip()}

    def tag_of(cid):
        return "a" if cid in shard_a else "b"

    # dry_run_report 按 members 元组索引，取已算好的压缩体积
    dry_by_members = {}
    for g in dry["auto_groups"]:
        dry_by_members[tuple(sorted(g["members"]))] = g

    csz_cache = {}

    def audio_csz(cid, fn):
        """兜底：只读 central directory 取音频压缩大小。"""
        if cid in csz_cache:
            return csz_cache[cid].get(fn, 0)
        b = base[str(cid)]
        p = os.path.join(WORK, "zips", tag_of(cid), b["zip"])
        out = {}
        try:
            with zipfile.ZipFile(p) as z:
                for i in z.infolist():
                    if i.filename.lower().endswith(AUDIO_EXT):
                        out[i.filename] = i.compress_size
        except Exception:
            pass
        csz_cache[cid] = out
        return out.get(fn, 0)

    def keeper_of(ids):
        cands = []
        for cid in ids:
            aud = base[str(cid)]["audio"]
            if aud:
                fn, crc, size = max(aud, key=lambda a: a[2])
                cands.append({"id": cid, "file": fn, "crc": crc, "size": size})
        if not cands:
            return None, []
        kept = max(cands, key=lambda c: (c["size"], -c["id"]))
        return kept, [c for c in cands if c["id"] != kept["id"]]

    auto = rep["auto_merge"]
    blocked_var = rep["blocked_variant"]

    ok_rows, blocked_rows, review_rows = [], [], []
    tier_stat = collections.defaultdict(lambda: {"groups": 0, "packages": 0, "stripped": 0,
                                                 "saved_csz": 0, "saved_raw": 0})
    verdict_tally = collections.Counter()
    verdict_pkgs = collections.Counter()

    for g in auto:
        ids = sorted(g["members"])
        sigs = set()
        for i in ids:
            a = base[str(i)]["audio"]
            if a:
                big = max(a, key=lambda x: x[2])
                sigs.add((big[1], big[2]))
        durs = [probe[i]["seconds"] for i in ids]
        spread = (max(durs) - min(durs)) / min(durs) if min(durs) > 0 else 1.0
        tier = "L1" if len(sigs) == 1 else ("L2" if spread < 0.01 else "L3")

        kept, stripped = keeper_of(ids)
        dryg = dry_by_members.get(tuple(ids), {})
        dry_strip = {s["id"]: s for s in dryg.get("stripped", [])}

        # L3 相关性结论
        entry = sim.get(tuple(ids))
        verdict = (entry or {}).get("verdict") or ("decode_failed" if entry else None)
        min_corr = (entry or {}).get("min_corr")
        if tier == "L3":
            verdict_tally[verdict or "missing"] += 1
            verdict_pkgs[verdict or "missing"] += len(ids)

        dur_map = {i: probe[i]["seconds"] for i in ids}
        dur_str = ";".join(f"{i}:{dur_map[i]:.3f}" for i in ids)
        crc_map = {}
        for i in ids:
            a = base[str(i)]["audio"]
            if a:
                big = max(a, key=lambda x: x[2])
                crc_map[i] = big[1]
        crc_str = ";".join(f"{i}:{crc_map.get(i,'')}" for i in ids)

        def delta_str(kept_id):
            if kept_id is None:
                return ""
            kd = dur_map[kept_id]
            out = []
            for i in ids:
                ms = round((dur_map[i] - kd) * 1000)
                out.append(f"{i}:{ms:+d}")
            return ";".join(out)

        # 本候选组剥离预算（压缩体积 / 原始体积）
        grp_csz, grp_raw = 0, 0
        for c in stripped:
            d = dry_strip.get(c["id"])
            csz = d["audio_compressed_size"] if d else audio_csz(c["id"], c["file"])
            grp_csz += csz
            grp_raw += c["size"]

        common = {
            "group_name": g["name"],
            "members": ";".join(map(str, ids)),
            "member_durations": dur_str,
            "member_audio_crc": crc_str,
            "min_corr": "" if min_corr is None else min_corr,
        }

        if tier == "L3" and verdict == "diff":
            blocked_rows.append({"group_name": g["name"], "block_reason": "diff",
                                  "variants": ";".join(g.get("variants", [])),
                                  "members": ";".join(map(str, ids)),
                                  "durations": dur_str, "min_corr": common["min_corr"]})
        elif tier == "L3" and verdict in ("unclear", "decode_failed", None, "missing"):
            review_rows.append({"group_name": g["name"], "block_reason": "unclear",
                                "variants": ";".join(g.get("variants", [])),
                                "members": ";".join(map(str, ids)),
                                "durations": dur_str, "min_corr": common["min_corr"],
                                "review_note": "待人工复核，禁止自动合并"})
        else:
            # L1 / L2 / L3-same → 可合并
            keeper_id = kept["id"] if kept else ""
            ok_rows.append({
                **common, "tier": tier, "keeper_id": keeper_id,
                "keeper_duration": f"{dur_map[keeper_id]:.3f}" if keeper_id else "",
                "saved_bytes": grp_csz,
                "duration_delta_ms": delta_str(keeper_id),
            })
            st = tier_stat[tier]
            st["groups"] += 1
            st["packages"] += len(ids)
            st["stripped"] += len(stripped)
            st["saved_csz"] += grp_csz
            st["saved_raw"] += grp_raw

    # 外部版本拦截组（不在 1121 候选闭环内，单列）
    var_rows = []
    for g in blocked_var:
        ids = sorted(g["members"])
        durs = [probe[i]["seconds"] for i in ids if i in probe]
        var_rows.append({
            "group_name": g["name"],
            "block_reason": "variant",
            "variants": ";".join(g["variants"]),
            "members": ";".join(map(str, ids)),
            "durations": ";".join(f"{i}:{probe[i]['seconds']:.3f}" for i in ids if i in probe),
            "min_corr": "",
        })
    blocked_rows.extend(var_rows)

    # 写 CSV
    os.makedirs(PLAN, exist_ok=True)
    ok_cols = ["group_name", "members", "member_durations", "member_audio_crc", "min_corr",
               "tier", "keeper_id", "keeper_duration", "saved_bytes", "duration_delta_ms"]
    bl_cols = ["group_name", "block_reason", "variants", "members", "durations", "min_corr"]
    rv_cols = bl_cols + ["review_note"]

    def write(name, cols, rows):
        with open(os.path.join(PLAN, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write("merge_ok.csv", ok_cols, ok_rows)
    write("merge_blocked.csv", bl_cols, blocked_rows)
    write("merge_review.csv", rv_cols, review_rows)

    # 闭环统计
    def agg(rows):
        return len(rows), sum(len(r["members"].split(";")) for r in rows)

    ok_g, ok_p = agg(ok_rows)
    bl_g, bl_p = agg(blocked_rows)
    rv_g, rv_p = agg(review_rows)
    var_g, var_p = len(var_rows), sum(len(r["members"].split(";")) for r in var_rows)
    # 候选 1121 闭环：ok + L3-diff + review（不含外部 variant）
    l3diff_rows = [r for r in blocked_rows if r["block_reason"] == "diff"]
    loop_g = ok_g + len(l3diff_rows) + rv_g
    loop_p = ok_p + sum(len(r["members"].split(";")) for r in l3diff_rows) + rv_p

    ok_saved_csz = sum(int(r["saved_bytes"]) for r in ok_rows)
    ok_saved_raw = sum(t["saved_raw"] for t in tier_stat.values())

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"auto_groups": len(auto), "auto_packages": sum(len(g["members"]) for g in auto),
                   "blocked_variant_groups": var_g, "blocked_variant_packages": var_p},
        "verdict_tally_groups": dict(verdict_tally),
        "verdict_tally_packages": dict(verdict_pkgs),
        "merge_ok": {"groups": ok_g, "packages": ok_p,
                     "stripped_packages": sum(t["stripped"] for t in tier_stat.values()),
                     "saved_compressed_bytes": ok_saved_csz,
                     "saved_raw_bytes": ok_saved_raw,
                     "saved_compressed_gib": round(ok_saved_csz / 2**30, 3),
                     "saved_compressed_gb": round(ok_saved_csz / 1e9, 3),
                     "by_tier": {k: dict(v) for k, v in tier_stat.items()}},
        "merge_blocked": {"groups": bl_g, "packages": bl_p,
                          "variant_groups": var_g, "variant_packages": var_p,
                          "l3_diff_groups": len(l3diff_rows),
                          "l3_diff_packages": sum(len(r["members"].split(";")) for r in l3diff_rows)},
        "merge_review": {"groups": rv_g, "packages": rv_p},
        "closed_loop": {
            "candidate_groups": 1121, "candidate_packages": 3332,
            "loop_groups": loop_g, "loop_packages": loop_p,
            "loop_groups_match": loop_g == 1121, "loop_packages_match": loop_p == 3332,
        },
    }
    json.dump(summary, open(os.path.join(PLAN, "plan_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
