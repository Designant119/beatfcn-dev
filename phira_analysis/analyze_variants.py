"""版本变体识别 + 合并可行性 dry-run（不动任何文件）。

规则：
- 从歌名剥掉括号/后缀里的版本标签，得到"曲名主干"
- 每条打 variant 标签；带不可共用标签的记录，即使主干、作曲、时长都相同，
  也不自动并入原版，进复核清单
- 时长门控 6%：短版/完整版分开（门控在读取 zip 的 Length 后二次校验）

输出：variant_report.json + 终端摘要，供人工复核后再决定别名合并口径。
"""
import json, os, re, unicodedata, collections, zipfile

WORK = "/home/user/Doubao/chats/38453554213740802/phira_full"
META = os.path.join(WORK, "charts_meta.json")
ZIPS = os.path.join(WORK, "zips")
DUR_TOL = 0.06

# (标签, 正则, 能否与原版共用特征)
# False = 编曲/演奏/人声/节奏/时长变了，禁止自动合并
VARIANT_RULES = [
    ("remix",        r"remix|rework|bootleg|\bedit\b|mashup|\bflip\b|re-edit", False),
    ("cover",        r"remake|re-?record(ed|ing)?|\bcover\b|翻唱|重录|self\s*cover", False),
    ("sped_up",      r"sped?\s*up|nightcore|\bfast\b|加速", False),
    ("slowed",       r"slow(ed)?\s*down|\bslowed\b|\breverb\b|减速", False),
    ("instrumental", r"instrumental|\binst\.?\b|off\s*vocal|karaoke|no\s*vocal|伴奏|無伴奏", False),
    ("short",        r"radio\s*edit|short\s*ver|tv\s*size|pv\s*ver|anime\s*ver|game\s*ver|cut\s*ver|剪辑", False),
    ("live_arrange", r"\blive\b|acoustic|piano\s*ver|guitar\s*ver|orchestra|arrange|不插电|バンド", False),
    ("remaster",     r"remaster(ed)?|re-master", True),
    ("extended",     r"extended|full\s*ver|long\s*ver|album\s*ver|完整版", True),
    ("original",     r"original\s*(mix|ver|version)|main\s*ver|原版", True),
]
MUTABLE = {tag for tag, _, ok in VARIANT_RULES if not ok}
SAFE = {tag for tag, _, ok in VARIANT_RULES if ok}

BRACKET = re.compile(r"[\(（\[【「『][^\)）\]】」』]*[\)）\]】」』]")
SEP = re.compile(r"\s*(?:/|／|、|&|＆|,|，|;|；|\+|vs\.?|versus|feat\.?|ft\.?)\s*", re.I)
KEEP = re.compile(r"[^\w぀-ヿ㐀-䶿一-鿿가-힯]")


def norm_text(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = KEEP.sub("", re.sub(r"\s+", "", s))
    return s.replace("_", "")


def comp_set(s):
    raw = unicodedata.normalize("NFKC", str(s or ""))
    return tuple(sorted(x for x in (norm_text(y) for y in SEP.split(raw)) if x))


def detect_variant(name):
    """返回 (variant_tag, safe_to_merge, 主干曲名)。主干剥掉版本括号/后缀。"""
    raw = unicodedata.normalize("NFKC", str(name or ""))
    low = raw.casefold()
    tags = []
    for vtag, pat, _ in VARIANT_RULES:
        if re.search(pat, low):
            tags.append(vtag)
    tag = tags[0] if tags else "none"
    safe = tag not in MUTABLE
    # 剥掉括号里命中版本标签的内容，再剥尾部 " - xxx"
    stem = BRACKET.sub(" ", raw)
    for vtag, pat, _ in VARIANT_RULES:
        stem = re.sub(pat, " ", stem, flags=re.I)
    stem = re.split(r"\s[-–—]\s*", stem)[0]
    return tag, safe, norm_text(stem)


def audio_duration(path):
    try:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith((".txt", ".yml", ".yaml")):
                    m = re.search(r"Length\s*:\s*([\d.]+)", z.read(n).decode("utf-8", "replace"), re.I)
                    if m:
                        return float(m.group(1))
    except Exception:
        pass
    return None


def main():
    meta = json.load(open(META))
    id2zip = {}
    for tag in ("a", "b"):
        d = os.path.join(ZIPS, tag)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".zip"):
                    m = re.match(r"(\d+)_", f)
                    if m:
                        id2zip.setdefault(int(m.group(1)), os.path.join(d, f))

    # 时长优先取 ffprobe 实测表（info.txt 的 Length 字段只覆盖 40% 的包）
    probe_path = os.path.join(WORK, "probe_durations.json")
    probe = {}
    if os.path.isfile(probe_path):
        probe = {int(k): v for k, v in json.load(open(probe_path)).items()}
        print(f"时长表 {len(probe)} 条", flush=True)

    groups = collections.defaultdict(list)
    tag_dist = collections.Counter()
    rows = []
    for r in meta:
        cid = int(r["id"])
        tag, safe, stem = detect_variant(r.get("name"))
        tag_dist[tag] += 1
        pv = probe.get(cid, {})
        dur = pv.get("seconds") if pv.get("seconds") else (
            audio_duration(id2zip[cid]) if cid in id2zip else None)
        row = dict(id=cid, name=r.get("name"), composer=r.get("composer"),
                   variant=tag, safe=safe, stem=stem, dur=dur,
                   zip=id2zip.get(cid))
        rows.append(row)
        groups[(stem, comp_set(r.get("composer")))].append(row)

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    auto_merge, blocked_variant, blocked_dur, review = [], [], [], []

    for k, members in multi.items():
        variants = {m["variant"] for m in members}
        durs = [m["dur"] for m in members if m["dur"] is not None]
        dur_bad = len(durs) >= 2 and min(durs) > 0 and (max(durs) - min(durs)) / min(durs) > DUR_TOL
        if variants & MUTABLE:
            blocked_variant.append(dict(name=k[0], members=[m["id"] for m in members],
                                       variants=sorted(variants),
                                       names=[m["name"] for m in members]))
        elif dur_bad:
            blocked_dur.append(dict(name=k[0], members=[m["id"] for m in members],
                                    durations=[round(m["dur"], 1) for m in members if m["dur"]]))
        else:
            auto_merge.append(dict(name=k[0], members=[m["id"] for m in members],
                                   variants=sorted(variants)))

    report = dict(
        total=len(rows),
        variant_distribution=dict(tag_dist),
        groups_total=len(groups),
        groups_multi=len(multi),
        auto_merge_groups=len(auto_merge),
        blocked_variant_groups=len(blocked_variant),
        blocked_duration_groups=len(blocked_dur),
        blocked_variant_records=sum(len(g["members"]) for g in blocked_variant),
        blocked_duration_records=sum(len(g["members"]) for g in blocked_dur),
        auto_merge=auto_merge, blocked_variant=blocked_variant, blocked_duration=blocked_dur,
    )
    json.dump(report, open(os.path.join(WORK, "variant_report.json"), "w"),
              ensure_ascii=False, indent=2)

    print(f"全量 {len(rows)} 条\n")
    print("版本标签分布：")
    for t, n in tag_dist.most_common():
        print(f"  {t:<14} {n:>4}")
    print(f"\n同主干多包组：{len(multi)}")
    print(f"  可自动合并：{len(auto_merge)} 组")
    print(f"  被版本标签拦住：{len(blocked_variant)} 组 / {report['blocked_variant_records']} 包")
    print(f"  被时长门控拦住：{len(blocked_dur)} 组 / {report['blocked_duration_records']} 包")
    print("\n=== 版本拦截样例（前 12）===")
    for g in blocked_variant[:12]:
        print(f"  [{','.join(g['variants'])}] {g['name'][:30]}")
        for n in g["names"][:4]:
            print(f"      · {n[:46]}")


if __name__ == "__main__":
    main()
