"""独立校验：不信任自报数字，自己重算。

扫 zips/a、zips/b 全部 zip：
  - 按文件名前缀解析 id
  - zipfile 可打开 + namelist 非空 + testzip() CRC 全量通过
与 charts_meta.json 的 id 集合对比：缺失 / 坏包 / 多余。
输出 verify.json。
"""
import json, os, re, sys, time, zipfile

from paths import WORK  # 语料根目录由 paths.py 统一解析（PHIRA_WORK 优先）
ZIPS = os.path.join(WORK, "zips")
META = os.path.join(WORK, "charts_meta.json")
AUDIO_EXT = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")


def safe(s, n=36):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(s)).strip()
    s = re.sub(r"_+", "_", s)
    return s[:n] or "untitled"


def expected_fname(r):
    return f"{r['id']}_{safe(r['name'])}_{safe(r.get('level', ''), 8)}.zip"


def main():
    meta = json.load(open(META))
    meta_by_id = {int(r["id"]): r for r in meta}
    meta_ids = set(meta_by_id)
    print(f"meta ids = {len(meta_ids)}", flush=True)

    valid = {}          # id -> {tag, file, entries, has_audio, size}
    bad = []            # {id, file, tag, reason}
    unparsed = []       # 文件前缀解析不出 id
    name_mismatch = []  # id 命中但文件名与预期不符

    t0 = time.time()
    n = 0
    for tag in ("a", "b"):
        d = os.path.join(ZIPS, tag)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                continue
            m = re.match(r"(\d+)_", f)
            if not m:
                if f.endswith(".zip"):
                    unparsed.append({"tag": tag, "file": f})
                continue
            cid = int(m.group(1))
            n += 1
            reason = None
            names, has_audio = [], False
            try:
                if os.path.getsize(p) < 100:
                    raise ValueError("file <100 bytes")
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
                    if not names:
                        raise ValueError("namelist empty")
                    bad_entry = z.testzip()
                    if bad_entry is not None:
                        raise ValueError(f"CRC mismatch on entry {bad_entry}")
                    has_audio = any(x.lower().endswith(AUDIO_EXT) for x in names)
            except Exception as e:
                reason = f"{type(e).__name__}: {str(e)[:120]}"
            if reason:
                bad.append({"id": cid, "tag": tag, "file": f, "reason": reason})
                continue
            valid[cid] = {"tag": tag, "file": f, "entries": len(names),
                          "has_audio": has_audio, "size": os.path.getsize(p)}
            r = meta_by_id.get(cid)
            if r and expected_fname(r) != f:
                name_mismatch.append({"id": cid, "tag": tag, "actual": f,
                                      "expected": expected_fname(r)})
            if n % 1000 == 0:
                print(f"  scanned {n}  elapsed {time.time()-t0:.0f}s", flush=True)

    disk_ids = set(valid) | {b["id"] for b in bad}
    missing = sorted(meta_ids - disk_ids)
    extra = sorted(disk_ids - meta_ids)

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta_total": len(meta_ids),
        "files_on_disk": n,
        "valid_count": len(valid),
        "missing_count": len(missing),
        "missing_ids": missing,
        "bad_count": len(bad),
        "bad_packages": bad,
        "extra_count": len(extra),
        "extra_ids": extra[:100],
        "unparsed_zip_count": len(unparsed),
        "unparsed_zips": unparsed[:100],
        "name_mismatch_count": len(name_mismatch),
        "name_mismatch": name_mismatch[:100],
        "by_tag": {
            "a": sum(1 for v in valid.values() if v["tag"] == "a"),
            "b": sum(1 for v in valid.values() if v["tag"] == "b"),
        },
        "has_audio_count": sum(1 for v in valid.values() if v["has_audio"]),
        "no_audio_count": sum(1 for v in valid.values() if not v["has_audio"]),
        "elapsed_sec": round(time.time() - t0, 1),
        "pass": len(missing) == 0 and len(bad) == 0 and len(valid) == len(meta_ids),
    }
    json.dump(result, open(os.path.join(WORK, "verify.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("missing_ids", "bad_packages", "extra_ids",
                                   "unparsed_zips", "name_mismatch")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
