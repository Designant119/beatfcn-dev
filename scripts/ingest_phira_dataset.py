"""把 Phira 谱面 zip 目录建成 full database（canon + targets/*.npy + receipt）。

就是 phira_zip_row() -> prepare_chart() 这条链的批量封装，网页 04 页的
「建数据集」按钮最终调这个。zip 一律只读，产物全写到 --out 下。

  python -m scripts.ingest_phira_dataset --zips /data/phira_charts \
      --meta phira_charts_meta.json --out /data/phira_db

断点续跑：receipt 已存在的谱面会被 prepare_chart 自己校验通过后跳过。
"""
import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_meta(path):
    raw = Path(path).read_text(encoding='utf-8').strip()
    try:
        obj = json.loads(raw)
        rows = obj if isinstance(obj, list) else (obj.get('results') or obj.get('charts') or [obj])
    except json.JSONDecodeError:
        rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    return {str(r['id']): r for r in rows if isinstance(r, dict) and r.get('id') is not None}


def split_for(audio_sha256):
    """按音频指纹切 train/val（80/20）。同一音频永远落同一边，避免同曲跨集泄漏。

    meta 和 zip 里都没有 split 字段，过去的 receipt 因此是 None，采样器一条训练样本都挑不出来。
    """
    import hashlib
    if not audio_sha256:
        return 'train'
    h = int(hashlib.sha256(str(audio_sha256).encode()).hexdigest()[:8], 16)
    return 'val' if h % 5 == 0 else 'train'


def zip_id(path):
    """文件名形如 {rating}_{name}_{level}_{id}.zip 或 {id}_{name}_{level}.zip，取最后一串数字。"""
    stem = Path(path).stem
    tail = stem.rsplit('_', 1)[-1]
    if tail.isdigit():
        return tail
    head = stem.split('_', 1)[0]
    return head if head.isdigit() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zips', required=True, help='谱面 zip 目录')
    ap.add_argument('--meta', default=str(ROOT / 'phira_charts_meta.json'), help='谱面元数据 json/jsonl')
    ap.add_argument('--out', required=True, help='full database 输出根（会建 canon/ 和 targets/）')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个（冒烟用）')
    ap.add_argument('--every', type=int, default=200, help='每处理多少个打一次进度')
    args = ap.parse_args()

    from beatfcn.ingest.pipeline import phira_zip_row
    from beatfcn.train.full_database import prepare_chart, digest

    zips_dir = Path(args.zips)
    out = Path(args.out)
    canon_dir = out / 'canon' / 'phira'
    canon_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.meta)

    files = sorted(p for p in zips_dir.rglob('*.zip'))
    if args.limit:
        files = files[:args.limit]

    counts = Counter()
    failed = []
    started = time.time()
    tmpdir = tempfile.mkdtemp(prefix='ingest-')
    for i, zpath in enumerate(files, 1):
        pid = zip_id(zpath)
        cand = meta.get(str(pid))
        if cand is None:
            counts['no_meta'] += 1
            failed.append({'zip': zpath.name, 'why': 'meta 里没有这个 id'})
            continue
        try:
            row = phira_zip_row(str(out), str(zpath), cand, tmpdir)
        except Exception as e:                    # noqa: BLE001
            counts['row_error'] += 1
            failed.append({'zip': zpath.name, 'why': f'{type(e).__name__}: {e}'[:200]})
            continue
        status = row.get('status')
        if status != 'parsed':
            counts[status or 'unknown'] += 1
            failed.append({'zip': zpath.name, 'why': status or 'unknown'})
            continue
        canon = canon_dir / f'{pid}.json'
        if not canon.is_file():
            counts['canon_missing'] += 1
            failed.append({'zip': zpath.name, 'why': 'canon 没写出来'})
            continue
        if not row.get('split'):
            try:
                audio = json.loads(canon.read_text(encoding='utf-8')).get('audio_sha256')
            except Exception:                     # noqa: BLE001
                audio = row.get('audio_sha256')
            row['split'] = split_for(audio)
        try:
            receipt = prepare_chart(row, canon, out, canonical_sha256=digest(canon))
        except Exception as e:                    # noqa: BLE001
            counts['target_error'] += 1
            failed.append({'zip': zpath.name, 'why': f'{type(e).__name__}: {e}'[:200]})
            continue
        counts['ok' if receipt.get('targets_prepared') else 'prepared_partial'] += 1
        if args.every and i % args.every == 0:
            el = time.time() - started
            print(f'[{i}/{len(files)}] ok={counts["ok"]} 失败={len(failed)} '
                  f'{i / max(el, 1e-9):.2f} 个/秒', flush=True)

    report = {'zips': len(files), 'elapsed_s': round(time.time() - started, 1),
              'counts': dict(counts), 'failed': failed[:200],
              'failed_total': len(failed), 'out': str(out)}
    (out / 'ingest_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完成：{counts.get("ok", 0)} 个建好，{len(failed)} 个失败，'
          f'用时 {(time.time() - started) / 60:.1f} 分')
    print(f'报告：{out / "ingest_report.json"}')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
