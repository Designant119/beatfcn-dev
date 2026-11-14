"""从 canon 直接生成提特征用的最小 tape（*-chart.json）。

batch_extract_features.py 只从 tape 里读两样：audio_sha256 和 duration_us。
完整 tape（build_full_native_tape.py）反过来要读已提好的 features 拿时长，
所以第一次提特征时那条路是死循环 —— 这个脚本就是打破死循环的那一环。

  python -m scripts.build_feature_tape --canon /data/phira_db/canon/phira --output /data/tape_min

同一个音频出现在多个谱面时只写一份（按 audio_sha256 去重），时长取首次见到的。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--canon', required=True, help='ingest 产出的 canon 目录')
    ap.add_argument('--output', required=True, help='最小 tape 输出目录')
    args = ap.parse_args()

    canon = Path(args.canon)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    seen, dup, bad = {}, 0, 0
    for f in sorted(canon.glob('*.json')):
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except Exception:                       # noqa: BLE001
            bad += 1
            continue
        sha = d.get('audio_sha256')
        sec = d.get('audio_sec')
        if not sha or not isinstance(sec, (int, float)) or sec <= 0:
            bad += 1
            continue
        if sha in seen:
            dup += 1
            continue
        seen[sha] = int(round(float(sec) * 1_000_000))

    for sha, dur in seen.items():
        (out / f'{sha}-chart.json').write_text(
            json.dumps({'audio_sha256': sha, 'duration_us': dur}, separators=(',', ':')),
            encoding='utf-8')

    print(f'canon {len(list(canon.glob("*.json")))} 个 -> 唯一音轨 {len(seen)} 份'
          f'（重复 {dup}，字段缺失 {bad}）')
    print(f'输出：{out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
