"""从 tape 汇总出请求级 pool，并用训练同款采样器冻结一份 exposure。

链路位置：build_full_native_tape.py 产出的是每个谱面一个 chart.json（内含 requests，
按 8 秒窗口切分）；训练侧的 LocalRequestSampler 吃的是「每请求一行」的表。
这层把前者展平成后者，再照 run_local_learning_curve 的策略采 steps 步，落成 exposure。

  python -m scripts.build_phira_exposure \
      --tape-dir /data/tape --seed 20261108 --steps 2400 \
      --output-pool /data/evolve/pool.json \
      --output-exposure /data/evolve/exposure-20261108-2400u.json

pool 里每行的字段与 tape 一致（controls/tokens/bands/split/mode/component/chart_id/stratum），
只额外补一个稳定 request_key，方便事后追溯。
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.train.local_sampling import LocalRequestSampler


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def condition_view(row, rng):
    """与 scripts/run_local_learning_curve.py 逐字一致的掩码策略，单独抄一份避免牵连 torch。"""
    import numpy as np
    value = np.asarray(row['controls'], np.float32).copy()
    known = np.flatnonzero(value[:, 2])
    choice = rng.random()
    if choice < .1:
        keep, kind = [], 'none'
    elif choice < .6:
        keep, kind = rng.sample(list(known), 1), 'single'
    elif choice < .8:
        keep, kind = rng.sample(list(known), min(len(known), rng.randint(2, 4))), 'subset'
    else:
        keep, kind = known, 'all'
    hide = [i for i in range(len(value)) if i not in keep]
    value[hide, :3] = 0
    return value, kind


def load_pool(tape_dir, limit_charts=0):
    records, charts = [], 0
    for f in sorted(tape_dir.glob('*-chart.json')):
        c = json.loads(f.read_text(encoding='utf-8'))
        rows = c.get('requests') or []
        if not rows:
            continue
        charts += 1
        for i, r in enumerate(rows):
            records.append(dict(
                split=c['split'], mode=c['mode'], component=c['component'],
                chart_id=c['chart_id'], stratum=c.get('stratum'), tokens=int(r['tokens']),
                controls=r['controls'], bands=r.get('bands'),
                event_count=int(r.get('event_count', 0)),
                request_key=f"{c['chart_id']}#{i}",
                condition_source=r.get('condition_source'),
            ))
        if limit_charts and charts >= limit_charts:
            break
    return records, charts


def sample(pool, seed, steps):
    sampler = LocalRequestSampler(pool, seed=seed)
    views = random.Random(seed)
    out = []
    for step in range(1, steps + 1):
        ids = sampler.next_batch()
        specified = []
        for i in ids:
            _, kind = condition_view(pool[i], views)
            specified.append(kind)
        out.append(dict(effective_update=step, indices=ids,
                        specified_condition_fields=specified))
    return out, sampler.pool_identity


def summary(pool, steps, seed):
    counts = {k: Counter() for k in ('mode', 'song', 'chart', 'stratum', 'kind')}
    tokens = 0
    for step in steps:
        for i, kind in zip(step['indices'], step['specified_condition_fields'], strict=True):
            r = pool[i]
            if r['split'] != 'train':
                raise ValueError(f"exposure 抽到了非训练谱面：{r['chart_id']}")
            tokens += r['tokens']
            counts['mode'][r['mode']] += 1
            counts['song'][str(r['component'])] += 1
            counts['chart'][r['chart_id']] += 1
            counts['stratum'][str(r['stratum'])] += 1
            counts['kind'][kind] += 1
    return dict(seed=seed, steps=len(steps), requests=sum(sum(c.values()) for c in counts.values()),
                valid_tokens=tokens, independent_songs=len(counts['song']),
                charts=len(counts['chart']),
                by_mode=dict(counts['mode']), by_stratum=dict(counts['stratum']),
                by_condition_kind=dict(counts['kind']))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tape-dir', required=True, help='含 *-chart.json 的 tape 目录')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, default=2400, help='有效更新步数')
    ap.add_argument('--output-pool', required=True)
    ap.add_argument('--output-exposure', required=True)
    ap.add_argument('--output-report', default='')
    ap.add_argument('--limit-charts', type=int, default=0, help='只取前 N 个谱面（试跑用）')
    args = ap.parse_args()

    pool, charts = load_pool(Path(args.tape_dir), args.limit_charts)
    if not pool:
        raise SystemExit(f'{args.tape_dir} 下没有可用的 *-chart.json')
    modes = Counter(r['mode'] for r in pool if r['split'] == 'train')
    print(f'展平：{charts} 个谱面 -> {len(pool)} 条请求；训练侧 mode 分布 {dict(modes)}')

    steps, identity = sample(pool, args.seed, args.steps)
    stats = summary(pool, steps, args.seed)
    pool_out = dict(id='phira-pool-v1', pool_identity=identity, charts=charts,
                    records=pool, summary=dict(requests=len(pool),
                    valid_tokens=sum(r['tokens'] for r in pool)))
    exposure = dict(id=f'exposure-{args.seed}-{args.steps}u-v1',
                    pool_identity=identity, seed=args.seed,
                    request_pool=args.output_pool, steps=steps, summary=stats)

    for path, obj in ((args.output_pool, pool_out), (args.output_exposure, exposure)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    if args.output_report:
        rep = dict(id='phira-exposure-build-v1', complete=True, pool_identity=identity,
                   charts=charts, summary=stats,
                   source_sha256={args.output_pool: sha(pool_out),
                                  args.output_exposure: sha(exposure)})
        Path(args.output_report).write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                            encoding='utf-8')

    print(json.dumps(dict(pool=len(pool), steps=stats['steps'], tokens=stats['valid_tokens'],
                          songs=stats['independent_songs'], charts=stats['charts']), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
