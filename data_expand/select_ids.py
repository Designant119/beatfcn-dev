#!/usr/bin/env python3
"""Pick an expansion set from phira_all_charts.jsonl.
Rules: 只要谱面有文件就收（全量口径，不再按评分/票数/reviewed 过滤）；
同一首歌最多 2 张、优先难度高的；已有 id 计入每首歌上限。
Usage: select_ids.py <jsonl> <existing_ids.txt> <target_total> <out_ids.txt>"""
import os
import sys, json, math, re, collections


def main():
    jl, ex, target, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    R = [json.loads(l) for l in open(jl, encoding='utf-8') if l.strip()]
    have = set(open(ex).read().split())

    def div(r):
        for t in r.get('tags') or []:
            if t in ('regular', 'plain', 'troll', 'visual'): return t
        return ''

    def key(r): return re.sub(r'[\W_]+', '', (r.get('name') or '').lower())

    per = collections.Counter(key(r) for r in R if str(r['id']) in have)
    seen_file = set(r.get('file') for r in R if str(r['id']) in have)

    def ok(r):
        # 全量口径：只要谱面文件可用就收，不再按评分/票数/reviewed 过滤。
        return bool(r.get('file'))

    def q(r):
        return (1 if (r.get('ranked') or r.get('stable')) else 0,
                round((r.get('rating') or 0) * math.log1p(r.get('ratingCount') or 0), 1),
                r.get('difficulty') or 0)

    C = sorted((r for r in R if str(r['id']) not in have and ok(r)), key=q, reverse=True)
    # group by song: take best-quality first, then within song prefer harder
    bysong = collections.defaultdict(list)
    for r in C: bysong[key(r)].append(r)
    pick = []
    existing = len(set(open(ex).read().split())) if os.path.exists(ex) else 0
    for r in C:
        if len(pick) >= target - existing: break
        k = key(r)
        if per[k] >= 2 or r['file'] in seen_file or not k: continue
        cands = [c for c in bysong[k] if str(c['id']) not in have and c['file'] not in seen_file]
        top = sorted(cands[:4], key=lambda c: c.get('difficulty') or 0, reverse=True)[0]
        pick.append(top); per[k] += 1; seen_file.add(top['file']); have.add(str(top['id']))
    open(out, 'w').write('\n'.join(str(r['id']) for r in pick) + '\n')
    print('candidates', len(C), 'songs', len(bysong), 'picked', len(pick))
    print(collections.Counter((bool(r.get('ranked') or r.get('stable')), div(r)) for r in pick))


if __name__ == '__main__':
    main()
