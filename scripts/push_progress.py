#!/usr/bin/env python3
"""把训练进度推到飞书群机器人，手机端直接看；不需要公网、不需要装客户端。

用法（在训练机仓库目录）：
  python3 scripts/push_progress.py --out /data/evolve/C1-depth12 \
      --webhook https://open.feishu.cn/open-apis/bot/v2/hook/xxxx

只读 progress.json 和 train.log，不改任何训练状态；断了就重试，不退出。
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

WEBHOOK = 'BEATFCN_FEISHU_HOOK'
STALL_S = 300          # 超过这么久没写进度，就告警
HEARTBEAT_S = 600      # 没变化也每 10 分钟报一次「还活着」


def read_state(out_dir):
    try:
        return json.loads((Path(out_dir) / 'progress.json').read_text())
    except Exception:
        return None


def last_json_line(log_path):
    try:
        tail = Path(log_path).read_text(errors='ignore').splitlines()[-400:]
    except Exception:
        return None
    for line in reversed(tail):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if isinstance(d, dict) and ('loss' in d or 'updates' in d):
            return d
    return None


def hm(seconds):
    if seconds is None:
        return '—'
    seconds = int(max(seconds, 0))
    if seconds < 90:
        return '%d 秒' % seconds
    if seconds < 5400:
        return '%d 分钟' % round(seconds / 60)
    return '%d 小时 %d 分' % (seconds // 3600, (seconds % 3600) // 60)


def num(v, digits=0):
    if v is None:
        return '—'
    try:
        s = '%.*f' % (digits, float(v))
    except (TypeError, ValueError):
        return str(v)
    if '.' in s:                       # 只在小数位内部去零，整数位的 0 必须留着
        s = s.rstrip('0').rstrip('.')
    return s


def lrn(v):
    """学习率用科学记数法，3e-05 比 0.00003 好认。"""
    if v is None:
        return '—'
    try:
        a = float(v)
    except (TypeError, ValueError):
        return str(v)
    return '%.1e' % a if a < 0.01 else num(a, 5)


def compose(name, st, lg, now):
    """拼一条给人看的中文消息；返回 (标题, 正文, 是否异常)."""
    phase = (st or {}).get('phase') or '?'
    updates = (st or {}).get('updates') or 0
    target = (st or {}).get('target_updates')
    horizon = (st or {}).get('horizon_updates')
    loss = (st or {}).get('loss') if st else None
    if lg and lg.get('loss') is not None:
        loss = lg['loss']
    lr = (st or {}).get('lr') if st and st.get('lr') is not None else (lg or {}).get('lr')
    rate = (st or {}).get('recent_updates_per_s') or (st or {}).get('rate_updates_per_s')
    eta = (st or {}).get('eta_s')
    gpu = (st or {}).get('gpu_peak_mib')
    idle = (st or {}).get('since_last_update_s')
    deadline = (st or {}).get('deadline_unix')

    if not st:
        head = '%s · 还没开跑（这目录里没有 progress.json）' % name
    elif phase in ('failed',):
        head = '%s · 训练挂了' % name
    elif phase == 'done':
        head = '%s · 训练跑完了' % name
    elif phase == 'deadline':
        head = '%s · 到点收工' % name
    elif phase in ('stuck_past_deadline',):
        head = '%s · 超过截止还在动' % name
    elif idle is not None and idle > STALL_S:
        head = '%s · 停住不动了' % name
    else:
        head = '%s · %s' % (name, '训练中' if phase == 'train' else phase)

    bad = phase in ('failed', 'stuck_past_deadline') or (idle is not None and idle > STALL_S)

    lines = []
    if target:
        pct = int(round(100 * updates / target)) if target else None
        lines.append('步数 %s / %s%s' % (num(updates), num(target),
                                         '' if pct is None else '（%d%%）' % pct))
    else:
        lines.append('步数 %s' % num(updates))
    if horizon and target and horizon != target:
        lines.append('方案上限 %s 步' % num(horizon))
    head2 = []
    if loss is not None:
        head2.append('loss %s' % num(loss, 4))
    if lr is not None:
        head2.append('lr %s' % lrn(lr))
    if rate:
        head2.append('%s 步/秒' % num(rate, 2))
    if head2:
        lines.append(' · '.join(head2))
    if eta and phase == 'train':
        tail = '预计还要 %s' % hm(eta)
        if deadline:
            tail += '（到点 %s 自动收工）' % time.strftime('%m-%d %H:%M', time.localtime(deadline))
        lines.append(tail)
    if gpu:
        lines.append('GPU 峰值 %s GB' % num(gpu / 1024, 1))
    if idle is not None:
        lines.append('日志 %s前更新' % hm(idle))
    if bad:
        lines.append('要人看一眼：去操作台「05 训练」下面的 TRAIN OUT，或 tmux attach -t train。')
    return head, '\n'.join(lines), bad


def post(webhook, head, body, bad):
    text = ('【%s】\n%s' % (head, body))
    payload = {'msg_type': 'text', 'content': {'text': text}}
    if bad:
        payload['content']['text'] = '⚠️ ' + text
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(webhook, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode('utf-8', 'replace')
    try:
        j = json.loads(raw)
    except Exception:
        return raw[:120]
    return '' if (j.get('code') in (0, None) and j.get('StatusCode') in (0, None)) else raw[:160]


def main():
    ap = argparse.ArgumentParser(description='把一次训练的进度推到飞书群（手机直接看）')
    ap.add_argument('--out', required=True, help='训练输出目录，里面有 progress.json（就是方案里的 out_dir）')
    ap.add_argument('--log', default='train.log', help='日志文件，默认当前目录的 train.log')
    ap.add_argument('--webhook', default=os.environ.get(WEBHOOK, ''),
                    help='飞书群机器人 webhook；也可用环境变量 %s' % WEBHOOK)
    ap.add_argument('--name', default=None, help='显示用的名字，默认取 progress.json 里的 id')
    ap.add_argument('--every', type=float, default=60, help='多久检查一次，默认 60 秒')
    ap.add_argument('--heartbeat', type=float, default=HEARTBEAT_S, help='没变化也多久报一次，默认 600 秒')
    ap.add_argument('--once', action='store_true', help='只推一条就退出')
    args = ap.parse_args()

    if not args.webhook:
        sys.exit('缺 webhook：把飞书群机器人的地址传给 --webhook，或设 %s' % WEBHOOK)

    out = Path(args.out)
    log = Path(args.log)
    name = args.name
    last_push, last_sig = 0.0, None
    while True:
        st = read_state(out)
        lg = last_json_line(log)
        name = name or (st or {}).get('id') or out.name
        head, body, bad = compose(name, st, lg, time.time())
        sig = ((st or {}).get('updates'), (st or {}).get('phase'), head)
        now = time.time()
        due = (sig != last_sig and now - last_push >= min(args.every, 30)) or \
              (now - last_push >= args.heartbeat) or (bad and sig != last_sig)
        if args.once or due:
            try:
                err = post(args.webhook, head, body, bad)
                if err:
                    print('[push] 飞书回了异常：%s' % err, file=sys.stderr)
                else:
                    print('[push] %s' % head)
                    last_push, last_sig = now, sig
            except Exception as e:
                print('[push] 发不出去（下一轮再试）：%s' % e, file=sys.stderr)
        if args.once:
            return
        time.sleep(max(args.every, 10))


if __name__ == '__main__':
    main()
