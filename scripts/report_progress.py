#!/usr/bin/env python3
"""把训练进度推到飞书，人在外面也能看（不需要公网、不需要 VPN）。

两种投递方式，选一个就行：

  1) 群机器人（最省事，一条 webhook URL，30 秒配好）
     python3 -m scripts.report_progress --webhook https://open.feishu.cn/open-apis/bot/v2/hook/xxxx
     飞书群 -> 设置 -> 群机器人 -> 添加 -> 自定义机器人 -> 复制 Webhook 地址

  2) 飞书文档（要自建应用，一次性配好，能一直往同一篇文档里追加）
     export FEISHU_APP_ID=cli_xxx
     export FEISHU_APP_SECRET=xxx
     python3 -m scripts.report_progress --doc <文档 token 或文档 URL>

轮询默认 60 秒，步数没变就不重复推；--once 只推一次（适合手点、或塞进定时任务）。

读的是训练器自己写的东西，不依赖本仓库的其它模块：
  - <out_dir>/progress.json   进度（Progress 类每 30 秒落一次盘）
  - <out_dir>/step-*.pt       快照，用来报告“最近一份存到第几步”
  - train.log                 每 10 步一行 JSON，用来兜底和补 loss
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

FEISHU = 'https://open.feishu.cn/open-apis'
STATE_DIR = os.path.join(os.environ.get('TMPDIR') or '/tmp', 'beatfcn-report')


def read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def tail_lines(path, n=40):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().splitlines()[-n:]
    except Exception:
        return []


def log_last(log_path):
    """train.log 最后一行能解析的 JSON（训练器每 10 步打一条）。"""
    for line in reversed(tail_lines(log_path, 60)):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return None


def latest_ckpt(out_dir):
    try:
        files = [f for f in os.listdir(out_dir) if re.fullmatch(r'step-\d{6,}\.pt', f)]
    except OSError:
        return None, 0
    if not files:
        return None, 0
    files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
    return files[-1], int(re.findall(r'\d+', files[-1])[0])


def human_duration(seconds):
    if seconds is None or seconds < 0:
        return '—'
    seconds = int(seconds)
    if seconds < 60:
        return '%d 秒' % seconds
    if seconds < 3600:
        return '%d 分' % (seconds // 60)
    if seconds < 86400:
        return '%d 小时 %d 分' % (seconds // 3600, (seconds % 3600) // 60)
    return '%d 天 %d 小时' % (seconds // 86400, (seconds % 86400) // 3600)


def fmt_clock(unix):
    try:
        return datetime.fromtimestamp(float(unix)).strftime('%m-%d %H:%M')
    except Exception:
        return '—'


def bar(done, total, width=20):
    if not total or total <= 0:
        return ''
    filled = max(0, min(width, int(width * done / total)))
    return '`' + '█' * filled + '░' * (width - filled) + '` %d/%d' % (done, total)


def collect(out_dir, log_path):
    """returns dict 描述当前状态；数据都没有时返回 None。"""
    prog = read_json(os.path.join(out_dir, 'progress.json')) or {}
    last = log_last(log_path) or {}
    updates = prog.get('updates')
    if updates is None:
        updates = last.get('updates')
    if updates is None:
        return None
    target = prog.get('target_updates') or last.get('target')
    horizon = prog.get('horizon_updates')
    start = prog.get('start_updates') or 0
    loss = prog.get('loss')
    if loss is None:
        loss = last.get('loss')
    rate = prog.get('recent_updates_per_s') or prog.get('rate_updates_per_s') or last.get('updates_per_s')
    stale = prog.get('since_last_update_s')
    if stale is None:
        stale = (time.time() - float(prog['written_unix'])) if prog.get('written_unix') else None
    done = updates - start
    left = (target - updates) if target else None
    eta = (left / rate) if (left and rate) else None
    ck_name, ck_step = latest_ckpt(out_dir)
    return dict(phase=prog.get('phase') or last.get('phase') or '?', updates=updates, start=start,
               target=target, horizon=horizon, loss=loss, lr=prog.get('lr') or last.get('lr'),
               rate=rate, eta=eta, stale=stale, done=done, deadline=prog.get('deadline_unix'),
               grad=prog.get('grad_norm') or last.get('grad'), ck_name=ck_name, ck_step=ck_step,
               gpu_mib=prog.get('gpu_peak_mib'), written=prog.get('written_unix'))


def plan_facts(plan_path):
    spec = read_json(plan_path) if plan_path else None
    if not spec:
        return {}
    sched = spec.get('schedule') or {}
    return dict(id=spec.get('id'), lr=spec.get('learning_rate'),
                every=spec.get('light_checkpoint_every'), keep=spec.get('light_checkpoint_keep'),
                out_dir=spec.get('out_dir'), horizon=sched.get('horizon_updates'),
                batch_hint=spec.get('loader_workers'))


def text_report(st, facts, first_run):
    """给飞书卡片 / 文档正文用的纯文本（markdown-ish）。"""
    lines = []
    title = facts.get('id') or '训练'
    head = '**%s**  %s' % (title, '刚开始' if first_run else '进行中')
    if st['phase']:
        head += ' · 阶段 `%s`' % st['phase']
    lines.append(head)
    if st['target']:
        lines.append(bar(st['updates'], st['target']))
    else:
        lines.append('已跑 **%d** 步' % st['updates'])
    bits = []
    if st['loss'] is not None:
        bits.append('loss **%.4f**' % float(st['loss']))
    if st['lr'] is not None:
        bits.append('lr %g' % float(st['lr']))
    if st['rate']:
        bits.append('%.2f 步/秒' % float(st['rate']))
    if st['grad'] is not None:
        bits.append('grad %.2f' % float(st['grad']))
    if bits:
        lines.append(' · '.join(bits))
    if st['eta']:
        lines.append('剩 **%s**（约 %s 跑完）' % (human_duration(st['eta']),
                                             fmt_clock(time.time() + st['eta'])))
    elif st['eta'] is None and st['target'] and st['updates'] >= st['target']:
        lines.append('**已经跑满了设定的步数**')
    if st['deadline']:
        left = float(st['deadline']) - time.time()
        if left > 0:
            lines.append('最晚 %s 收工（还剩 %s）' % (fmt_clock(st['deadline']), human_duration(left)))
        else:
            lines.append('已过最晚时间 %s，应该在收尾了' % fmt_clock(st['deadline']))
    if st['ck_name']:
        lines.append('最近一份快照 `%s`（第 %d 步）' % (st['ck_name'], st['ck_step']))
    if st['gpu_mib']:
        lines.append('显存峰值 %d MiB' % int(st['gpu_mib']))
    if st['stale'] is not None and st['stale'] > 180:
        lines.append('⚠️ **%s** 没更新过数字了，可能在等数据或卡住'
                     % human_duration(st['stale']))
    if facts:
        tail = []
        if facts.get('lr'):
            tail.append('学习率 %g' % float(facts['lr']))
        if facts.get('every'):
            tail.append('每 %d 步存一份，留 %s 份' % (facts['every'], facts.get('keep') or '?'))
        if facts.get('horizon'):
            tail.append('方案上限 %d 步' % facts['horizon'])
        if tail:
            lines.append('_%s_' % ' · '.join(tail))
    return '\n'.join(lines)


def post_json(url, payload, headers=None, timeout=15):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST',
                                 headers={'Content-Type': 'application/json; charset=utf-8',
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def send_webhook(url, text, ok=True):
    card = {
        'msg_type': 'interactive',
        'card': {
            'config': {'wide_screen_mode': True},
            'header': {'template': 'green' if ok else 'orange',
                       'title': {'tag': 'plain_text', 'content': '训练进度'}},
            'elements': [{'tag': 'div', 'text': {'tag': 'lark_md', 'content': text}}],
        },
    }
    return post_json(url, card)


def tenant_token(app_id, app_secret):
    r = post_json(FEISHU + '/auth/v3/tenant_access_token/internal',
                  {'app_id': app_id, 'app_secret': app_secret})
    token = r.get('tenant_access_token')
    if not token:
        raise SystemExit('拿不到 tenant_access_token：%s' % r)
    return token


def doc_id_of(target):
    """接受文档 URL、/docx/xxx 或裸 token。"""
    m = re.search(r'/(?:docx|docs)/([A-Za-z0-9]{12,})', target or '')
    return m.group(1) if m else (target or '').strip()


def append_to_doc(token, doc, text):
    url = '%s/docx/v1/documents/%s/blocks/%s/children?document_revision_id=-1' % (FEISHU, doc, doc)
    payload = {'children': [{
        'block_type': 2,
        'text': {'elements': [{'text_run': {'content': text, 'text_element_style': {}}}], 'style': {}},
    }], 'index': -1}
    return post_json(url, payload, headers={'Authorization': 'Bearer ' + token})


def main():
    a = argparse.ArgumentParser(description='训练进度推到飞书')
    a.add_argument('--plan', help='方案 json；用它的 out_dir/learning_rate 等')
    a.add_argument('--out-dir', help='直接指定训练输出目录（含 progress.json 和 step-*.pt）')
    a.add_argument('--log', help='训练日志，默认 <仓库>/train.log')
    a.add_argument('--webhook', default=os.environ.get('FEISHU_WEBHOOK'), help='群机器人 webhook')
    a.add_argument('--doc', default=os.environ.get('FEISHU_DOC'), help='飞书文档 token 或 URL')
    a.add_argument('--interval', type=int, default=60, help='轮询秒数，默认 60')
    a.add_argument('--once', action='store_true', help='只推一次就退出')
    a.add_argument('--dry-run', action='store_true', help='只打印将发送的内容，不发网络请求')
    a.add_argument('--state', default=os.path.join(STATE_DIR, 'last.json'))
    args = a.parse_args()

    facts = plan_facts(args.plan)
    out_dir = args.out_dir or facts.get('out_dir')
    if not out_dir:
        raise SystemExit('要么给 --out-dir，要么给 --plan（方案里有 out_dir）')
    log_path = args.log or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train.log')
    if not args.webhook and not args.doc:
        raise SystemExit('要给 --webhook 或 --doc（也可以用环境变量 FEISHU_WEBHOOK / FEISHU_DOC）')

    token = None
    if args.doc:
        app_id, secret = os.environ.get('FEISHU_APP_ID'), os.environ.get('FEISHU_APP_SECRET')
        if not (app_id and secret):
            raise SystemExit('用 --doc 需要先设 FEISHU_APP_ID 和 FEISHU_APP_SECRET')
        if not args.dry_run:
            token = tenant_token(app_id, secret)
    doc = doc_id_of(args.doc) if args.doc else None

    os.makedirs(os.path.dirname(args.state) or '.', exist_ok=True)
    last = read_json(args.state) or {}
    first_sent = bool(last.get('sent'))

    while True:
        st = collect(out_dir, log_path)
        if st is None:
            print('[report] %s 里还没有 progress.json，训练可能没起' % out_dir, flush=True)
            if args.once:
                return 1
            time.sleep(args.interval)
            continue
        same = last.get('updates') == st['updates'] and last.get('phase') == st['phase']
        if same and not args.once and last.get('sent'):
            time.sleep(args.interval)
            continue
        # 只对“还没发过”的情况叫“刚开始”，之后一律“进行中”
        text = text_report(st, facts, first_run=(not first_sent))
        stamp = time.strftime('%H:%M')
        if args.dry_run:
            print('---- webhook ----\n%s\n---- 正文 ----\n%s\n' % (args.webhook or '(未配)', text), flush=True)
        else:
            if args.webhook:
                r = send_webhook(args.webhook, text)
                print('[report] %s webhook -> %s' % (stamp, r.get('code', r)), flush=True)
            if doc and token:
                r = append_to_doc(token, doc, '【%s】\n%s' % (stamp, text))
                print('[report] %s 文档追加 -> %s' % (stamp, r.get('code', r)), flush=True)
        try:
            with open(args.state, 'w', encoding='utf-8') as f:
                json.dump({'updates': st['updates'], 'phase': st['phase'], 'sent': True,
                           'at': time.time()}, f)
        except Exception:
            pass
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    sys.exit(main())
