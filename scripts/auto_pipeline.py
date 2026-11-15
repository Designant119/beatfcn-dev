"""从某一步开始，一路往下自动跑，哪步失败就停。给“一条龙/全部跑完”用。

每步由 harness.start() 起在各自 tmux 会话里；本进程只负责等它结束、检查产物、判断下一步。
"""
import argparse
import time

from scripts.harness import ORDER, demo_train, log, running, start, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='from_step', default=None, choices=ORDER)
    ap.add_argument('--to', dest='to_step', default='train', choices=ORDER)
    args = ap.parse_args()

    steps = list(ORDER)
    if demo_train():
        steps = ['train']                       # 演示环境只有这一步能跑
    begin = args.from_step
    if not begin:
        st = status()
        begin = next((s for s in steps if not st[s]['done']), steps[-1])
    seq = steps[steps.index(begin): steps.index(args.to_step) + 1]
    print(f'自动流水线：{" → ".join(seq)}', flush=True)

    for step in seq:
        st = status()[step]
        if st['done'] and st.get('verified'):
            print(f'[{step}] 已经跑完，跳过', flush=True)
            continue
        if not st['running']:
            print(f'[{step}] 启动…', flush=True)
            res = start(step, {})
            if not res.get('ok'):
                errs = '；'.join(res.get('errors') or [])
                print(f'[{step}] 启动失败：{errs}', flush=True)
                return 2
        while running(step):
            time.sleep(5)
        st = status()[step]
        if not st['done']:
            print(f'[{step}] 没跑成：{st.get("why") or "没看到产物"}', flush=True)
            print(log(step, 20), flush=True)
            return 3
        print(f'[{step}] 完成', flush=True)
    print('全部跑完。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
