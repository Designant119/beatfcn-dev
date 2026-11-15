"""造一套“能直接起训练”的演示数据（CPU 也能跑）。

这台机器没有真实谱面库、没有显卡，为了让人能先看到训练跑起来，
这里合成 1 张 8 秒 4K 谱面 + 200 帧 1024 维假特征，写进 .harness/demo_train，
并留下 demo_train.json。harness 看到它就会：
  * 训练参数自动指向这套假数据
  * 跳过“前面几步必须跑完”的前置检查
真实训练前删掉 demo_train.json 即可恢复原样。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.events.canonical import Event, LaneLoc
from scripts.harness import ROOT, _state_dir
from scripts.prepare_local_request_tape import requests_for_chart


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = _state_dir() / 'demo_train'
    tape, feats = root / 'tape', root / 'features'
    audio_sha = 'a' * 64
    feat_dir = feats / audio_sha
    for d in (tape, feat_dir, root / 'ckpt'):
        d.mkdir(parents=True, exist_ok=True)

    duration_us = 8_000_000
    events = [Event(f'e{i}', 200_000 + i * 220_000, 200_000 + i * 220_000, 'tap', LaneLoc(i % 4))
              for i in range(32)]
    rows, tokens, targets = requests_for_chart(events, 'mania_4k', duration_us, 'train')

    tok_path = tape / 'demo-tokens.npy'
    np.save(tok_path, tokens)
    tgt_path = tape / 'demo-target.npy'
    np.save(tgt_path, np.stack([targets['onset_us'], targets['end_us'],
                                targets['kind'], targets['lane']], axis=1))

    chart = dict(chart_id='demo-tiny', audio_sha256=audio_sha, canonical_sha256='b' * 64,
                 mode='mania_4k', split='train', duration_us=duration_us,
                 target_path=str(tgt_path), token_path=str(tok_path),
                 target_sha256=sha(tgt_path), token_sha256=sha(tok_path), requests=rows)
    (tape / 'demo-chart.json').write_text(json.dumps(chart, ensure_ascii=False), encoding='utf-8')

    frames = duration_us // 40_000
    rng = np.random.default_rng(0)
    for name in ('L12.npy', 'L23.npy'):
        np.save(feat_dir / name, rng.normal(0, 0.5, (frames, 1024)).astype(np.float32))

    def _rel(p):
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except Exception:                     # noqa: BLE001
            return str(p)
    cfg = dict(tape=_rel(tape), features=_rel(feats),
               out=_rel(root / 'ckpt' / 'tiny.pt'), steps=600)
    (_state_dir() / 'demo_train.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                                 encoding='utf-8')
    print(json.dumps(dict(ok=True, **cfg), ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
