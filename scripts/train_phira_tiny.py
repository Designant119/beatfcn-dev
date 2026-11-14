"""从零训练的最小闭环：用本项目自己的 features 布局跑几步，产出可被 draft_song 加载的 ckpt。

历史实验的 panel 读 inputs/r09_positive_pairs 与 inputs/d65_local（L9/16/24），
那是另一套目录。这里直接读 batch_extract_features 的产物 features/<sha>/L12.npy、
L23.npy、mel10_complete.npy，把 1024 维按 bands=1 装进模型的五维契约，
因此不必改模型代码，只把 config 设成 taps=2 / bands=1 / tap_dim=1024。

  python -m scripts.train_phira_tiny --tape /tmp/full_tape --features /tmp/f100/features \
      --steps 2 --out /tmp/tiny-phira.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.model.full_event_codec import EventRequest, RhythmEvent, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.audio_dependence import token_losses


def collate_sequences(seqs):
    fields = seqs[0].keys()
    out = {}
    for k in fields:
        if k == 'allowed':
            out[k] = torch.from_numpy(np.stack([s[k] for s in seqs]).astype(bool))
        elif k == 'targets':
            out[k] = torch.from_numpy(np.stack([s[k] for s in seqs]).astype(np.int64))
        else:
            v = [s[k] for s in seqs]
            out[k] = torch.from_numpy(np.stack(v)) if isinstance(v[0], np.ndarray) else v
    return out


def build_inputs(cdir, sha, lo, hi, controls):
    L12 = np.load(cdir / 'L12.npy', mmap_mode='r')
    L23 = np.load(cdir / 'L23.npy', mmap_mode='r')
    lo, hi = max(0, lo - 4), min(len(L12), hi + 4)
    local = np.stack([np.asarray(L12[lo:hi], np.float32), np.asarray(L23[lo:hi], np.float32)])  # [2,T,1024]
    local = local[None, ..., None, :]                                     # [1,2,T,1,1024]
    song = np.asarray(L12[lo:hi], np.float32).mean(0, keepdims=True)[None]  # [1,1,1024]
    ref = np.zeros((1, 1, 12), np.float32)
    ref[0, 0, 1] = (hi - lo) * 0.04
    ref[0, 0, -1] = 1.0
    T = hi - lo
    return dict(
        local=torch.from_numpy(local),
        local_times=torch.from_numpy((np.arange(T)*0.04+0.02).astype(np.float32)[None]),
        local_valid=torch.ones(1, T, dtype=torch.bool),
        song=torch.from_numpy(song),
        song_times=torch.zeros(1, 1),
        song_valid=torch.ones(1, 1, dtype=torch.bool),
        mel=torch.zeros(1, 4, 128),
        mel_times=torch.zeros(1, 4),
        mel_valid=torch.ones(1, 4, dtype=torch.bool),
        reference=torch.from_numpy(ref),
        reference_times=torch.zeros(1, 1),
        reference_valid=torch.ones(1, 1, dtype=torch.bool),
        controls=torch.from_numpy(np.asarray(controls, np.float32)[None]),
        mode=torch.zeros(1, dtype=torch.int64),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tape', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    cfg = FullEventConfig(d=128, heads=4, kv_heads=2, ffn=256, encoder_layers=1,
                          decoder_layers=2, taps=2, bands=1, tap_dim=1024,
                          mel_bins=0, intensity_dim=0)
    model = FullEventModel(cfg).to(args.device)
    print(f'参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    charts = sorted(Path(args.tape).glob('*-chart.json'))
    used = 0
    for step in range(1, args.steps + 1):
        chart = json.loads(charts[step % len(charts)].read_text(encoding='utf-8'))
        tokens = np.load(chart['token_path'], allow_pickle=False)
        cdir = Path(args.features) / chart['audio_sha256']
        row = chart['requests'][min(2, len(chart['requests']) - 1)]
        kw = dict(row['request'])
        kw['known_carry'] = tuple(RhythmEvent(**e) for e in (kw.get('known_carry') or ()))
        req = EventRequest(**kw)
        seg = [int(t) for t in tokens[row['token_start']:row['token_stop']]]
        seq = teacher_sequence(seg, req)
        n_frames = len(np.load(cdir / 'L12.npy', mmap_mode='r'))
        lo = max(0, int(int(kw['start_us']) / 1_000_000 * 25))
        hi = min(n_frames, int(int(kw['end_us']) / 1_000_000 * 25))
        inputs = build_inputs(cdir, chart['audio_sha256'], lo, hi, row['controls'])
        seqs = {k: torch.from_numpy(v[None]) if isinstance(v, np.ndarray) else v for k, v in seq.items()}
        seqs['allowed'] = seqs['allowed'].bool()
        seqs['targets'] = seqs['targets'].long()
        losses, _, _ = token_losses(model, inputs, seqs)
        jump = torch.tensor([float(seqs['allowed'].sum().item())])
        loss = (losses.sum() / torch.clamp(jump, min=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        used += 1
        print(json.dumps(dict(step=step, loss=round(float(loss), 4),
                              tokens=int(seqs['targets'].numel()), chart=chart['chart_id'])))

    torch.save({'config': vars(cfg), 'model': model.state_dict()}, args.out)
    print(f'ckpt -> {args.out}（训练 {used} 步）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
