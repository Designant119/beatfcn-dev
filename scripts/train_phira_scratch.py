"""从 0 初始化训练 BeatFCN 大模型（默认 l322m 322M 主线）。

不是 fine-tune：不读 init checkpoint，直接随机初始化 `FullEventConfig`，用
tape 里的谱面请求 + 已提取的 MERT 特征做下一步监督。适合旧 checkpoint 丢了之后
重新起一条自己的 322M 主线。

  python -m scripts.train_phira_scratch --plan plans/evolve/l322m-scratch.json \
      --updates 2400 --device cuda

方案里写 tape / features / config / steps / lr / out / out_dir。每一步可累积多条
请求（accum_steps），物理 microbatch 先只支持 1；4090 24G 用 1 + 梯度检查点。
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.model.full_event_codec import EventRequest, RhythmEvent, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.audio_dependence import token_losses


def build_inputs(cdir, start_us, end_us, controls, mode, cfg):
    """把一条请求需要的音频/特征装成模型输入契约。

    L12/L23 是 25fps 的 MERT 两层特征；mel10_complete 是 100fps 的 128 维 mel。
    没有 mel 文件时给 4 帧零，保证 demo/旧特征目录也能跑通；正式数据应带
    mel10_complete.npy。
    """
    L12 = np.load(cdir / 'L12.npy', mmap_mode='r')
    L23 = np.load(cdir / 'L23.npy', mmap_mode='r')
    lo, hi = max(0, start_us // 40_000), min(len(L12), end_us // 40_000)
    lo, hi = max(0, lo - 4), min(len(L12), hi + 4)
    if hi <= lo:
        raise ValueError('请求时间窗为空：%s..%s' % (start_us, end_us))
    local = np.stack([np.asarray(L12[lo:hi], np.float32), np.asarray(L23[lo:hi], np.float32)])
    local = local[None, ..., None, :]                                     # [1,2,T,1,1024]
    song = np.asarray(L12[lo:hi], np.float32).mean(0, keepdims=True)[None]  # [1,1,1024]
    ref = np.zeros((1, 1, 12), np.float32)
    ref[0, 0, 1] = (end_us - start_us) / 1_200_000_000
    ref[0, 0, -1] = 1.0
    T = hi - lo
    times = (np.arange(lo, hi) * 0.04 + 0.02 - start_us / 1e6).astype(np.float32)

    mel_path = cdir / 'mel10_complete.npy'
    if cfg.mel_bins and mel_path.exists():
        mel_all = np.load(mel_path, mmap_mode='r')
        lo_mel = max(0, start_us // 10_000)
        hi_mel = min(len(mel_all), end_us // 10_000)
        if hi_mel <= lo_mel:
            raise ValueError('mel 时间窗为空：%s..%s' % (start_us, end_us))
        mel = np.asarray(mel_all[lo_mel:hi_mel], np.float32)[None]         # [1,T_mel,128]
        if mel.shape[-1] != cfg.mel_bins:
            raise ValueError('mel 维度 %d 与 config.mel_bins=%d 不一致：%s'
                             % (mel.shape[-1], cfg.mel_bins, mel_path))
        mel_times = (np.arange(lo_mel, hi_mel) * 0.01 + 0.005 - start_us / 1e6).astype(np.float32)[None]
    else:
        mel = np.zeros((1, 4, cfg.mel_bins), np.float32)
        mel_times = np.zeros((1, 4), np.float32)
    T_mel = mel.shape[1]

    out = dict(
        local=torch.from_numpy(local),
        local_times=torch.from_numpy(times[None]),
        local_valid=torch.ones(1, T, dtype=torch.bool),
        song=torch.from_numpy(song),
        song_times=torch.zeros(1, 1),
        song_valid=torch.ones(1, 1, dtype=torch.bool),
        mel=torch.from_numpy(mel),
        mel_times=torch.from_numpy(mel_times),
        mel_valid=torch.ones(1, T_mel, dtype=torch.bool),
        reference=torch.from_numpy(ref),
        reference_times=torch.zeros(1, 1),
        reference_valid=torch.ones(1, 1, dtype=torch.bool),
        controls=torch.from_numpy(np.asarray(controls, np.float32)[None]),
        mode=torch.tensor([0 if mode == 'mania_4k' else 1], dtype=torch.int64),
    )
    if cfg.intensity_dim:
        # 先给“未指定”的空曲线；训练模型仍会看到这路条件，但不会假装有标签。
        out['intensity'] = torch.zeros(1, 4, cfg.intensity_dim)
        out['intensity_times'] = torch.zeros(1, 4)
        out['intensity_valid'] = torch.zeros(1, 4, dtype=torch.bool)
    return out


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def checkpoint_state(model):
    return getattr(model, '_orig_mod', model).state_dict()


def save_checkpoint(path, cfg, model, optimizer, step, save_optimizer):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'schema': 'beatfcn.scratch-checkpoint.v1', 'config': vars(cfg),
             'model': checkpoint_state(model), 'step': int(step)}
    if save_optimizer:
        state['optimizer'] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def latest_checkpoint(plan, out):
    candidates = sorted(Path(plan.get('out_dir') or out.parent).glob('step-*.pt'))
    if out.exists():
        candidates.append(out)
    return candidates[-1] if candidates else None


def pick_chart(charts, step, micro, seed):
    return charts[(step * 7 + micro * 3 + seed) % len(charts)]


def make_batch(chart, features, cfg, require_mel=False):
    tokens = np.load(chart['token_path'], allow_pickle=False)
    cdir = features / chart['audio_sha256']
    if cfg.mel_bins and require_mel and not (cdir / 'mel10_complete.npy').exists():
        raise ValueError('322M 从0训练需要 mel10_complete.npy：%s' % cdir)
    req_rows = chart.get('requests') or []
    if not req_rows:
        raise ValueError('chart 没有 requests：%s' % chart.get('chart_id'))
    row = req_rows[min(2, len(req_rows) - 1)]
    kw = dict(row['request'])
    kw['known_carry'] = tuple(RhythmEvent(**e) for e in (kw.get('known_carry') or ()))
    req = EventRequest(**kw)
    seg = [int(t) for t in tokens[row['token_start']:row['token_stop']]]
    if not seg:
        raise ValueError('chart token 区间为空：%s' % chart.get('chart_id'))
    seq = teacher_sequence(seg, req)
    inputs = build_inputs(cdir, int(kw['start_us']), int(kw['end_us']),
                          row['controls'], kw['mode'], cfg)
    seqs = {k: torch.from_numpy(v[None]) if isinstance(v, np.ndarray) else v
            for k, v in seq.items()}
    seqs['allowed'] = seqs['allowed'].bool()
    seqs['targets'] = seqs['targets'].long()
    return inputs, seqs, chart['chart_id'], int(seqs['targets'].numel())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--updates', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--save-optimizer', action='store_true')
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--deadline-unix', type=int, default=None)
    args = ap.parse_args()

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('kind') != 'scratch':
        raise SystemExit('这个脚本只跑 kind=scratch 的从0方案；fine-tune 请用 evolve_finetune')
    if 'config' not in plan:
        raise SystemExit('从0方案必须自带 config（模型结构不能靠旧 checkpoint 猜）')
    cfg = FullEventConfig(**plan['config'])
    updates = int(args.updates if args.updates is not None else plan.get('steps') or 2400)
    if updates < 1:
        raise SystemExit('updates 必须 >= 1')
    if int(plan.get('physical_microbatch_requests', 1)) != 1:
        print('提示：从0脚本当前只按 1 条/次跑；方案里的 physical_microbatch 会忽略', flush=True)

    charts = sorted(Path(plan['tape']).glob('*-chart.json'))
    if not charts:
        raise SystemExit('tape 里找不到 *-chart.json：%s' % plan['tape'])
    features = Path(plan['features'])
    if not features.is_dir() or not any(features.iterdir()):
        raise SystemExit('特征目录不存在或为空：%s' % features)

    out = Path(plan['out'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(plan.get('out_dir') or out.parent / (out.stem + '-checkpoints'))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(plan.get('seed', 20261115))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    device = args.device or plan.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    torch.set_num_threads(int(plan.get('loader_workers', 4)))

    start_step = 0
    model = FullEventModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(plan.get('lr', 1e-4)),
                                  betas=(0.9, 0.95), weight_decay=0.01)
    if args.resume:
        ckpt = latest_checkpoint(plan, out)
        if ckpt is None:
            raise SystemExit('要 resume 但没找到 checkpoint：%s' % out_dir)
        blob = torch.load(ckpt, map_location='cpu', weights_only=False)
        cfg = FullEventConfig(**blob['config'])
        model = FullEventModel(cfg).to(device)
        model.load_state_dict(blob['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(plan.get('lr', 1e-4)),
                                      betas=(0.9, 0.95), weight_decay=0.01)
        if args.save_optimizer and blob.get('optimizer'):
            optimizer.load_state_dict(blob['optimizer'])
        start_step = int(blob.get('step', 0))
        print('resume from %s step=%d' % (ckpt, start_step), flush=True)

    compile_model = bool(plan.get('compile', False)) and not args.no_compile
    if compile_model and device == 'cuda':
        model = torch.compile(model)
    accum = max(1, int(plan.get('accum_steps', 4)))
    sched = plan.get('schedule') or {}
    base_lr = float(plan.get('lr', 1e-4))
    warmup = max(0, int(sched.get('warmup_updates', 0)))
    floor_lr = float(sched.get('minimum_lr', base_lr * 0.1))
    every = int(plan.get('light_checkpoint_every', 300))
    keep = int(plan.get('light_checkpoint_keep', 2))
    save_opt = bool(plan.get('save_optimizer', False)) or args.save_optimizer
    deadline = args.deadline_unix
    print('从0训练：%s · %.2fM 参数 · updates=%d · accum=%d · device=%s'
          % (plan.get('model') or plan_path.stem,
             sum(p.numel() for p in model.parameters()) / 1e6,
             updates, accum, device), flush=True)

    step = start_step
    while step < updates:
        if deadline is not None and time.time() > deadline:
            print('到 deadline，停在 step=%d' % step, flush=True)
            break
        step += 1
        if step <= warmup:
            lr_now = base_lr * step / max(1, warmup)
        else:
            progress = min(1.0, (step - warmup) / max(1, updates - warmup))
            lr_now = floor_lr + 0.5 * (base_lr - floor_lr) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr_now
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        tokens = 0
        chart_ids = []
        for micro in range(accum):
            chart = pick_chart(charts, step, micro, seed)
            chart = json.loads(chart.read_text(encoding='utf-8'))
            inputs, seqs, chart_id, n_tokens = make_batch(
                chart, features, cfg, require_mel=bool(plan.get('require_mel', False)))
            inputs = to_device(inputs, device)
            seqs = to_device(seqs, device)
            losses, _, _ = token_losses(model, inputs, seqs)
            valid = seqs['valid'].float()
            loss = (losses * valid).sum() / valid.sum().clamp(min=1)
            (loss / accum).backward()
            loss_sum += float(loss.detach())
            tokens += n_tokens
            chart_ids.append(chart_id)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(json.dumps({'step': step, 'loss': round(loss_sum / accum, 5),
                          'lr': round(lr_now, 8), 'tokens': tokens,
                          'charts': chart_ids}, ensure_ascii=False), flush=True)
        if every > 0 and step % every == 0 and step < updates:
            path = out_dir / ('step-%06d.pt' % step)
            save_checkpoint(path, cfg, model, optimizer, step, save_opt)
            numbered = sorted(out_dir.glob('step-*.pt'))
            for old in numbered[:-max(1, keep)]:
                try:
                    old.unlink()
                except OSError:
                    pass
            print('checkpoint -> %s' % path, flush=True)

    save_checkpoint(out, cfg, model, optimizer, step, save_opt)
    print('done step=%d -> %s' % (step, out), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
