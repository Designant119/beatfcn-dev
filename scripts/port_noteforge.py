"""把 noteforge 开源权重移植成 beatfcn 可直接加载的 checkpoint。

背景（2026-11-16 实测）：noteforge/models/noteforge-r5.pt 与 beatfcn 的
FullEventModel 结构一致，395 个张量里 392 个形状严丝合缝（318,160,903 参数），
没有任何单边多余的键。要处理的是词表相关的层：

    token_embedding.weight   (1985, 896)  -> beatfcn (1904, 896)   重初始化
    output.weight            (1985, 896)  -> beatfcn (1904, 896)   重初始化
    field_embedding.weight   (10, 896)    -> beatfcn (9, 896)      重初始化
    control_ids.weight       (8, 896)     -> beatfcn (7, 896)      重排

前三个差 81：noteforge 的 mania 字段带 81 格 4-lane pattern，beatfcn 把 lane
预测整个退役了（lane 归编辑器管）。

control_ids 差的是**行序**，而且两边连排序规则都不同：
    noteforge: CONTROL_FIELDS = tuple(BAND_EDGES)         -> 插入顺序
               density / peak / hold_head / hold_occupancy / chord / jack / flick / drag
    beatfcn:   CONTROL_FIELDS = tuple(sorted(BAND_EDGES)) -> 字母序
               chord / density / drag / flick / hold_head / hold_occupancy / peak
所以直接搬会让每个旋钮都读到别人的嵌入。这里按字段名映射（CONTROL_KEEP），
实算出来是 (4, 0, 7, 6, 2, 3, 1)，jack_ratio 因为已退役被自然丢掉。

用法（在有 r5 权重的那台机器上跑，不要在小内存机器上跑）：

    python -m scripts.port_noteforge \
        --src /data/noteforge-r5.pt \
        --out /data/beatfcn-r5-ported.pt

    # 只核对形状、不写文件、不占内存（meta 设备）
    python -m scripts.port_noteforge --src ... --check

产物是 beatfcn 格式的 checkpoint（schema=beatfcn.port.v1）：
  - config 沿用 noteforge 的尺寸（d=896 / ffn=3584 / encoder_layers=3），
    只有 controls 改成 7 跟 beatfcn 的数据对齐；想变小再走 scripts/evolve_surgery.py。
  - vocabulary 是 beatfcn 的 1904 词表。
  - 重初始化的层用固定 seed，可复现。
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beatfcn.events import controls as C  # noqa: E402,F401
from beatfcn.model import features as F  # noqa: E402
from beatfcn.model.full_event_codec import EventVocabulary  # noqa: E402
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel  # noqa: E402

SCHEMA = 'beatfcn.port.v1'

# 控制面板的行序两边**完全不同**，不能按位置硬搬：
#   noteforge: CONTROL_FIELDS = tuple(BAND_EDGES)         -> BAND_EDGES 的插入顺序
#   beatfcn:   CONTROL_FIELDS = tuple(sorted(BAND_EDGES)) -> 字母序
# 所以按字段名映射。beatfcn 的字段表直接从代码取，它以后加字段这里自动跟上。
NOTEFORGE_CONTROLS = ('density_notes_per_second', 'peak_notes_per_second', 'hold_head_ratio',
                      'hold_occupancy_ratio', 'chord_ratio', 'jack_ratio', 'flick_ratio', 'drag_ratio')
CONTROL_KEEP = tuple(NOTEFORGE_CONTROLS.index(f) for f in F.CONTROL_FIELDS)
REINIT = ('token_embedding.weight', 'output.weight', 'field_embedding.weight')


def load_source(path):
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(saved, dict) or 'model' not in saved or 'config' not in saved:
        raise SystemExit(f'{path} 不是可识别的 checkpoint（缺 model/config）')
    return saved


def target_config(saved, controls):
    return replace(FullEventConfig(**dict(saved['config'])), controls=controls)


def compare(saved, controls=7):
    """返回 (shape_bad, reord_bad, only_weight, only_model, conf, vocab)。"""
    conf = target_config(saved, controls)
    vocab = EventVocabulary(**saved.get('vocabulary', {}))
    with torch.device('meta'):
        model = FullEventModel(conf, vocab)
    want = model.state_dict()
    weight = saved['model']
    shape_bad, reord_bad = [], []
    for k in sorted(set(weight) & set(want)):
        if tuple(weight[k].shape) == tuple(want[k].shape):
            continue
        if k == 'control_ids.weight' and weight[k].shape[1] == want[k].shape[1]:
            reord_bad.append(k)
        else:
            shape_bad.append(k)
    only_weight = sorted(set(weight) - set(want))
    only_model = sorted(set(want) - set(weight))
    return shape_bad, reord_bad, only_weight, only_model, conf, vocab


def port(saved, seed=20261116, controls=7):
    """抄一致层 + 重排控制行 + 重初始化词表层；返回 (state, config, vocabulary, report)。"""
    shape_bad, reord_bad, only_weight, only_model, conf, vocab = compare(saved, controls)
    if only_weight or only_model:
        raise SystemExit('两边存在单边参数，移植脚本只处理已知差异，请先人工核对：'
                         f'only_weight={only_weight[:5]} only_model={only_model[:5]}')
    torch.manual_seed(seed)
    fresh = FullEventModel(conf, vocab)
    fresh_sd = fresh.state_dict()
    state, copied, reordered = {}, 0, []
    for key, value in saved['model'].items():
        if key in reord_bad:
            state[key] = value[list(CONTROL_KEEP)].to(torch.float32).clone()
            reordered.append(key)
        elif key in shape_bad:
            state[key] = fresh_sd[key].detach().to(torch.float32).clone()
        else:
            state[key] = value.to(torch.float32)
            copied += 1
    del fresh, fresh_sd
    report = dict(copied_layers=copied, reinit_layers=shape_bad, reordered_layers=reordered,
                  reinit_parameters=sum(saved['model'][k].numel() for k in shape_bad),
                  source_parameters=sum(v.numel() for v in saved['model'].values()))
    return state, conf, vocab, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True, type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--controls', type=int, default=7, help='控制面板行数，beatfcn 是 7')
    ap.add_argument('--seed', type=int, default=20261116)
    ap.add_argument('--check', action='store_true', help='只核对形状，不写文件')
    args = ap.parse_args(argv)

    saved = load_source(args.src)
    shape_bad, reord_bad, only_weight, only_model, conf, vocab = compare(saved, args.controls)
    weight = saved['model']
    total = sum(v.numel() for v in weight.values())
    print(f'源：{args.src}  张量 {len(weight)}  参数 {total}')
    print(f'词表：beatfcn={vocab.size}  权重侧 output={weight["output.weight"].shape[0]}'
          f'  控制：目标 {args.controls} 行 / 权重 {weight["control_ids.weight"].shape[0]} 行')
    print(f'可直抄 {len(weight) - len(shape_bad) - len(reord_bad)} 层 /'
          f' 重初始化 {len(shape_bad)} 层 / 重排 {len(reord_bad)} 层 /'
          f' 只权重 {len(only_weight)} / 只模型 {len(only_model)}')
    for k in shape_bad:
        print(f'  重来：{k}  源 {tuple(weight[k].shape)}')
    for k in reord_bad:
        print(f'  重排：{k}  源 {tuple(weight[k].shape)} -> {list(CONTROL_KEEP)}')
    if args.check:
        print('--check：只核对，不写文件。')
        return 0
    if not args.out:
        raise SystemExit('要写文件就传 --out')
    state, config, vocabulary, report = port(saved, seed=args.seed, controls=args.controls)
    out = dict(schema=SCHEMA, config=dict(config.__dict__), vocabulary=dict(vocabulary.__dict__),
               model=state, parameters=sum(v.numel() for v in state.values()),
               provenance=dict(source=str(args.src), source_schema=saved.get('schema'),
                               source_parameters=report['source_parameters'],
                               reinit_layers=report['reinit_layers'],
                               reordered_layers=report['reordered_layers'],
                               reinit_parameters=report['reinit_parameters'], seed=args.seed,
                               controls=args.controls))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f'已写 {args.out}  参数 {out["parameters"]}')
    print(f'重来 {len(report["reinit_layers"])} 层 / {report["reinit_parameters"]} 参数；'
          f'重排 {len(report["reordered_layers"])} 层')
    for k in report['reinit_layers']:
        print(f'  {k}')
    for k in report['reordered_layers']:
        print(f'  {k}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
