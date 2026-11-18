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

control_ids 差的是**行序**，这里有个踩过的坑（2026-11 真出过事故）：
    beatfcn 代码里有**两份** CONTROL_FIELDS，顺序并不一样：
      beatfcn/model/features.py:46       tuple(sorted(BAND_EDGES)) -> 字母序（特征工程那一侧用）
      beatfcn/train/event_conditions.py:16 tuple(BAND_EDGES)      -> 插入序（训练喂模型用）
    模型的 control_ids.weight 第 i 行，对应的正是 encode_controls() 写进去的第 i 个字段，
    而 encode_controls 遍历的是 event_conditions 那份。所以**训练序才是模型口径**。

    如果拿 features.py 那份当基准，会得到 (4, 0, 7, 6, 2, 3, 1)，七行里每一行都
    读到别人的嵌入（density 拿 chord 的行那种），训出来七个旋钮完全不听话，
    而且这种错位不会报错、loss 也照常下降，只有训完试用才发现。
    正确映射是 (0, 1, 2, 3, 4, 6, 7)，只有 jack（源第 5 行）因已退役被丢掉。

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

from beatfcn.train.event_conditions import CONTROL_FIELDS as MODEL_CONTROL_FIELDS  # noqa: E402
from beatfcn.model.full_event_codec import FIELDS as EVENT_FIELDS  # noqa: E402
from beatfcn.model.full_event_codec import EventVocabulary  # noqa: E402
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel  # noqa: E402

SCHEMA = 'beatfcn.port.v1'

# 控制面板的行序两边**完全不同**，不能按位置硬搬：
#   noteforge: CONTROL_FIELDS = tuple(BAND_EDGES)         -> BAND_EDGES 的插入顺序
#   beatfcn:   train/event_conditions.py 的 CONTROL_FIELDS -> 也是 tuple(BAND_EDGES) 插入顺序
#              model/features.py 的 CONTROL_FIELDS 是 tuple(sorted(...)) 字母序，那是特征工程
#              那一侧用的，**不是**喂进模型 control_ids 的顺序。
# 关键：encode_controls() 按 event_conditions.CONTROL_FIELDS 逐行写控制向量，模型
# control_ids.weight 的每一行就是那个字段的嵌入，所以移植必须对齐**训练序**，
# 对齐 features.py 的字母序会让七行全部错位（density 读到 chord 的嵌入那种）。
NOTEFORGE_CONTROLS = ('density_notes_per_second', 'peak_notes_per_second', 'hold_head_ratio',
                      'hold_occupancy_ratio', 'chord_ratio', 'jack_ratio', 'flick_ratio', 'drag_ratio')
CONTROL_KEEP = tuple(NOTEFORGE_CONTROLS.index(f) for f in MODEL_CONTROL_FIELDS)

# 词表层的对齐规则。两边只差一个已退役的 mania 字段，其余字段名与尺寸完全一致，
# 所以不整层重初始化，而是按字段名把旧表的行搬到新表位置——除了 mania 的 81 格，
# 1904 格里的每一个都原样继承（前 4 个字段连同后面的 member/kind/tail 一起对齐）。
NOTEFORGE_FIELDS = ('group', 'head_block', 'head_cell', 'head_ms', 'mania', 'member',
                    'kind', 'tail_block', 'tail_cell', 'tail_ms')
VOCAB_LAYERS = ('token_embedding.weight', 'output.weight')
FIELD_LAYER = 'field_embedding.weight'


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


def noteforge_sizes(vocab_dict):
    """noteforge 的词表分段（含已退役的 mania 81 格）。"""
    blocks = vocab_dict['max_duration_ms'] // 2000 + 1
    return (2, blocks + vocab_dict['max_copy_times'], 50, 40, 81, 2, 4, blocks, 50, 40)


def remap_rows(old, src_fields, src_sizes, dst_fields, dst_sizes, pad=2):
    """按字段名把旧表的行搬到新表位置。

    目标表没有的字段（mania）丢弃；词表层末尾有 2 格 padding，field_embedding
    没有（pad=0，行数就是字段个数）。
    """
    src_off, off = {}, 0
    for f, s in zip(src_fields, src_sizes):
        src_off[f] = off
        off += s
    new = torch.empty(sum(dst_sizes) + pad, old.shape[1], dtype=old.dtype)
    dst_off = 0
    for f, s in zip(dst_fields, dst_sizes):
        if f not in src_off:
            raise SystemExit(f'目标字段 {f} 在源词表里不存在，移植规则需要更新')
        new[dst_off:dst_off + s] = old[src_off[f]:src_off[f] + s]
        dst_off += s
    if pad:
        new[sum(dst_sizes):] = old[sum(src_sizes):sum(src_sizes) + pad]
    return new


def port(saved, seed=20261116, controls=7):
    """抄一致层 + 重排控制行 + 按字段名对齐词表层；返回 (state, config, vocabulary, report)。"""
    shape_bad, reord_bad, only_weight, only_model, conf, vocab = compare(saved, controls)
    if only_weight or only_model:
        raise SystemExit('两边存在单边参数，移植脚本只处理已知差异，请先人工核对：'
                         f'only_weight={only_weight[:5]} only_model={only_model[:5]}')
    src_sizes = noteforge_sizes(dict(saved['vocabulary']))
    state, copied, reordered, remapped = {}, 0, [], []
    for key, value in saved['model'].items():
        if key in reord_bad:
            state[key] = value[list(CONTROL_KEEP)].clone()
            reordered.append(key)
        elif key in shape_bad:
            if key in VOCAB_LAYERS:
                state[key] = remap_rows(value, NOTEFORGE_FIELDS, src_sizes,
                                        EVENT_FIELDS, tuple(vocab.sizes))
            elif key == FIELD_LAYER:
                state[key] = remap_rows(value, NOTEFORGE_FIELDS, (1,) * len(NOTEFORGE_FIELDS),
                                        EVENT_FIELDS, (1,) * len(EVENT_FIELDS), pad=0)
            else:
                raise SystemExit(f'未知的形状不符层 {key}，移植规则需要更新')
            remapped.append(f'{key} {tuple(value.shape)} -> {tuple(state[key].shape)}')
        else:
            # 源权重全是 float32，直接引用 mmap 张量，torch.save 时按需读入，
            # 不复制一份等大的内存。
            state[key] = value
            copied += 1
    report = dict(copied_layers=copied, reord_layers=reordered, remapped_layers=remapped,
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
                               remapped_layers=report['remapped_layers'],
                               reordered_layers=report['reord_layers'],
                               controls=args.controls,
                               note='词表层按字段名对齐：除已退役的 mania 81 格外全部继承'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f'已写 {args.out}  参数 {out["parameters"]}')
    print(f'按字段对齐 {len(report["remapped_layers"])} 层 / 重排 {len(report["reord_layers"])} 层')
    for k in report['remapped_layers']:
        print(f'  {k}')
    for k in report['reord_layers']:
        print(f'  {k}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
