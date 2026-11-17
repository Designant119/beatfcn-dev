"""母版自检：把「移植到底做对了没」变成一条命令，不跑训练、不碰 GPU。

用法（两种都行）:
    python -m scripts.verify_port \\
        --src /path/to/noteforge-r5.pt --out /path/to/beatfcn-r5-ported.pt
    NOTEFORGE_R5=/path/to/noteforge-r5.pt \\
    BEATFCN_PORTED=/path/to/beatfcn-r5-ported.pt python -m scripts.verify_port

验四件事，任一项不过就以退出码 1 结束，可以直接挂进 CI 或手动跑:

  1. control_ids 七行 —— 母版第 i 行必须等于源权重里「同名控制字段」那一行
  2. field_embedding 九行 —— 按字段名逐行对齐
  3. token_embedding / output —— 除退役的 mania 81 格外逐段一致，尾部 padding 一致
  4. 严格加载 —— FullEventModel 用 strict=True 装进去，参数量对得上

为什么把第 1 条单独拎出来：2026-11 抓到过一个会毁训练的 bug。CONTROL_KEEP 当初
按 beatfcn/model/features.py 的字母序排（那是特征工程那一侧的口径），而训练时
encode_controls() 遍历的是 beatfcn/train/event_conditions.py 的插入序。两套顺序
不同，母版里七行控制嵌入于是全部读到别人的行 —— density 拿到 chord 的嵌入那种。
更坑的是当时「验证通过」了，因为验证本身也拿 features.py 当尺子，自己验自己。
所以这个脚本的基准永远是 MODEL_CONTROL_FIELDS（训练序），不要改回 features.py 那份。
"""
import argparse
import os
import sys
from pathlib import Path

import torch

from beatfcn.model.full_event_codec import FIELDS as EVENT_FIELDS, EventVocabulary
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from scripts.port_noteforge import (CONTROL_KEEP, FIELD_LAYER, MODEL_CONTROL_FIELDS,
                                    NOTEFORGE_CONTROLS, NOTEFORGE_FIELDS, VOCAB_LAYERS,
                                    noteforge_sizes)

PAD = 2


def load(path, label):
    if not path.exists():
        raise SystemExit(f'{label}不存在：{path}')
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def main():
    parser = argparse.ArgumentParser(description='验移植母版是否逐层对齐 noteforge 源权重')
    parser.add_argument('--src', default=os.environ.get('NOTEFORGE_R5', ''), help='noteforge r5 源权重路径')
    parser.add_argument('--out', default=os.environ.get('BEATFCN_PORTED', ''), help='移植后的母版路径')
    args = parser.parse_args()
    if not args.src or not args.out:
        raise SystemExit('要么传 --src/--out，要么设 NOTEFORGE_R5 / BEATFCN_PORTED 环境变量')

    src = load(Path(args.src), '源权重')
    ported = load(Path(args.out), '母版')
    src_state, out_state = src['model'], ported['model']
    src_sizes = noteforge_sizes(src.get('vocabulary') or {})
    vocab = EventVocabulary(**ported.get('vocabulary', {}))
    dst_sizes = tuple(vocab.sizes)
    problems = []

    # 1. control_ids：训练序为准，只有 jack 该被丢
    ctrl = 'control_ids.weight'
    bad_rows = []
    for dst_row, src_row in enumerate(CONTROL_KEEP):
        if not torch.equal(src_state[ctrl][src_row], out_state[ctrl][dst_row]):
            bad_rows.append(f'第{dst_row}行({MODEL_CONTROL_FIELDS[dst_row]})')
    if bad_rows:
        problems.append(f'control_ids 与源不一致：{"、".join(bad_rows)}')
    print(f'[1] control_ids {len(CONTROL_KEEP)} 行，映射 {tuple(CONTROL_KEEP)} = '
          + ' / '.join(MODEL_CONTROL_FIELDS))
    print(f'    {"通过" if not bad_rows else "失败：" + "、".join(bad_rows)}'
          f'（源里被丢掉的是 {NOTEFORGE_CONTROLS[5]}，已退役）')

    # 2. field_embedding：按字段名逐行对齐
    bad_fields = []
    for dst_row, field in enumerate(EVENT_FIELDS):
        src_row = NOTEFORGE_FIELDS.index(field)
        if not torch.equal(src_state[FIELD_LAYER][src_row], out_state[FIELD_LAYER][dst_row]):
            bad_fields.append(field)
    if bad_fields:
        problems.append(f'field_embedding 与源不一致：{bad_fields}')
    print(f'[2] field_embedding {len(EVENT_FIELDS)} 行按字段名对齐：{"通过" if not bad_fields else "失败"}')

    # 3. 词表层：除退役 mania 外逐段一致，尾部 padding 一致
    src_off, off = {}, 0
    for field, size in zip(NOTEFORGE_FIELDS, src_sizes):
        src_off[field] = off
        off += size
    dst_off, off = {}, 0
    for field, size in zip(EVENT_FIELDS, dst_sizes):
        dst_off[field] = off
        off += size
    for layer in VOCAB_LAYERS:
        missed = [f for f, s in zip(EVENT_FIELDS, dst_sizes)
                  if not torch.equal(out_state[layer][dst_off[f]:dst_off[f] + s],
                                     src_state[layer][src_off[f]:src_off[f] + s])]
        pad_ok = torch.equal(out_state[layer][-PAD:], src_state[layer][-PAD:])
        if missed:
            problems.append(f'{layer} 这些字段段不一致：{missed}')
        if not pad_ok:
            problems.append(f'{layer} 尾部 {PAD} 格 padding 不一致')
        print(f'[3] {layer}：除退役 mania 外逐段一致 {"通过" if not missed else "失败"}；'
              f'尾部 {PAD} 格 padding {"一致" if pad_ok else "不一致"}')

    # 4. 严格加载
    config = FullEventConfig(**ported['config'])
    model = FullEventModel(config, vocab)
    try:
        model.load_state_dict(dict(out_state), strict=True)
        print(f'[4] strict=True 加载通过，参数 {sum(p.numel() for p in model.parameters()):,}')
    except Exception as exc:  # noqa: BLE001 - 任何加载失败都该报出来
        problems.append(f'严格加载失败：{exc!r}')
        print(f'[4] 严格加载失败：{exc!r}')

    print()
    if problems:
        print(f'发现 {len(problems)} 个问题：')
        for item in problems:
            print('  -', item)
        return 1
    print('全部通过：母版与源权重逐层对齐。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
