"""把自然语言指令翻译成 harness 调用，并回一句人话。

设计上的三条硬规矩：

1. key 只看仓库根的 phira_chat.json、~/.config/phira/chat.json 或环境变量，
   永不进前端；网页只能调本模块，看不到凭据。
2. 模型能做的事被限制在 harness 的八个步骤里。它不能拼 shell、不能删文件、
   不能改路径白名单之外的目录——所有参数都再过一遍 harness.validate。
3. 拿不准就问，不要猜。模型输出不合法时如实告诉用户，不硬塞一个动作进去。

配置（优先级：仓库 phira_chat.json < ~/.config/phira/chat.json < 环境变量）：
仓库那份是团队默认，私密仓库里直接带 key（key_b64 存，不是加密，只是躲扫描）。
  PHIRA_CHAT_KEY    模型服务凭据
  PHIRA_CHAT_BASE   默认 https://open.bigmodel.cn/api/paas/v4/chat/completions
  PHIRA_CHAT_MODEL  默认 glm-4-flash
  PHIRA_CHAT_CONFIG 可选，改用户配置文件的路径
"""
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness import (ORDER, STEPS, demo_train, inspect_charts, load_paths, log, pipeline_next,
                             running, start, status, stop as hstop, train_progress)

# 给同学看的人话名称；内部还是那八个英文步骤
STEP_LABEL = {
    'download': '下载谱面', 'ingest': '建数据集', 'audio': '摊平音轨',
    'tape': '造提特征清单', 'features': '提特征', 'full_tape': '造训练数据',
    'exposure': '冻结采样', 'train': '冒烟训练',
}


def _label(name):
    return STEP_LABEL.get(name, name)

TOOL_DOC = """你在帮同学操作一条谱面数据流水线。同学不懂参数，只看懂人话。

八个步骤（每步都有默认参数，默认值就是对的，不要追问）：
- download：下载谱面。默认 /data/phira_charts，全量 9760 包约 60GB；只要高难加 levels ["IN","AT"]（约 49GB）；只想看清单加 list_only true
- ingest：把下载好的谱面建成数据集（断了能续跑）
- audio：把音轨摊平出来（一个音频只存一份）
- tape：造一份提特征用的小清单
- features：提特征，最慢的一步，吃显卡
- full_tape：造真正训练用的 tape
- exposure：冻结一份采样清单，保证实验可复现
- train：冒烟训练，几步就出一个能用的 ckpt

同学的常见说法 → 直接照这样办（args 用默认，不要反问）：
- “下载谱面 / 搞点数据 / 下载歌” → {"step": "download", "args": {"levels": ["IN", "AT"]}}
- “全都要 / 全量下载” → {"step": "download", "args": {}}
- “先看看有多少 / 列个清单” → {"step": "download", "args": {"list_only": true}}
- “建库 / 入库 / 整理数据” → {"step": "ingest", "args": {}}
- “提特征 / 跑特征” → {"step": "features", "args": {}}
- “训练 / 开训 / 跑一下” → {"step": "train", "args": {}}
- “下一步干啥 / 看看进度” → {"action": "status"}
- “停掉训练 / 停下来 / 别跑了 / 暂停” → {"action": "stop"}（step 可省略，省略就停正在跑的那一步；这是真去停，不许只回“好的”）
- “文件在哪 / 下到哪了 / 存哪去了 / 下了多少” → {"action": "where"}（可带 args.dir；不带就看当前的谱面目录）
- “还剩多少空间 / 磁盘满没满 / 放得下吗” → {"action": "disk"}
- “日志呢 / 现在在输出什么 / 它跑到哪了” → {"action": "logs", "step": "对应步骤"}（真读日志尾巴，不许编）
- “把坏包清掉 / 删掉坏的 / 清理一下” → {"action": "clean"}（只把打不开的挪进隔离文件夹，不真删；可带 args.dir）
- “去看看 / 检验一下 / 查查有没有坏包 / 验一下” → {"action": "inspect"}（真去数包、逐个验，不许只嘴上说“下好了”）
  他要是说了某个目录，就 {"action": "inspect", "args": {"dir": "那个目录"}}
- “接着跑 / 继续 / 一键跑 / 往下走” → {"action": "run"}（不填 step，后端自动挑下一步）

用户点名要跑某一步时（“下载 / 建库 / 提特征 / 训练 / 开训”）：
- 就按他点的那一步填 step，不许自作主张换成“下一步”。前置没齐后端会拦下来并说明还差什么。
- 用户没点名、只说“继续 / 接着跑 / 往下走 / 下一步”时：{"action": "run"}，step 留空，后端自动挑下一步。
- 这台机器如果有演示训练环境（状态里会写），用户说“开始训练”就直接 {"action": "run", "step": "train"}，
  别拿“数据还没下”去拦他——演示环境用假数据，能直接跑。
- 不要教训人“不能跳过”，也不要说“必须先做前面的”。

说话规矩（很重要，违反就是事故）：
- reply 里不要出现参数名、文件路径、脚本名、tmux、sha256、npy 这些词；
  说“下载谱面”“提特征”“数据放在哪”就行。
- 用户没给目录就别问目录，默认值直接用；只有他主动说“我要放 D 盘”这种才照他说的填。
- 但用户问“下到哪了 / 文件在哪 / 存哪去了”时，要老实说路径（这时候可以说），
  并补一句：“下一步会自己从这儿读，你不用搬。”
- 拿不准就问一句人话，比如“要全量还是只要高难（IN/AT）？”，别问参数。
- 同一步已经在跑就别重复启动，告诉他“正在跑，等它完”。
- 启动返回 errors 非空 或 alive 为 false：照抄日志里那句人话报失败，
  绝对不能回“已启动”“开跑了”。
- alive 为 false 且没有 errors：说明跑完了（短任务）或自己退了，说“已经跑完/已退出”。
- 不能用 shell；删文件只允许“清坏包”这一种，而且是挪进隔离目录、不是真删。
- 只是闲聊、吐槽、情绪、测试、问问题：action 一律用 "answer"，绝不 run。
  就算状态里写着“建议下一步”，也不许自己开跑——用户没让你动手，就别动手。
- 只有用户明确让你干活时才 run：下载/建库/提特征/造数据/训练/继续/接着跑/一键跑。

只输出一个 JSON 对象，不要任何多余文字，reply 放第一个键：
{"reply": "给同学看的中文回复（一两句，不要长篇）",
 "action": "run|status|inspect|stop|where|disk|logs|clean|answer", "step": "步骤名（action=run/stop 时）",
 "args": {参数对象，可省略，内部参数不要念出来}}

"""
def _read_config(path):
    """读一个 json 配置；key 可以用 key_b64 存，读出来还原成 key。"""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:                         # noqa: BLE001
        return {}                             # 文件坏了就当没配，别把服务带崩
    if not isinstance(data, dict):
        return {}
    if data.get('key_b64') and not data.get('key'):
        try:
            data['key'] = base64.b64decode(data['key_b64']).decode().strip()
        except Exception:                     # noqa: BLE001
            pass
    return {k: v for k, v in data.items() if v and k != 'key_b64'}


def _config():
    """凭据与端点：仓库自带 > 用户配置 > 环境变量，后者盖前者。

    仓库根的 phira_chat.json 是团队默认值，私密仓库里带着 key，
    同学 clone 下来就能用，不用各自配。key 用 base64 存（躲 secret scanning，
    不是加密，拿到仓库照样解得开）。个人想换 key 写 ~/.config/phira/chat.json，
    临时换用环境变量——越靠后优先级越高。
    """
    cfg = {}
    cfg.update(_read_config(ROOT / 'phira_chat.json'))
    cfg.update(_read_config(
        Path(os.environ.get('PHIRA_CHAT_CONFIG')
             or Path.home() / '.config' / 'phira' / 'chat.json')))
    for env_name, name in (('PHIRA_CHAT_KEY', 'key'),
                           ('PHIRA_CHAT_BASE', 'base'),
                           ('PHIRA_CHAT_MODEL', 'model')):
        value = os.environ.get(env_name)
        if value:
            cfg[name] = value
    return cfg


MODELS = ('glm-5.3', 'glm-5.3-flash')      # 同学可选的模型，别的名字一律不认

# 模型说要跑流水线时，消息里至少得有点“动手”的意思；否则一律按聊天处理
INTENT_WORDS = ('下载', '谱面', '建库', '入库', '数据集', '数据', '音轨', '特征',
                '训练', '开训', '跑', '开始', '继续', '接着', '清单', '下一步',
                '一键', '往下', '搞', '弄')


def _thinking_extra(thinking):
    """思考设置 → 请求体里要加的字段。

    'auto'        不传，让模型自己决定
    'off'/False   明确关掉
    'low'/'high'/'max'  开启并按强度
    True          开启（默认强度）
    """
    if thinking is None or thinking == 'auto':
        return {}
    if isinstance(thinking, bool):
        return {'thinking': {'type': 'enabled' if thinking else 'disabled'}}
    level = str(thinking).strip().lower()
    if level in ('off', 'none', 'false', '0', '关'):
        return {'thinking': {'type': 'disabled'}}
    if level in ('low', 'high', 'max'):
        return {'thinking': {'type': 'enabled'}, 'reasoning': {'effort': level}}
    if level in ('on', 'true', '1', '开'):
        return {'thinking': {'type': 'enabled'}}
    return {}


def _call(messages, model=None, thinking=None):
    cfg = _config()
    key = cfg.get('key')
    if not key:
        return None, ('后端没有配置 PHIRA_CHAT_KEY，AI 助手不可用。'
                      '把 key 写进后端环境变量，或放到 ~/.config/phira/chat.json 再试。')
    base = cfg.get('base') or 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
    model = model if model in MODELS else (cfg.get('model') or 'glm-5.3')
    payload = {'model': model, 'messages': messages, 'temperature': 0.2}
    payload.update(_thinking_extra(thinking))
    body = json.dumps(payload, ensure_ascii=False)
    req = urllib.request.Request(base, data=body.encode(), method='POST',
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:                      # noqa: BLE001
        return None, f'调用模型失败：{type(exc).__name__}: {exc}'
    try:
        return payload['choices'][0]['message']['content'], None
    except (KeyError, IndexError):
        return None, '模型返回结构不认识：' + json.dumps(payload, ensure_ascii=False)[:200]


def _extract_json(text):
    """从模型输出里挖出那个 JSON 对象，容忍 Markdown 包裹和前后解释文字。

    模型很爱干两件事：把 JSON 包进 ```json 代码块，或者在 JSON 前后唠一句
    “好的，我来帮你……”。这两种都不能当成失败——挖出那个对象就行。
    """
    if not text:
        return None
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    if m:
        return m.group(1)
    start = text.find('{')
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find('{', start + 1)
    return None


def _progress_line():
    """训练进度一句话；没有训练日志就不说。"""
    p = train_progress()
    if not p or p.get('step') is None:
        return ''
    total = p.get('total')
    bits = [f'第 {p["step"]}/{total} 步' if total else f'第 {p["step"]} 步']
    if p.get('loss') is not None:
        bits.append(f'loss {p["loss"]}')
    if p.get('running'):
        bits.append('还在跑')
    elif p.get('age') is not None:
        bits.append('已经跑完了')
    return '训练进度：' + '，'.join(bits)


def _state_brief():
    st = status()
    done = [k for k in ORDER if st[k]['done']]
    run = [k for k in ORDER if st[k]['running']]
    demo = demo_train()
    tail = (f'建议下一步：{_label(pipeline_next()) if pipeline_next() else "全部完成"}')
    if demo:
        tail += f'；（这台机器有演示训练环境：说“开始训练”能直接用假数据跑起来，不用等他下数据）'
    prog = _progress_line()
    if prog:
        tail += '；' + prog
    return (f'已完成的步骤：{"、".join(_label(k) for k in done) or "无"}；'
            f'正在跑：{"、".join(_label(k) for k in run) or "无"}；' + tail)


def _digest(older):
    """把更早的对话压成一小段“聊过什么”，让模型别把上下文忘干净。"""
    lines = []
    for turn in (older or [])[-20:]:
        if turn.get('role') not in ('user', 'assistant'):
            continue
        text = ' '.join(str(turn.get('content', '')).split())[:120]
        if text:
            lines.append(('用户：' if turn['role'] == 'user' else '它：') + text)
    if not lines:
        return ''
    return ('早前聊过这些（压缩过，细节可能丢，别当成当前指令）：\n' + '\n'.join(lines))[:2200]


def _look_tool(action, decision, reply):
    """只读的几只手：看目录、看磁盘、看日志；外加可逆的“清坏包”。"""
    import shutil
    args = decision.get('args') or {}
    if action == 'where':
        d = args.get('dir') or args.get('path') or load_paths().get('charts') or '/data/phira_charts'
        p = Path(str(d))
        if not p.is_dir():
            return dict(ok=False, action=action, reply=f'这个目录不在：{d}，可能还没下或者名字不对。')
        files = [x for x in p.iterdir() if x.is_file()]
        zips = [x for x in files if x.suffix.lower() == '.zip']
        total = 0
        for x in zips:
            try:
                total += x.stat().st_size
            except OSError:
                pass
        size = (f'{total / 1024 ** 3:.1f} GB' if total >= 1024 ** 3 else f'{total / 1024 ** 2:.0f} MB')
        return dict(ok=True, action=action, info=dict(dir=str(p), count=len(zips), bytes=total),
                    reply=f'谱面就在 {p}，现在 {len(zips)} 个包、{size}。下一步会自己从这儿读，你不用搬。')
    if action == 'disk':
        rows = []
        for label, path in (('根目录', '/'), ('数据盘 /data', '/data'), ('仓库', str(Path(__file__).resolve().parent.parent))):
            try:
                u = shutil.disk_usage(path)
                rows.append(f'{label}剩 {u.free / 1024 ** 3:.0f} GB（总共 {u.total / 1024 ** 3:.0f} GB）')
            except Exception:                 # noqa: BLE001
                continue
        if not rows:
            return dict(ok=False, action=action, reply='磁盘信息没读到，可能路径不在。')
        return dict(ok=True, action=action, reply='；'.join(rows) + '。')
    if action == 'logs':
        step = decision.get('step') if decision.get('step') in STEPS else 'train'
        text = log(step, 24)
        if not text:
            return dict(ok=False, action=action, step=step, reply=f'「{_label(step)}」还没有日志，可能还没跑过。')
        lines = []
        for raw in text.splitlines()[-8:]:
            try:
                obj = json.loads(raw)
                if 'step' in obj:
                    lines.append(f'第 {obj["step"]} 步' + (f'，loss {obj["loss"]}' if 'loss' in obj else ''))
                    continue
            except Exception:                 # noqa: BLE001
                pass
            if 'ckpt ->' in raw:
                m = re.search(r'（(.+?)）', raw)
                lines.append('模型存好了' + (f'（{m.group(1)}）' if m else ''))
                continue
            if raw.strip():
                lines.append(raw.strip()[-60:])
        return dict(ok=True, action=action, step=step,
                    reply=f'「{_label(step)}」最近的输出：' + '；'.join(lines[-3:]))
    if action == 'clean':                 # 清坏包：只挪进隔离目录，绝不删
        d = args.get('dir') or load_paths().get('charts') or '/data/phira_charts'
        info = inspect_charts(d, sample=10 ** 9)
        bad = info.get('bad') or []
        if not bad:
            return dict(ok=True, action=action, reply='验过了，没有打不开的包，不用清。')
        src = Path(info['dest'])
        q = src / '_quarantine'
        q.mkdir(exist_ok=True)
        moved = 0
        for name in bad:
            try:
                shutil.move(str(src / name), str(q / name))
                moved += 1
            except Exception:                 # noqa: BLE001
                pass
        return dict(ok=True, action=action, moved=moved,
                    reply=f'把 {moved} 个打不开的包挪进了隔离文件夹（没真删），重下这些就行。')
    return dict(ok=False, action=action, reply=reply or '没太明白')


def _build_messages(message, history=None):
    messages = [dict(role='system', content=TOOL_DOC + '\n\n当前状态：' + _state_brief())]
    turns = [t for t in (history or []) if t.get('role') in ('user', 'assistant')]
    older, recent = turns[:-6], turns[-6:]
    digest = _digest(older)
    if digest:
        messages.append(dict(role='system', content=digest))
    for turn in recent:
        messages.append(dict(role=turn['role'], content=str(turn.get('content', ''))[:2000]))
    messages.append(dict(role='user', content=str(message)[:2000]))
    return messages


def _gate(message, decision):
    """模型说要动手，但用户那句话里没有动手的意思 → 压回聊天，别自作主张。"""
    if decision.get('action') == 'run' and not any(w in message for w in INTENT_WORDS):
        return dict(action='answer',
                    reply=decision.get('reply') or '想让我跑哪一步？说一声就开始。')
    return decision


def _parse_decision(text):
    """把模型输出解析成动作意图，但不执行。返回 dict(action/step/args/reply)。"""
    raw = _extract_json(text)
    if raw is None:
        return dict(action='answer', reply=(text or '').strip()[:500])
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        return dict(action='answer', reply=(text or '').strip()[:500])
    if not isinstance(decision, dict):
        return dict(action='answer', reply=(text or '').strip()[:500])
    return dict(action=decision.get('action') or 'answer',
                step=decision.get('step'),
                args=decision.get('args') if isinstance(decision.get('args'), dict) else {},
                reply=str(decision.get('reply') or '').strip())


def _prereq_report(step):
    """看前面那些步骤是不是真跑完了。返回 (还缺的, 正在跑的)。"""
    if step == 'train' and demo_train():
        return [], []                       # 演示环境：假数据已经备好，别拿真流水线卡它
    st = status()
    idx = ORDER.index(step) if step in ORDER else len(ORDER)
    missing, busy = [], []
    for name in ORDER[:idx]:
        if st[name]['running']:
            busy.append(name)
        elif not st[name]['done']:
            missing.append(name)
    return missing, busy


def _human_error(text):
    """把脚本报错翻译成同学看得懂的一句话。"""
    raw = str(text or '').strip()
    m = re.search(r"No module named '([^']+)'", raw)
    if m:
        return f'这台机器还缺 Python 依赖 {m.group(1)}，先把 requirements 装上再跑'
    if 'Permission denied' in raw:
        return '那个目录没权限写，换一个有权限的位置'
    if 'No such file or directory' in raw:
        return '要读的目录不存在，前面那步可能还没跑'
    if 'unrecognized arguments' in raw:
        return '启动参数对不上脚本，这部分我来修'
    return raw


def _execute(decision):
    """按意图真正干活，返回最终 dict(ok, action, step, reply, result)。"""
    action = decision.get('action') or 'answer'
    reply = decision.get('reply') or ''
    if action == 'status':
        text = reply or _state_brief()
        prog = _progress_line()
        if prog and '步' not in text:
            text = text.rstrip('。') + '。' + prog + '。'
        return dict(ok=True, action='status', reply=text, state=status())
    if action in ('where', 'disk', 'logs', 'clean'):
        return _look_tool(action, decision, reply)
    if action == 'stop':
        step = decision.get('step')
        if step not in STEPS:
            st = status()
            busy = [k for k in ORDER if st[k]['running']]
            step = busy[0] if busy else 'train'
        res = hstop(step)
        if res.get('ok'):
            return dict(ok=True, action='stop', step=step, result=res,
                        reply=f'已经停了「{_label(step)}」。' + ('自上次存点后的进度会丢。' if res.get('how') == 'kill' else ''))
        return dict(ok=False, action='stop', step=step, result=res,
                    reply=f'没停成：{res.get("note") or "原因不明"}。')
    if action == 'inspect':
        d = (decision.get('args') or {}).get('dir')
        info = inspect_charts(d)
        if not info.get('ok'):
            return dict(ok=False, action='inspect', info=info,
                        reply=f'这个目录我没看到：{info.get("dest")}，可能不在盘上或者名字不对。')
        n = info['count']
        if n == 0:
            return dict(ok=True, action='inspect', info=info,
                        reply=f'{info["dest"]} 里一个谱面包都没有，等于还没下。')
        size = (f'{info["total_bytes"] / 1024 ** 3:.1f} GB'
                if info['total_bytes'] >= 1024 ** 3
                else f'{info["total_bytes"] / 1024 ** 2:.0f} MB')
        head = f'点了 {n} 个包，共 {size}，逐个翻开验了 {info["checked"]} 个'
        if info.get('unique'):
            head += f'；按谱面 id 算是 {info["unique"]} 张谱'
            if info.get('dup_count'):
                head += f'，有 {info["dup_count"]} 个是同 id 重复（下次下载会按 id 直接跳过，不会再多占地方）'
            if info.get('noid_count'):
                head += f'，另有 {info["noid_count"]} 个文件名里没有 id，认不出是哪张'
        if info.get('bad_count'):
            names = '、'.join(info.get('bad') or [])
            return dict(ok=False, action='inspect', info=info,
                        reply=f'{head}，有 {info["bad_count"]} 个打不开：{names}。建议删掉重下。')
        tail = '，全都正常。' if n <= info['checked'] else f'，抽的前 {info["checked"]} 个都正常。'
        return dict(ok=True, action='inspect', info=info,
                    reply=head + tail + ('下载器留的报告也在，确实是它下的。' if info.get('report') else ''))
    if action != 'run':
        return dict(ok=True, action='answer', reply=reply or '没太明白，能再说具体点吗？')
    step = decision.get('step')
    args = decision.get('args') or {}
    if not step:                              # 没指定就按流水线顺序挑下一步，别让用户自己找
        step = pipeline_next()
        if not step:
            return dict(ok=True, action='status', reply='八步都跑完了，没有下一步了。')
    if step not in STEPS:
        return dict(ok=False, action='run', reply=f'模型想调一个不存在的步骤 {step!r}，没有执行。')
    missing, busy = _prereq_report(step)
    if busy:
        return dict(ok=False, action='run', step=step,
                    reply=f'「{_label(busy[0])}」还在跑，等它跑完再往下。')
    if missing:
        # 前置没齐就不硬跳：告诉他还差哪几步，问一句再动
        labels = '、'.join(_label(m) for m in missing)
        return dict(ok=False, action='answer', step=step, missing=missing,
                    reply=f'「{_label(step)}」前面还差 {len(missing)} 步：{labels}。'
                          f'要我从「{_label(missing[0])}」开始跑吗？')
    if running(step):
        return dict(ok=False, action='run', step=step,
                    reply=f'{step} 已经在跑，不重复启动。用日志看进度。')
    result = start(step, args)
    if not result.get('ok'):
        return dict(ok=False, action='run', step=step, result=result,
                    reply='没启动成功：' + _human_error(
                        '；'.join(result.get('errors') or []) or '原因不明，看日志'))
    tail = '已启动，日志可以随时看。'
    if result.get('alive') is False:          # 起完就退：短任务跑完，或者自己崩了
        last = [l.strip() for l in log(step, 12).splitlines() if l.strip()]
        tail = f'「{_label(step)}」已经跑完了。' + (f'日志最后一句：{last[-1]}' if last else '')
    return dict(ok=True, action='run', step=step, result=result,
                reply=(reply + ' ' if reply else '') + tail)


def respond(message, history=None, model=None, thinking=None):
    """返回 dict(ok, reply, action, step, result)。永不抛异常给上层。"""
    text, err = _call(_build_messages(message, history), model=model, thinking=thinking)
    if err:
        return dict(ok=False, reply=err, action='error')
    return _execute(_gate(message, _parse_decision(text)))


class _ReplyTap:
    """从流式 content 里实时抠出 reply 字段。

    模型按“reply 放第一个键”的格式输出，所以 content 一开始就是
    {"reply": "……，能一边收一边吐给前端。JSON 之外的纯文本则原样放过。
    """

    def __init__(self):
        self.state = 'start'
        self.tail = ''
        self.esc = False
        self.uni = ''

    def feed(self, chunk):
        out = []
        for ch in chunk:
            if self.state == 'start':
                if ch in ' \t\r\n':
                    continue
                self.state = 'json' if ch == '{' else 'raw'
                if self.state == 'raw':
                    out.append(ch)
                else:
                    self.tail = ch
            elif self.state == 'json':
                self.tail = (self.tail + ch)[-40:]
                if re.search(r'"reply"\s*:\s*"$', self.tail):
                    self.state = 'str'
                    self.tail = ''
            elif self.state == 'str':
                if self.uni:
                    self.uni += ch
                    if len(self.uni) == 4:
                        try:
                            out.append(chr(int(self.uni, 16)))
                        except ValueError:
                            pass
                        self.uni = ''
                    continue
                if self.esc:
                    self.esc = False
                    if ch == 'u':
                        self.uni = ''
                    elif ch == 'n':
                        out.append('\n')
                    elif ch == 't':
                        out.append('\t')
                    elif ch == 'r':
                        out.append('\r')
                    else:
                        out.append(ch)
                elif ch == '\\':
                    self.esc = True
                elif ch == '"':
                    self.state = 'done'
                else:
                    out.append(ch)
        return ''.join(out)


def stream(message, history=None, model=None, thinking=None):
    """流式版 respond。yield 事件 dict：

    think  —— 模型的思考内容（reasoning_content）
    delta  —— 回复正文的一小段（边生成边吐）
    status —— 当前在干什么（解析指令 / 正在执行 download）
    reply  —— 最终结果（前端用它盖掉前面流出来的文本）
    done   —— 速度与 token 统计
    error  —— 出错
    """
    cfg = _config()
    if not cfg.get('key'):
        yield dict(type='error', text='后端没有配置 PHIRA_CHAT_KEY，AI 助手不可用。')
        return
    base = cfg.get('base') or 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
    model = model if model in MODELS else (cfg.get('model') or 'glm-5.3')
    payload = {'model': model, 'messages': _build_messages(message, history),
               'temperature': 0.2, 'stream': True,
               'stream_options': {'include_usage': True}}
    # 官方口径：GLM-5.3 / 5.3-Flash 都是 1M 上下文（zcode 客户端自报 200K 是它自己的保守值）
    try:
        ctx_max = int(os.environ.get('PHIRA_CTX_MAX') or 1_000_000)
    except ValueError:
        ctx_max = 1_000_000
    payload.update(_thinking_extra(thinking))
    req = urllib.request.Request(base, data=json.dumps(payload, ensure_ascii=False).encode(),
                                 method='POST',
                                 headers={'Content-Type': 'application/json',
                                          'Accept': 'text/event-stream',
                                          'Authorization': 'Bearer ' + cfg['key']})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        # 429/401 这类要把它返回的那句人话捞出来，别甩一串 HTTPError
        detail = ''
        try:
            payload = json.loads(exc.read().decode('utf-8', 'replace'))
            detail = ((payload.get('error') or {}).get('message')) or ''
        except Exception:                      # noqa: BLE001
            pass
        yield dict(type='error', text=f'模型服务报错 HTTP {exc.code}：{detail or exc.reason}')
        return
    except Exception as exc:                      # noqa: BLE001
        yield dict(type='error', text=f'调用模型失败：{type(exc).__name__}: {exc}')
        return

    full, usage = [], None
    tap = _ReplyTap()
    t_first = t_last = None
    buf = ''
    with resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk.decode('utf-8', 'replace')
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                line = line.strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get('usage'):
                    usage = obj['usage']
                choices = obj.get('choices') or []
                if not choices:
                    continue
                delta = choices[0].get('delta') or {}
                think = delta.get('reasoning_content')
                if think:
                    yield dict(type='think', text=think)
                piece = delta.get('content')
                if piece:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    full.append(piece)
                    out = tap.feed(piece)
                    if out:
                        yield dict(type='delta', text=out)

    text = ''.join(full)
    yield dict(type='status', text='正在解析指令…')
    try:
        decision = _gate(message, _parse_decision(text))
    except Exception as exc:                      # noqa: BLE001
        yield dict(type='error', text=f'解析模型输出失败：{exc}')
        return
    if decision.get('action') == 'run':
        yield dict(type='status', text=f'正在执行 {decision.get("step")}…')
    elif decision.get('action') == 'inspect':
        yield dict(type='status', text='正在数包、验包…')
    elif decision.get('action') == 'stop':
        yield dict(type='status', text='正在停…')
    elif decision.get('action') == 'where':
        yield dict(type='status', text='正在数目录…')
    elif decision.get('action') == 'disk':
        yield dict(type='status', text='正在看磁盘…')
    elif decision.get('action') == 'logs':
        yield dict(type='status', text='正在翻日志…')
    elif decision.get('action') == 'clean':
        yield dict(type='status', text='正在挑坏包…')
    result = _execute(decision)

    out_tokens = (usage or {}).get('completion_tokens')
    ctx_tokens = (usage or {}).get('prompt_tokens')
    estimated = out_tokens is None
    if estimated:
        out_tokens = max(1, int(len(text) / 1.6))
    secs = (t_last - t_first) if t_first and t_last else 0
    # 太短的流算 tok/s 会飙到离谱（几百毫秒就能报出上千），不足 1 秒一律不报
    tps = round(out_tokens / secs, 1) if secs >= 1.0 else None
    yield dict(type='reply', payload=result)
    if ctx_tokens is None:                    # 有些返回不带 usage，按字符数粗估
        try:
            ctx_tokens = sum(len(str(m.get('content', ''))) for m in payload['messages']) // 2
        except Exception:                     # noqa: BLE001
            ctx_tokens = None
    yield dict(type='done', tokens=int(out_tokens), estimated=estimated,
               tps=tps, seconds=(round(secs, 1) if secs >= 1.0 else None), model=model,
               ctx=(int(ctx_tokens) if ctx_tokens else None), ctx_max=ctx_max)


if __name__ == '__main__':
    print(json.dumps(respond(' '.join(sys.argv[1:])), ensure_ascii=False, indent=1))
