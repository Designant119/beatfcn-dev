"""把自然语言指令翻译成 harness 调用，并回一句人话。

设计上的三条硬规矩：

1. key 只从环境变量或 ~/.config/phira/chat.json 读，永不进仓库、永不进前端。
   网页只能调本模块，看不到凭据。
2. 模型能做的事被限制在 harness 的八个步骤里。它不能拼 shell、不能删文件、
   不能改路径白名单之外的目录——所有参数都再过一遍 harness.validate。
3. 拿不准就问，不要猜。模型输出不合法时如实告诉用户，不硬塞一个动作进去。

配置（环境变量优先，其次配置文件；文件形如 {"key": "...", "base": "...", "model": "..."}，
放在 ~/.config/phira/chat.json 并 chmod 600，别放进仓库）：
  PHIRA_CHAT_KEY    必填，模型服务凭据
  PHIRA_CHAT_BASE   默认 https://open.bigmodel.cn/api/paas/v4/chat/completions
  PHIRA_CHAT_MODEL  默认 glm-4-flash
  PHIRA_CHAT_CONFIG 可选，配置文件路径（默认 ~/.config/phira/chat.json）
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness import ORDER, STEPS, log, pipeline_next, running, start, status

TOOL_DOC = """你可以调用八个步骤，每个步骤对应流水线的一段：

""" + '\n'.join(
    f'- {name}：{STEPS[name]["module"]}；可用参数 '
    + '、'.join(f'{k}' + ('（必填）' if req else f'（默认 {d}）')
                for k, (d, req, _t) in STEPS[name]['args'].items())
    for name in ORDER
) + """

规则：
- 用户没说清楚的参数用默认值，不要瞎编路径。
- 同一步已经在跑就不要重复启动，告诉用户在跑。
- 不能用 shell，不能删文件，不能写到白名单之外。
- 用户只是想聊天或问问题时，action 用 "answer"，不要硬套一个动作。
- 用户问状态时，action 用 "status"。

只输出一个 JSON 对象，不要任何多余文字：
{"action": "run|status|answer", "step": "步骤名（action=run 时）",
 "args": {参数对象，可省略}, "reply": "给用户看的中文回复（一两句，不要长篇）"}
"""


def _config():
    """凭据与端点：环境变量优先，其次 ~/.config/phira/chat.json。

    文件放在用户目录、不进仓库，也不回传网页；环境变量永远盖过文件，
    方便临时换 key 或换模型不用改文件。
    """
    cfg = {}
    path = Path(os.environ.get('PHIRA_CHAT_CONFIG')
                or Path.home() / '.config' / 'phira' / 'chat.json')
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                cfg.update({k: v for k, v in data.items() if v})
        except Exception:                     # noqa: BLE001
            pass                              # 文件坏了就当没配，别把服务带崩
    for env_name, name in (('PHIRA_CHAT_KEY', 'key'),
                           ('PHIRA_CHAT_BASE', 'base'),
                           ('PHIRA_CHAT_MODEL', 'model')):
        value = os.environ.get(env_name)
        if value:
            cfg[name] = value
    return cfg


def _call(messages):
    cfg = _config()
    key = cfg.get('key')
    if not key:
        return None, ('后端没有配置 PHIRA_CHAT_KEY，AI 助手不可用。'
                      '把 key 写进后端环境变量，或放到 ~/.config/phira/chat.json 再试。')
    base = cfg.get('base') or 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
    model = cfg.get('model') or 'glm-4-flash'
    body = json.dumps({'model': model, 'messages': messages, 'temperature': 0.2}, ensure_ascii=False)
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


def _state_brief():
    st = status()
    done = [k for k in ORDER if st[k]['done']]
    run = [k for k in ORDER if st[k]['running']]
    return (f'已完成的步骤：{", ".join(done) or "无"}；'
            f'正在跑：{", ".join(run) or "无"}；'
            f'建议下一步：{pipeline_next() or "全部完成"}')


def respond(message, history=None):
    """返回 dict(ok, reply, action, step, result)。永不抛异常给上层。"""
    messages = [dict(role='system', content=TOOL_DOC + '\n\n当前状态：' + _state_brief())]
    for turn in (history or [])[-6:]:
        if turn.get('role') in ('user', 'assistant'):
            messages.append(dict(role=turn['role'], content=str(turn.get('content', ''))[:2000]))
    messages.append(dict(role='user', content=str(message)[:2000]))

    text, err = _call(messages)
    if err:
        return dict(ok=False, reply=err, action='error')

    raw = _extract_json(text)
    if raw is None:
        # 模型完全没给 JSON，那就把它说的话原样回给用户，不硬猜动作
        return dict(ok=True, action='answer', reply=text.strip()[:500])
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        return dict(ok=True, action='answer', reply=text.strip()[:500])

    action = decision.get('action') or 'answer'
    reply = str(decision.get('reply') or '').strip()

    if action == 'status':
        return dict(ok=True, action='status', reply=reply or _state_brief(), state=status())
    if action != 'run':
        return dict(ok=True, action='answer', reply=reply or '没太明白，能再说具体点吗？')

    step = decision.get('step')
    args = decision.get('args') or {}
    if step not in STEPS:
        return dict(ok=False, action='run', reply=f'模型想调一个不存在的步骤 {step!r}，没有执行。')
    if running(step):
        return dict(ok=False, action='run', step=step,
                    reply=f'{step} 已经在跑，不重复启动。用日志看进度。')
    result = start(step, args)
    if not result.get('ok'):
        return dict(ok=False, action='run', step=step, result=result,
                    reply=(reply + ' ' if reply else '') + '没启动成功：' + '；'.join(result.get('errors') or []))
    return dict(ok=True, action='run', step=step, result=result,
                reply=(reply + ' ' if reply else '') + f'已启动 {step}，日志可以随时看。')


if __name__ == '__main__':
    print(json.dumps(respond(' '.join(sys.argv[1:])), ensure_ascii=False, indent=1))
