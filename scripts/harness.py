"""八步流水线的统一驱动层：把每一步包成一条可启动、可查日志、可查状态的命令。

网页按钮、命令行、以及对话式操作（chat_agent）都调这里，避免三套逻辑各写一份。
每一步都在独立 tmux 会话里跑，跟在 console_server 里已有的下载/建库同一种方式，
断连不中断、可以随时读日志。

状态和日志不再写 /tmp（临时盘会被回收，一回收 status 就全报 false），
落在 repo 下的 .harness/（已 gitignore），可用 PHIRA_HARNESS_HOME 指到别处。

  python -m scripts.harness --status
  python -m scripts.harness --step audio --args '{"audio_dir": "/data/phira_audio"}'
  python -m scripts.harness --step features --args '{"device": "cpu"}' --dry-run
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_HOME = {}


def _state_dir():
    """状态与日志落盘位置。

    /tmp 是临时盘、重启会被回收；状态一丢 status 全报 false、日志也读不到。
    优先 repo 下的 .harness/（已 gitignore），repo 不可写就退到 ~/.phira-harness。
    想换地方直接设 PHIRA_HARNESS_HOME。
    """
    if 'dir' in _HOME:
        return _HOME['dir']
    candidates = []
    env = os.environ.get('PHIRA_HARNESS_HOME')
    if env:
        candidates.append(Path(env))
    candidates += [ROOT / '.harness', Path.home() / '.phira-harness']
    for cand in candidates:
        try:
            (cand / 'state').mkdir(parents=True, exist_ok=True)
            (cand / 'logs').mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        _HOME['dir'] = cand
        return cand
    _HOME['dir'] = Path('/tmp')          # 最后兜底，至少不耽误跑
    return _HOME['dir']

# 每一步：脚本、会话名、参数表（默认值 + 必填标记 + 类型）
STEPS = {
    'download': dict(
        module='scripts.phira_local_download', session='phira-dl',
        args={'dest': ('/data/phira_charts', True, str),
              'levels': (None, False, list), 'workers': (6, False, int),
              'no_hash_audio': (False, False, bool),
              'list_only': (False, False, bool)},
        outputs=('dest/_download_report.json',),
    ),
    'ingest': dict(
        module='scripts.ingest_phira_dataset', session='phira-ingest',
        args={'zips': ('/data/phira_charts', True, str),
              'out': ('/data/phira_db', True, str),
              'meta': (str(ROOT / 'phira_charts_meta.json'), False, str),
              'limit': (0, False, int)},
        outputs=('out/canon/phira', 'out/targets'),
    ),
    'audio': dict(
        module='scripts.extract_phira_audio', session='phira-audio',
        args={'zips': ('/data/phira_charts', True, str),
              'output': ('/data/phira_audio', True, str)},
        outputs=('output',),
    ),
    'tape': dict(
        module='scripts.build_feature_tape', session='phira-tape',
        args={'canon': ('/data/phira_db/canon/phira', True, str),
              'output': ('/data/tape_min', True, str)},
        outputs=('output',),
    ),
    'features': dict(
        module='scripts.batch_extract_features', session='phira-feat',
        args={'tape_dir': ('/data/tape_min', True, str),
              'audio_dir': ('/data/phira_audio', True, str),
              'features_dir': ('/data/features', True, str),
              'mert_source': ('/data/weights/mert', True, str),
              'device': ('cuda', False, str), 'decode_workers': (6, False, int),
              'log': ('/data/extract.jsonl', True, str),
              'until_idle_s': (900, False, float), 'shard': ('0/1', False, str)},
        outputs=('features_dir/features',),
    ),
    'full_tape': dict(
        module='scripts.build_full_native_tape', session='phira-fulltape',
        args={'targets_root': ('/data/phira_db/targets', True, str),
              'features_root': ('/data/features/features', True, str),
              'output': ('/data/tape', True, str),
              'workers': (8, False, int), 'limit': (0, False, int)},
        outputs=('output',),
    ),
    'exposure': dict(
        module='scripts.build_phira_exposure', session='phira-exp',
        args={'tape_dir': ('/data/tape', True, str),
              'canon': ('/data/phira_db/canon/phira', False, str),
              'seed': (20261108, True, int), 'steps': (2400, True, int),
              'output_pool': ('/data/evolve/pool.json', True, str),
              'output_exposure': ('/data/evolve/exposure.json', True, str),
              'output_report': ('', False, str)},
        outputs=('output_exposure',),
    ),
    'train': dict(
        module='scripts.train_phira_tiny', session='phira-train',
        args={'tape': ('/data/tape', True, str),
              'features': ('/data/features/features', True, str),
              'steps': (2, False, int), 'lr': (1e-3, False, float),
              'out': ('/data/ckpt/tiny.pt', True, str),
              'device': ('cpu', False, str)},
        outputs=('out',),
    ),
}

# 执行顺序：一键跑时按这个走；draft 不在流水线里，它需要一条音频
ORDER = ('download', 'ingest', 'audio', 'tape', 'features', 'full_tape', 'exposure', 'train')


def _resolve_output(step, args, spec):
    """outputs 里的相对写法（'out/canon/phira'）= 该步某个参数值 + 子路径。"""
    key, _, suffix = spec.partition('/')
    base = Path(str(args.get(key) or ''))
    return base / suffix if suffix else base


def validate(step, args):
    """按参数表校验并填默认值。返回 (filled, errors)。"""
    if step not in STEPS:
        return {}, [f'未知步骤 {step}，可用：{", ".join(STEPS)}']
    spec = STEPS[step]
    filled, errors = {}, []
    for name, (default, required, kind) in spec['args'].items():
        value = args.get(name, default)
        if value is None and required:
            errors.append(f'缺参数 {name}')
            continue
        if value is None:
            continue
        if kind is bool:
            filled[name] = bool(value)
        elif kind is int:
            try:
                filled[name] = int(value)
            except (TypeError, ValueError):
                errors.append(f'{name} 需要整数，收到 {value!r}')
        elif kind is float:
            try:
                filled[name] = float(value)
            except (TypeError, ValueError):
                errors.append(f'{name} 需要数字，收到 {value!r}')
        elif kind is list:
            filled[name] = list(value) if not isinstance(value, str) else [value]
        else:
            filled[name] = str(value)
    unknown = set(args) - set(spec['args'])
    if unknown:
        errors.append('不认识的参数：' + ', '.join(sorted(unknown)))
    # 演示环境：这台机器没真数据也没显卡，用一套假数据让训练真能跑起来
    demo = demo_train() if step == 'train' else None
    if demo:
        for name in ('tape', 'features', 'out'):
            if name not in (args or {}) and demo.get(name):
                filled[name] = str(demo[name])
        if 'steps' not in (args or {}) and demo.get('steps'):
            filled['steps'] = int(demo['steps'])
    return filled, errors


def build(step, args):
    """拼一条命令字符串。参数已经过 validate，这里只做拼接与转义。"""
    spec = STEPS[step]
    parts = [f"{sys.executable} -m {spec['module']}"]
    for name, value in args.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        elif isinstance(value, list):
            # --levels 这类参数收的是一个逗号串，不能按 shell 多参数摊开
            parts.append(flag + ' ' + _quote(','.join(str(v) for v in value)))
        else:
            parts.append(flag + ' ' + _quote(str(value)))
    return 'cd ' + _quote(str(ROOT)) + ' && ' + ' '.join(parts)


def _quote(text):
    return '"' + text.replace('"', '\\"') + '"' if ' ' in text else text


def start(step, args, *, dry_run=False):
    """校验 + 起 tmux。返回 dict，网页和命令行都直接用这个结构。"""
    explicit = set((args or {}).keys())
    filled, errors = validate(step, args)
    if errors:
        return dict(ok=False, step=step, errors=errors)
    filled = apply_aliases(step, filled, explicit)
    cmd = build(step, filled)
    if dry_run:
        return dict(ok=True, step=step, command=cmd, dry_run=True)
    session = STEPS[step]['session']
    try:
        has = subprocess.run(['tmux', 'has-session', '-t', session],
                             capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return dict(ok=False, step=step, command=cmd,
                    errors=['机器上没有 tmux：sudo apt install -y tmux'])
    if has.returncode == 0:
        return dict(ok=False, step=step, command=cmd,
                    errors=[f'已有 {session} 在跑，先 tmux attach -t {session} 看看'])
    remember_paths(step, filled)
    log_path = _state_dir() / 'logs' / f'{session}.log'
    # tmux 会丢当前 shell 的部分环境：torch 装在 ~/.local 时，
    # 不把 PYTHONUSERBASE 带进去，一进会话就 ModuleNotFoundError
    pre = ''
    for key in ('PYTHONUSERBASE', 'PYTHONPATH', 'VIRTUAL_ENV'):
        value = os.environ.get(key)
        if value:
            pre += f'export {key}={shlex.quote(value)}; '
    wrapped = pre + f'{cmd} 2>&1 | tee {log_path}'
    try:
        started = subprocess.run(['tmux', 'new-session', '-d', '-s', session, wrapped],
                                 capture_output=True, text=True, timeout=10)
    except Exception as exc:                      # noqa: BLE001
        return dict(ok=False, step=step, command=cmd, errors=[f'{type(exc).__name__}: {exc}'])
    if started.returncode != 0:
        return dict(ok=False, step=step, command=cmd,
                    errors=[('tmux 启动失败：' + (started.stderr or '无输出'))[:200]])
    time.sleep(0.6)          # 参数不对的脚本会秒退，别急着报“已启动”
    alive = running(step)
    # 只看最后几行会漏掉 Traceback 的开头（栈有十几行），所以多读一点、多认几种错
    tail = [l.rstrip() for l in log(step, 40).splitlines() if l.strip()]
    blow = '\n'.join(tail)
    crashed = any(k in blow for k in ('Traceback', 'error:', 'Error:', 'unrecognized arguments',
                                      'Permission denied', 'No such file',
                                      'ModuleNotFoundError', 'ImportError'))
    if not alive and crashed:
        return dict(ok=False, step=step, command=cmd, session=session, alive=False,
                    errors=['启动后立刻就退了：' + (tail[-1][:200] if tail else '无输出')])
    (_state_dir() / 'state' / f'{step}.json').write_text(
        json.dumps(filled, ensure_ascii=False), encoding='utf-8')
    return dict(ok=True, step=step, command=cmd, session=session, alive=alive,
                log=f'tmux attach -t {session}')


def log(step, lines=40):
    session = STEPS[step]['session'] if step in STEPS else step
    path = _state_dir() / 'logs' / f'{session}.log'
    if not path.is_file():
        return ''
    return '\n'.join(path.read_text(errors='replace').splitlines()[-lines:])


def running(step):
    session = STEPS[step]['session'] if step in STEPS else step
    try:
        out = subprocess.run(['tmux', 'has-session', '-t', session],
                             capture_output=True, text=True, timeout=10)
        return out.returncode == 0
    except FileNotFoundError:
        return False


def _artifact_ok(path):
    """文件就看在不在；目录还得非空，否则半截中断也会被当成已完成。"""
    try:
        if path.is_file():
            return True, None
        if path.is_dir():
            return any(True for _ in path.iterdir()), '目录是空的'
        return False, '还没生成'
    except Exception as exc:                  # noqa: BLE001
        return False, str(exc)


def _download_failed(step, args):
    """下载报告里失败几个，页面可以顺手提一句。"""
    if step != 'download':
        return 0
    dest = args.get('dest')
    if not dest:
        return 0
    f = Path(str(dest)) / '_download_report.json'
    if not f.is_file():
        return 0
    try:
        return int(json.loads(f.read_text(encoding='utf-8')).get('failed') or 0)
    except Exception:                         # noqa: BLE001
        return 0


def status():
    """各步产物是否已存在，供网页显示“哪步已经跑完”。"""
    report = {}
    for step in ORDER:
        spec = STEPS[step]
        args, _ = validate(step, {})
        for name, (default, required, _kind) in spec['args'].items():
            if default is not None:
                args[name] = default
        args = apply_aliases(step, args, ())
        # 上次真跑过就按那次的实际参数查产物，否则默认路径会误报
        saved = _state_dir() / 'state' / f'{step}.json'
        if saved.is_file():
            try:
                args.update(json.loads(saved.read_text(encoding='utf-8')))
            except Exception:                     # noqa: BLE001
                pass
        # 只拉了清单不算“下载完成”，否则下一步会以为谱面已经在盘上
        if step == 'download' and args.get('list_only'):
            present, why = False, '只拉了清单'
        elif step == 'download' and not _resolve_output(step, args, 'dest/_download_report.json').exists():
            # 别人从浏览器/别的工具下的，没有报告；只要目录里有 zip 就认
            dest = Path(str(args.get('dest') or ''))
            try:
                has_zip = dest.is_dir() and any(dest.glob('*.zip'))
            except Exception:                 # noqa: BLE001
                has_zip = False
            present = bool(has_zip)
            why = '目录里已有 zip（没跑过下载器）' if present else '还没生成'
        else:
            checks = [_artifact_ok(_resolve_output(step, args, o)) for o in spec['outputs']]
            present = all(ok for ok, _ in checks)
            why = next((msg for ok, msg in checks if not ok), None)
        report[step] = dict(done=present, running=running(step),
                            session=spec['session'], why=why,
                            failed=_download_failed(step, args) if present else 0)
    return report


def demo_train():
    """有没有“能直接起训练”的演示环境；有就返回它的路径和步数。"""
    f = _state_dir() / 'demo_train.json'
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
    except Exception:                         # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    if not Path(str(data.get('tape', ''))).is_dir():
        return None
    if not Path(str(data.get('features', ''))).is_dir():
        return None
    return data


def train_progress():
    """训练跑到第几步、loss 多少。读日志最后一条 json，不猜。"""
    log_path = _state_dir() / 'logs' / 'phira-train.log'
    if not log_path.is_file():
        return None
    try:
        lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()[-200:]
        age = time.time() - log_path.stat().st_mtime
    except OSError:
        return None
    last = None
    for line in lines:
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            row = json.loads(line)
        except Exception:                     # noqa: BLE001
            continue
        if isinstance(row, dict) and row.get('step') is not None:
            last = row
    total = None
    try:
        total = validate('train', {})[0].get('steps')
    except Exception:                         # noqa: BLE001
        pass
    if last is None:
        return dict(step=None, total=total, loss=None, running=running('train'),
                    age=round(age, 1))
    return dict(step=int(last['step']), total=total,
                loss=(round(float(last['loss']), 4) if last.get('loss') is not None else None),
                running=running('train'), age=round(age, 1))


def inspect_charts(dest=None, sample=None):
    """真去目录里点数、验包：多少个、多大、有没有打不开的。

    不是看状态文件，也不是看清单——是拿 zipfile 一个个翻开验。
    """
    import zipfile
    paths = load_paths()
    dest = Path(str(dest or paths.get('charts') or '/data/phira_charts'))
    if not dest.is_dir():
        return dict(ok=False, dest=str(dest), count=0, reason='目录不存在')
    zips = sorted(p for p in dest.glob('*.zip') if p.is_file())
    total = 0
    for p in zips:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    limit = sample if sample else min(len(zips), 100)
    bad, checked = [], 0
    for p in zips[:limit]:
        checked += 1
        try:
            with zipfile.ZipFile(p) as z:
                if not z.namelist():
                    bad.append(p.name)
                    continue
                if z.testzip() is not None:
                    bad.append(p.name)
        except Exception:                     # noqa: BLE001
            bad.append(p.name)
    return dict(ok=True, dest=str(dest), count=len(zips), total_bytes=total,
                checked=checked, bad=bad[:10], bad_count=len(bad),
                report=(dest / '_download_report.json').is_file())


def output_paths(step):
    """这一步的产物现在会写到哪、在不在。给页面显示“文件在哪”。"""
    spec = STEPS.get(step)
    if not spec:
        return []
    args, _ = validate(step, {})
    for name, (default, _required, _kind) in spec['args'].items():
        if default is not None:
            args[name] = default
    args = apply_aliases(step, args, ())
    saved = _state_dir() / 'state' / f'{step}.json'
    if saved.is_file():
        try:
            args.update(json.loads(saved.read_text(encoding='utf-8')))
        except Exception:                     # noqa: BLE001
            pass
    out = []
    if step == 'download':                    # 用户问“文件在哪”要看谱面目录，不是那份报告
        dest = Path(str(args.get('dest') or ''))
        item = dict(path=str(dest), exists=(dest / '_download_report.json').exists())
        if args.get('list_only'):
            item['exists'] = False
            item['note'] = '只拉了清单，谱面还没下'
        return [item]
    list_only = step == 'download' and bool(args.get('list_only'))
    for o in spec['outputs']:
        path = _resolve_output(step, args, o)
        item = dict(path=str(path), exists=path.exists())
        if list_only:
            item['exists'] = False
            item['note'] = '只拉了清单，谱面还没下'
        out.append(item)
    return out

# ---- 路径记忆：某一处改过目录，后面几步自动跟着走 ----
# arg 名 -> (别名, 子路径)；别名之间像接力，谁先说就听谁的
PATH_ALIAS = {
    'download': {'dest': ('charts', '')},
    'ingest':   {'zips': ('charts', ''), 'out': ('db', '')},
    'audio':    {'zips': ('charts', ''), 'output': ('audio', '')},
    'tape':     {'canon': ('db', '/canon/phira'), 'output': ('tape_min', '')},
    'features': {'tape_dir': ('tape_min', ''), 'audio_dir': ('audio', ''),
                 'features_dir': ('features', '')},
    'full_tape': {'targets_root': ('db', '/targets'),
                  'features_root': ('features', '/features'), 'output': ('tape', '')},
    'exposure': {'tape_dir': ('tape', ''), 'canon': ('db', '/canon/phira'),
                 'output_pool': ('evolve', '/pool.json'),
                 'output_exposure': ('evolve', '/exposure.json')},
    'train':    {'tape': ('tape', ''), 'features': ('features', '/features'),
                 'out': ('ckpt', '')},
}


def _paths_file():
    return _state_dir() / 'paths.json'


def load_paths():
    """已经定下来的几个位置（charts/db/audio/…）。"""
    f = _paths_file()
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:                         # noqa: BLE001
        return {}


def remember_paths(step, args):
    """把这步真正用的目录记下来，后面几步照着走。"""
    mapping = PATH_ALIAS.get(step) or {}
    known = load_paths()
    changed = False
    for arg, (alias, suffix) in mapping.items():
        value = args.get(arg)
        if not isinstance(value, str) or not value:
            continue
        base = value[:-len(suffix)] if suffix and value.endswith(suffix) else value
        if known.get(alias) != base:
            known[alias] = base
            changed = True
    if changed:
        try:
            _paths_file().parent.mkdir(parents=True, exist_ok=True)
            _paths_file().write_text(json.dumps(known, ensure_ascii=False, indent=1), encoding='utf-8')
        except Exception:                    # noqa: BLE001
            pass
        # 谱面目录换了，就把下载那步的记忆也对齐，不然页面还按老目录判“只拉了清单”
        want = known.get('charts')
        if want:
            try:
                st_file = _state_dir() / 'state' / 'download.json'
                if st_file.is_file():
                    data = json.loads(st_file.read_text(encoding='utf-8'))
                    if data.get('dest') != want:
                        data['dest'] = want
                        data.pop('list_only', None)
                        st_file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
            except Exception:                # noqa: BLE001
                pass


def apply_aliases(step, args, explicit=()):
    """没被用户明确指定的路径参数，用记忆里的位置替换默认值。"""
    mapping = PATH_ALIAS.get(step) or {}
    known = load_paths()
    for arg, (alias, suffix) in mapping.items():
        if arg in explicit:
            continue
        base = known.get(alias)
        if base:
            args[arg] = base + suffix
    return args


def pipeline_next():
    """下一步该做什么：第一个还没产物的步骤。"""
    st = status()
    for step in ORDER:
        if not st[step]['done'] and not st[step]['running']:
            return step
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', choices=sorted(STEPS))
    ap.add_argument('--args', default='{}', help='JSON 参数对象')
    ap.add_argument('--dry-run', action='store_true', help='只回命令，不启动')
    ap.add_argument('--status', action='store_true', help='报告各步产物与运行状态')
    ap.add_argument('--log', help='读某一步的日志')
    ap.add_argument('--lines', type=int, default=40)
    a = ap.parse_args()

    if a.status:
        return print(json.dumps(status(), ensure_ascii=False, indent=1))
    if a.log:
        return print(log(a.log, a.lines))
    if not a.step:
        return print('用 --step 或 --status 或 --log')
    try:
        args = json.loads(a.args)
    except json.JSONDecodeError as exc:
        return print(f'--args 不是合法 JSON：{exc}')
    print(json.dumps(start(a.step, args, dry_run=a.dry_run), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
