#!/usr/bin/env python3
"""训练机操作台后端 —— 零依赖，一条命令跑起来。

用法（WSL / Linux 里）:
    python3 console_server.py                      # 默认 0.0.0.0:7777
    python3 console_server.py 8080                 # 自定义端口
    python3 console_server.py 7777 --token 口令     # 设访问码（公网暴露时必开）

想从外网看（同学家机器在 NAT 后面，不用碰路由器）:
    1) Windows 侧下载 cloudflared（单文件 exe，cloudflare 官网）
    2) cmd 里:  cloudflared.exe tunnel --url http://localhost:7777
    3) 它打印一个 https://随机词.trycloudflare.com —— 发给自己，浏览器打开即可
    隧道只在 cloudflared 黑窗口开着时有效；配合 --token 使用更稳。

然后浏览器打开 http://localhost:7777 （手机连同一 WiFi 用 http://本机IP:7777）。
页面会自动进入 LIVE 模式：01/02 自检全自动，不需要复制粘贴任何东西。

接口（给 同学训练操作页.html 用，也可手动 curl）:
    GET  /api/ping      探活
    GET  /api/env       环境自检（硬件 hw / 软件 sw / 各盘 disk；WSL 里还会经 powershell.exe 读 Windows 宿主侧的卡/内存/盘）
    GET  /api/status    训练阶段状态（读同目录 stages.json，暂无则空表）
    GET  /api/trainlog  训练输出尾部 + 解析出的最新全部参数（轮数/步数/lr/loss…日志里有什么就给什么；
                        验证侧指标训练行里不打，另从 reports 的验证记录以 val. 前缀并入，见 _val_metrics）
                        来源优先级：BEATFCN_TRAIN_LOG 环境变量 → stages.json 各阶段的 "log" 路径 →
                        train.log / reports/train_console.log 约定位置 → tmux 会话 train 的屏幕缓冲
    GET  /api/train-plan 05 页计划下拉框：REPO/plans/evolve/*.json 的名字 + learning_rate/存点间隔（只读展示）
    POST /api/action    白名单动作：{"id":"env-recheck"} 清缓存重查环境；{"id":"train-start"} 按 05 页控件值
                        由后端拼装命令、tmux 会话 train 里启动（dry_run=true 只回命令不启动）。
                        无删除/停止类动作

设计约束:
    * 只用标准库，装完系统就能跑，先于 torch 存在
    * 只暴露只读探测 + 白名单动作，绝不提供删除类操作
    * 子进程全部带超时；torch 检测走子进程，本进程永不 import torch
"""
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse, parse_qs, quote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------- 配置（按需改这三行） ----------------
REPO = os.environ.get('BEATFCN_REPO') or os.path.expanduser('~/beatfcn-dev')     # 代码目录（换过路径就设 BEATFCN_REPO）
EXPECTED_HEAD = ''                             # 期望的 git 短提交号；空串=不校验（只显示当前版本）。
                                               # 别写死具体 SHA：每次提交都会变，页面会永远报"不是这一版"。
HTML_NAMES = ('同学训练操作页.html', 'index.html')

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = ''          # 非空时所有 /api/* 需带 ?token= 或 X-Token 头
_cache = {'env': (0.0, None), 'torch': (0.0, None), 'win': (0.0, None)}   # (时间戳, 结果)
ENV_TTL = 15.0       # 秒：/api/env 整体缓存（须大于页面 12s 轮询间隔，否则每轮都冷探测）
TORCH_TTL = 60.0     # 秒：torch 子进程检测缓存（首次 import 慢）
WIN_TTL = 20.0       # 秒：Windows 宿主侧探测缓存（powershell 启动约 1 秒，别每轮都起）
TRAIN_TAIL_LINES = 300        # /api/trainlog 回给页面的尾部行数
TRAIN_READ_BYTES = 96 * 1024  # 日志文件只读末尾这么多字节，别把整份日志搬进内存
_PS = shutil.which('powershell.exe') or shutil.which('pwsh') or shutil.which('powershell')
_torch_probing = False  # 后台探测进行中（单飞）


def _run(cmd, timeout=10):
    """跑一条命令，返回 stdout 首行（去空白）。失败返回 None。"""
    try:
        out = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                             text=True, timeout=timeout)
        line = (out.stdout or '').strip().splitlines()
        return line[0].strip() if line else None
    except Exception:
        return None


def _ok(cmd, timeout=30):
    """按退出码判定（check 命令可以没有输出，如 test -f）。"""
    try:
        out = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=timeout)
        return out.returncode == 0
    except Exception:
        return False


def _item(key, label, value, status, note='', group='sw'):
    return {'key': key, 'label': label, 'value': value or '', 'status': status, 'note': note, 'group': group}


def _gb(bytes_free):
    return round(bytes_free / (1000 ** 3), 1)


def _torch_probe():
    """子进程里问 torch 版本与 CUDA 可用性，结果缓存。"""
    now = time.time()
    ts, hit = _cache['torch']
    if hit is not None and now - ts < TORCH_TTL:
        return hit
    probe = (
        "import sys\n"
        "try:\n"
        "    import torch\n"
        "    print(torch.__version__)\n"
        "    print('yes' if torch.cuda.is_available() else 'no')\n"
        "except Exception as e:\n"
        "    print('ERR:' + type(e).__name__)\n"
        "    sys.exit(0)\n"
    )
    try:
        out = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                             text=True, timeout=90)
        lines = (out.stdout or '').strip().splitlines()
    except Exception:
        lines = []
    if not lines or lines[0].startswith('ERR:'):
        hit = ('__missing__', None)
    else:
        hit = (lines[0], (lines[1].lower() == 'yes') if len(lines) > 1 else None)
    _cache['torch'] = (now, hit)
    return hit


def _torch_nowait():
    """缓存热直接用；冷则后台单飞探测，立刻回 __pending__（页面下一轮轮询拿到真值）。"""
    global _torch_probing
    now = time.time()
    ts, hit = _cache['torch']
    if hit is not None and now - ts < TORCH_TTL:
        return hit
    if not _torch_probing:
        _torch_probing = True

        def _done():
            global _torch_probing
            _torch_probe()
            _torch_probing = False

        threading.Thread(target=_done, daemon=True).start()
    return ('__pending__', None)


def _ps(script, timeout=25):
    """在 WSL 里调 Windows 的 powershell.exe 执行一行脚本，返回 stdout。不在 WSL 里则 None。"""
    if not _PS:
        return None
    try:
        out = subprocess.run([_PS, '-NoProfile', '-NonInteractive', '-Command', script],
                             capture_output=True, text=True, timeout=timeout)
        return (out.stdout or '').strip() or None
    except Exception:
        return None


WIN_SCRIPT = (
    "$g=(Get-CimInstance Win32_VideoController).Name -join ' / ';"
    "$r=[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB);"
    "$f=(Get-PSDrive -PSProvider FileSystem | ?{$_.Free -gt 0} | "
    "%{\"$($_.Name):$([math]::Round($_.Free/1GB))G\"}) -join ',';"
    "\"GPU=$g\";\"RAM=$r\";\"FREE=$f\""
)


def win_items():
    """Windows 宿主侧实测（显卡/物理内存/各盘余量）。页面的 01 层不再需要人手动粘输出。"""
    now = time.time()
    ts, hit = _cache['win']
    if hit is not None and now - ts < WIN_TTL:
        return hit
    if not _PS:
        hit = [_item('WINSIDE', 'Windows 侧', '', 'none',
                     '没找到 powershell.exe：后端不在 WSL 里跑，或 PATH 没带 Windows 目录', 'hw')]
        _cache['win'] = (now, hit)
        return hit
    raw = _ps(WIN_SCRIPT)
    if not raw:
        hit = [_item('WINSIDE', 'Windows 侧', '', 'none',
                     'powershell.exe 起不来或超时（Windows 侧没开 WSL 互操作）', 'hw')]
        _cache['win'] = (now, hit)
        return hit
    kv = {}
    for line in raw.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            kv[k.strip().upper()] = v.strip()

    out = []
    gpu = kv.get('GPU', '')
    if gpu:
        ok = bool(re.search(r'4090|3090|4080', gpu, re.I))
        out.append(_item('WINGPU', '显卡(Windows 实测)', gpu,
                         'pass' if ok else 'warn', '' if ok else '不是预定的训练卡', 'hw'))
    ram = kv.get('RAM', '')
    if ram.isdigit():
        g = int(ram)
        st = 'pass' if g >= 30 else ('warn' if g >= 16 else 'bad')
        out.append(_item('WINRAM', '内存(Windows 实测)', f'{g}G', st,
                         '' if st == 'pass' else ('要调小 BEATFCN_MAX_LOADER_WORKERS' if st == 'warn' else '太小'), 'hw'))
    free = kv.get('FREE', '')
    if free:
        parts = [p for p in free.split(',') if p]
        maxg = 0
        for p in parts:
            m = re.search(r'(\d+)G', p)
            if m:
                maxg = max(maxg, int(m.group(1)))
        st = 'pass' if maxg >= 400 else ('warn' if maxg >= 250 else 'bad')
        out.append(_item('WINFREE', '空盘(Windows 实测)', free, st,
                         '把谱面和特征放到最大那块盘上' if st != 'pass' else '', 'hw'))
    if not out:
        out = [_item('WINSIDE', 'Windows 侧', '', 'none', '没解析到 GPU/RAM/FREE 行', 'hw')]
    _cache['win'] = (now, out)
    return out


def data_items():
    """层04：训练真正要的东西在不在——按方案 json 里引用的路径数文件，不看人嘴上说的“导好了”。

    一个方案跑起来只需要两样东西：起始权重（init, .pt）和数据集（exposure, .json）。
    这两样在不在，直接决定这个方案能不能开跑，所以逐个文件检查，最后给一行“几个方案齐了”。
    """
    if not os.path.isdir(REPO):
        return [_item('DATA', '数据目录', '', 'none',
                      f'没找到 {REPO}：代码还没拉下来，数据无从检查', 'data')]
    root = (DATA_ROOT or '/data').rstrip('/') or '/data'
    plans_dir = os.path.join(REPO, 'plans', 'evolve')
    inits, exposures, total, ready = {}, {}, 0, 0
    try:
        names = sorted(os.listdir(plans_dir))
    except OSError:
        names = []
    for name in names:
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(plans_dir, name), encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            continue           # 坏的方案不参与统计，05 页会给它留空行
        if not isinstance(cfg, dict):
            continue
        total += 1
        init = cfg.get('init')
        exp = (cfg.get('exposure') or {}).get('path') if isinstance(cfg.get('exposure'), dict) else None
        ok_init = isinstance(init, str) and os.path.isfile(_repoint(init, root))
        ok_exp = isinstance(exp, str) and os.path.isfile(_repoint(exp, root))
        if ok_init and ok_exp:
            ready += 1
        for path, bucket, good in ((init, inits, ok_init), (exp, exposures, ok_exp)):
            if isinstance(path, str) and path:
                base = os.path.basename(_repoint(path, root))
                bucket[base] = bucket.get(base, True) and good
    items = []
    for base, good in sorted(exposures.items()):
        items.append(_item('EXP ' + base, '数据集', base, 'pass' if good else 'bad',
                           '' if good else '不在：%s 里没有这个文件' % root, 'data'))
    for base, good in sorted(inits.items()):
        items.append(_item('INIT ' + base, '起始权重', base, 'pass' if good else 'bad',
                           '' if good else '不在：%s 里没有这个文件' % root, 'data'))
    if total:
        items.append(_item('PLANS', '可开跑的方案', '%d / %d' % (ready, total),
                           'pass' if ready == total else ('warn' if ready else 'bad'),
                           '' if ready == total else '缺东西的方案开跑会直接报错', 'data'))
    else:
        items.append(_item('PLANS', '方案清单', '', 'none',
                           'plans/evolve 里没有方案，05 页也就没东西可选', 'data'))
    # 老盘布局（inputs/ + reports/ 那套）如果还在，顺手数一下，不影响上面的判定
    feat = tape = seen = 0
    for d in (os.path.join(REPO, 'inputs'), os.path.join(REPO, 'reports')):
        if not os.path.isdir(d):
            continue
        for dirpath, dirnames, filenames in os.walk(d):
            seen += 1
            if seen > 3000:
                break
            names_set = set(filenames)
            if 'L12.npy' in names_set and 'L23.npy' in names_set:
                feat += 1
            tape += sum(1 for f in filenames if f.endswith('.jsonl'))
        if seen > 3000:
            break
    if seen:
        items.append(_item('LEGACY', '旧版 inputs/reports', '%d 首 / %d 份清单' % (feat, tape), 'none',
                           '这套目录跟现在的 evolve 方案无关，只是盘上残留', 'data'))
    return items


def env_items():
    """全部自检项。group: hw=层01 / sw=层02 / disk=层04。"""
    items = []

    # ---- 硬件（层01） ----
    gpu = _run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    if not gpu:
        items.append(_item('GPU', '显卡', '', 'bad', 'nvidia-smi 没输出：WSL 装完没重启，或驱动没装', 'hw'))
        items.append(_item('VRAM', '显存', '', 'none', '见上', 'hw'))
    else:
        ok = bool(re.search(r'4090|3090|4080', gpu, re.I))
        items.append(_item('GPU', '显卡', gpu,
                           'pass' if ok else 'warn', '' if ok else '不是预定的训练卡，可能跑不动', 'hw'))
    vram_txt = _run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader']) or ''
    m = re.search(r'(\d+)', vram_txt)
    if m:
        vram_mib = int(m.group(1))
        vram_gb = round(vram_mib / 1024)
        st = 'pass' if vram_gb >= 22 else ('warn' if vram_gb >= 15 else 'bad')
        items.append(_item('VRAM', '显存', f'{vram_gb}G', st,
                           '' if st == 'pass' else ('要降批大小' if st == 'warn' else '不够'), 'hw'))
    elif gpu:
        items.append(_item('VRAM', '显存', '', 'none', '没读到', 'hw'))

    try:
        with open('/proc/meminfo') as f:
            memtotal_kb = int(re.search(r'MemTotal:\s+(\d+)', f.read()).group(1))
        ram_gb = round(memtotal_kb / (1024 ** 2))
        st = 'pass' if ram_gb >= 30 else ('warn' if ram_gb >= 16 else 'bad')
        items.append(_item('RAM', '内存(WSL可见)', f'{ram_gb}G', st,
                           '' if st == 'pass' else '要调小 BEATFCN_MAX_LOADER_WORKERS' if st == 'warn' else '太小', 'hw'))
    except Exception:
        items.append(_item('RAM', '内存', '', 'none', '没读到', 'hw'))

    # ---- Windows 宿主侧实测（层01） ----
    items.extend(win_items())

    # ---- 数据在不在（层04）：数文件，不等人报"导好了" ----
    items.extend(data_items())

    # ---- 盘余量（层04 + 层01）：代码和数据在同一块盘，只盯这块 ----
    repo_free = repo_total = 0
    if os.path.isdir(REPO):
        try:
            u = shutil.disk_usage(REPO)
            repo_free, repo_total = _gb(u.free), _gb(u.total)
        except Exception:
            pass
    if repo_total:
        # 盘是给定的 1T，不算够不够；数字只用来确认"看的是那块盘"
        st = 'warn' if repo_free < 100 else 'none'
        items.append(_item('REPODISK', '代码+数据盘', f'{repo_free}G 空闲 / 共 {repo_total}G', st,
                           '余量不到 100G：存快照和日志可能中途撑满，先清点旧产物' if st == 'warn' else '', 'disk'))
    else:
        st = 'none'
        items.append(_item('REPODISK', '代码+数据盘', '', 'none',
                           '代码还没拉下来，数据放哪块盘还没定，这行先空着', 'disk'))
    mounts = [('/', 'Linux 根盘')]
    try:
        for name in sorted(os.listdir('/mnt')):
            p = f'/mnt/{name}'
            if os.path.isdir(p):
                mounts.append((p, f'Windows {name.upper()}: 盘'))
    except Exception:
        pass
    for path, label in mounts:
        try:
            u = shutil.disk_usage(path)
        except Exception:
            continue
        free = _gb(u.free)
        items.append(_item(f'DISK{path}', label, f'{free}G 空闲', 'none',
                           '列出来给你看的，判定只看「代码+数据盘」那行', 'disk'))
    if repo_total:
        items.append(_item('FREE', '盘余量', f'{repo_free}G', 'pass',
                           '盘是给定的，够不够不用算', 'hw'))
    else:
        items.append(_item('FREE', '盘余量', '', 'none',
                           '等代码拉下来，这行才知道看哪块盘', 'hw'))

    # ---- 软件（层02） ----
    pv = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
    items.append(_item('PY', 'python', pv,
                       'pass' if sys.version_info >= (3, 10) else 'warn',
                       '' if sys.version_info >= (3, 10) else '版本偏老', 'sw'))

    tver, tcuda = _torch_nowait()
    if tver == '__pending__':
        items.append(_item('TORCH', 'torch', '检测中', 'none', '首次 import 慢，十几秒后自动出结果', 'sw'))
        items.append(_item('CUDATORCH', 'CUDA 可用', '', 'none', '等 torch 检测完成', 'sw'))
    elif tver == '__missing__':
        items.append(_item('TORCH', 'torch', '', 'bad', '没装：pip install -r requirements-fc.txt', 'sw'))
        items.append(_item('CUDATORCH', 'CUDA 可用', '', 'none', '先装 torch', 'sw'))
    else:
        items.append(_item('TORCH', 'torch', tver, 'pass', '', 'sw'))
        if tcuda is True:
            items.append(_item('CUDATORCH', 'CUDA 可用', 'True', 'pass', '', 'sw'))
        elif tcuda is False:
            items.append(_item('CUDATORCH', 'CUDA 可用', 'False', 'bad', 'torch 看不到显卡', 'sw'))
        else:
            items.append(_item('CUDATORCH', 'CUDA 可用', '', 'none', '没读到', 'sw'))

    ff = _run(['ffmpeg', '-version'])
    items.append(_item('FFMPEG', 'ffmpeg', ff or '', 'pass' if ff else 'bad',
                       '' if ff else '没装：sudo apt install -y ffmpeg', 'sw'))

    head = _run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'])
    if not head:
        items.append(_item('HEAD', '代码版本', '', 'bad', f'没找到 {REPO}', 'sw'))
    elif not EXPECTED_HEAD:
        items.append(_item('HEAD', '代码版本', head, 'pass', '', 'sw'))
    elif head.startswith(EXPECTED_HEAD):
        items.append(_item('HEAD', '代码版本', head, 'pass', '对上了', 'sw'))
    else:
        items.append(_item('HEAD', '代码版本', head, 'warn', f'不是 {EXPECTED_HEAD} 这一版', 'sw'))

    return items


def get_env():
    now = time.time()
    ts, hit = _cache['env']
    if hit is not None and now - ts < ENV_TTL:
        return hit
    hit = env_items()
    _cache['env'] = (now, hit)
    return hit


def get_status():
    """训练阶段状态。stages.json（可选，同目录）声明阶段:
    [{"id":"train","label":"训练","check":"命令","start":"命令","stop":"命令"}]
    目前只回显声明与 check 的退出码，start/stop 按下页面按钮才会经 /api/action 调用。"""
    path = os.path.join(HERE, 'stages.json')
    stages = []
    if os.path.isfile(path):
        try:
            declared = json.load(open(path, encoding='utf-8'))
            for s in declared:
                st = ('pass' if _ok(s['check']) else 'bad') if s.get('check') else 'none'
                stages.append({'id': s.get('id'), 'label': s.get('label'), 'status': st})
        except Exception as e:
            stages.append({'id': 'error', 'label': f'stages.json 解析失败: {e}', 'status': 'bad'})
    return {'stages': stages}


# ---------------- 05 训练参数 / 启动（train-plan + train-start） ----------------

_PLAN_NAME_RE = re.compile(r'[A-Za-z0-9._-]+\.json')


def train_plans():
    """05 页「训练方案」下拉框 + 详情：REPO/plans/evolve/*.json 的只读清单。
    字段缺了就给 None，绝不造数；目录不在就空列表。名字先过 _PLAN_NAME_RE——白名单只收
    无壳安全字符的文件名，train-start 拼 shell 命令时才敢直接往里嵌。"""
    plans = []
    root = (DATA_ROOT or '/data').rstrip('/') or '/data'
    d = os.path.join(REPO, 'plans', 'evolve')
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return plans
    for name in names:
        if not _PLAN_NAME_RE.fullmatch(name) or not os.path.isfile(os.path.join(d, name)):
            continue
        rec = {'name': name, 'learning_rate': None, 'light_checkpoint_every': None,
               'light_checkpoint_keep': None, 'out_dir': None, 'horizon_updates': None,
               'init': None, 'seed': None, 'loader_workers': None}
        try:
            with open(os.path.join(d, name), encoding='utf-8') as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                lr = cfg.get('learning_rate')
                if isinstance(lr, (int, float)) and not isinstance(lr, bool):
                    rec['learning_rate'] = lr
                for key in ('light_checkpoint_every', 'light_checkpoint_keep', 'seed', 'loader_workers'):
                    v = cfg.get(key)
                    if isinstance(v, int) and not isinstance(v, bool):
                        rec[key] = v
                for key in ('out_dir', 'init'):
                    v = cfg.get(key)
                    if isinstance(v, str) and v:
                        rec[key] = _repoint(v, root)
                sched = cfg.get('schedule')
                if isinstance(sched, dict):
                    h = sched.get('horizon_updates')
                    if isinstance(h, int) and not isinstance(h, bool):
                        rec['horizon_updates'] = h
        except Exception:
            pass          # 单个 json 坏了不影响整张清单：名字照列，字段留空
        plans.append(rec)
    return plans


# 数据根。方案 json 里写死的是 /data/...，换台机器就找不到东西；
# 用 BEATFCN_DATA_ROOT 指到实际位置（比如硬盘挂到 /mnt/e 或同学家的 ~/data），
# 下面的 _repoint 只改副本，仓库里的原方案一个字节不动。
DATA_ROOT = os.environ.get('BEATFCN_DATA_ROOT', '/data') or '/data'


def _repoint(obj, root):
    """把方案里所有以 /data 开头的字符串换成 root 开头。递归 dict/list，其它原样返回。"""
    if isinstance(obj, str):
        if obj == '/data' or obj.startswith('/data/'):
            return root.rstrip('/') + obj[len('/data'):]
        return obj
    if isinstance(obj, list):
        return [_repoint(x, root) for x in obj]
    if isinstance(obj, dict):
        return {k: _repoint(v, root) for k, v in obj.items()}
    return obj


def _plan_copy_with_checkpoint(plan_name, every=None, microbatch=None):
    """把方案复制到临时目录，改 store 间隔、必要时把 /data 换成 DATA_ROOT，返回副本绝对路径。
    原方案文件一个字节都不动；副本放系统临时目录，不进仓库。没改动就返回原路径。"""
    src = os.path.join(REPO, 'plans', 'evolve', plan_name)
    with open(src, encoding='utf-8') as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError('方案不是 json 对象')
    changed = False
    if isinstance(every, int):
        cfg['light_checkpoint_every'] = every
        changed = True
    if isinstance(microbatch, int):
        cfg['physical_microbatch_requests'] = microbatch
        changed = True
    root = DATA_ROOT.rstrip('/') or '/data'
    if root != '/data':
        cfg = _repoint(cfg, root)
        changed = True
    if not changed:
        return src
    d = os.path.join(tempfile.gettempdir(), 'beatfcn-run-plans')
    os.makedirs(d, exist_ok=True)
    tag = '' if root == '/data' else '-%s' % hashlib.md5(root.encode()).hexdigest()[:6]
    if isinstance(microbatch, int):
        tag += '-mb%d' % microbatch
    dst = os.path.join(d, plan_name.replace('.json', tag + '.json'))
    tmp = dst + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, dst)
    return dst


def _train_start(body):
    """05 页「开始训练」。参数逐项白名单校验，命令一律后端拼（前端传来的命令串一律无视），
    起在 tmux 会话 train 里。返回 (http_code, {'ok','command','note'})；启动失败带 tmux 的 stderr。"""
    def bad(note):
        return 400, {'ok': False, 'command': '', 'note': note}

    def want_int(key, lo, hi):
        v = body.get(key)
        if isinstance(v, bool):
            return None
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if not isinstance(v, int) or not lo <= v <= hi:
            return None
        return v

    workers = want_int('workers', 1, 16)
    hours = want_int('deadline_hours', 1, 168)
    plan = body.get('plan')
    device = body.get('device')
    dry_run = body.get('dry_run', False)
    flags = {k: body.get(k, False) for k in ('resume', 'save_optimizer', 'no_compile')}

    if workers is None:
        return bad('workers 要是 1–16 的整数（收到 %r）' % (body.get('workers'),))
    if hours is None:
        return bad('deadline_hours 要是 1–168 的小时数（收到 %r）' % (body.get('deadline_hours'),))
    if not isinstance(plan, str) or not plan:
        return bad('plan 要是 plans/evolve 里的文件名')
    if plan != os.path.basename(plan):
        return bad('plan 只接受文件名，不许带路径：%r' % plan)
    if not isinstance(dry_run, bool):
        return bad('dry_run 要是布尔')
    for k, v in flags.items():
        if not isinstance(v, bool):
            return bad('%s 要是布尔（收到 %r）' % (k, v))
    if device not in ('cuda', 'cpu'):
        return bad('device 只接受 cuda / cpu（收到 %r）' % (device,))
    # 训练起来以后顺手把进度推到飞书（可选）。地址只认飞书群机器人，别让它变成任意 URL 的起跳板。
    # 不再从网页收：启动服务时用环境变量给一次（BEATFCN_FEISHU_WEBHOOK=...），页面上不留痕迹
    feishu = os.environ.get('BEATFCN_FEISHU_WEBHOOK', '')
    if isinstance(feishu, str) and feishu.strip():
        feishu = feishu.strip()
        if len(feishu) > 300 or not re.match(
                r'https://open\.(feishu\.cn|larksuite\.com)/open-apis/bot/v2/hook/[\w-]+$', feishu):
            return bad('飞书地址只接受群机器人 webhook：https://open.feishu.cn/open-apis/bot/v2/hook/…')
    else:
        feishu = None
    known = {p['name']: p for p in train_plans()}
    if not known:
        return bad('%s 里没有可用的 json 计划，先放计划再开跑' % os.path.join(REPO, 'plans', 'evolve'))
    if plan not in known:
        return bad('plan 不在 plans/evolve 里：%r' % plan)

    # 步数上限就是这个方案自己写的 horizon（exposure 序列按这个长度造好，超了训练脚本会 ValueError）
    base_every = known[plan].get('light_checkpoint_every')
    horizon = known[plan].get('horizon_updates')
    up_hi = horizon if isinstance(horizon, int) and horizon > 0 else 2400
    updates = want_int('updates', 100, up_hi)
    if updates is None:
        return bad('updates 要是 100–%d 的整数（这个方案最多 %d 步；收到 %r）'
                   % (up_hi, up_hi, body.get('updates')))
    # 存点间隔不能超过这个方案自己的总步数：训练脚本的存点条件是 current % every == 0，
    # every 大于 updates 就永远命不中，中途一个快照都不会有（只剩结束那份）
    ck_hi = min(2400, up_hi)
    ck_every = want_int('checkpoint_every', 50, ck_hi) if 'checkpoint_every' in body else base_every
    if 'checkpoint_every' in body and ck_every is None:
        return bad('checkpoint_every 要是 50–%d 的整数步数（这个方案最多 %d 步；收到 %r）'
                   % (ck_hi, up_hi, body.get('checkpoint_every'),))
    # 一次前向塞几条（显存）：1/2/4 是单条流，8/16 走分组打包，等效 batch 不变。
    # 只认这几个值，训练脚本和 checkpoint 校验也是这一套。默认 4 = 不写进方案。
    micro = body.get('microbatch', 4)
    if isinstance(micro, bool) or not isinstance(micro, int) or micro not in (1, 2, 4, 8, 16):
        return bad('microbatch 只能是 1 / 2 / 4 / 8 / 16（收到 %r）' % (micro,))
    plan_path = os.path.join(REPO, 'plans', 'evolve', plan)
    ck_note = ''
    root = (DATA_ROOT or '/data').rstrip('/') or '/data'
    every_changed = isinstance(ck_every, int) and ck_every != base_every
    mb_changed = micro != 4
    if every_changed or mb_changed or root != '/data':
        try:
            plan_path = _plan_copy_with_checkpoint(
                plan, ck_every if every_changed else None, micro if mb_changed else None)
        except Exception as e:
            return bad('生成临时方案失败：' + str(e))
        bits = []
        if every_changed:
            bits.append('存点间隔改成每 %d 步' % ck_every)
        if mb_changed:
            bits.append('一次性塞 %d 条' % micro)
        if root != '/data':
            bits.append('数据目录指到 %s' % root)
        ck_note = '（%s：已生成临时方案 %s，原方案没动）' % ('；'.join(bits), plan_path)

    deadline_ts = int(time.time() + hours * 3600)
    q = shlex.quote
    cmd = ('cd {repo} && BEATFCN_MAX_LOADER_WORKERS={workers} python -m scripts.evolve_finetune '
           '--plan {plan} --updates {updates} --deadline-unix {deadline}'
           .format(repo=q(REPO), workers=workers,
                   plan=q(plan_path),
                   updates=updates, deadline=deadline_ts))
    if flags['resume']:
        cmd += ' --resume'
    if flags['save_optimizer']:
        cmd += ' --save-optimizer'
    if flags['no_compile']:
        cmd += ' --no-compile'
    cmd += ' --device {dev} 2>&1 | tee -a {log}'.format(dev=device, log=q(os.path.join(REPO, 'train.log')))

    if dry_run:
        return 200, {'ok': True, 'command': cmd,
                     'note': 'dry_run：只生成命令，未检查 tmux、未启动'
                             + ('；也没起飞书上报' if feishu else '') + ck_note}

    try:
        probe = subprocess.run(['tmux', 'has-session', '-t', 'train'],
                               capture_output=True, text=True, timeout=8)
    except FileNotFoundError:
        return 200, {'ok': False, 'command': cmd, 'note': '机器上没有 tmux：sudo apt install -y tmux 装上再点'}
    except Exception as e:
        return 200, {'ok': False, 'command': cmd, 'note': 'tmux has-session 没跑起来：' + str(e)}
    if probe.returncode == 0:
        return 200, {'ok': False, 'command': cmd, 'note': '已有训练在跑，先 tmux attach -t train 看看'}
    try:
        started = subprocess.run(['tmux', 'new-session', '-d', '-s', 'train', cmd],
                                 capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return 200, {'ok': False, 'command': cmd, 'note': '机器上没有 tmux：sudo apt install -y tmux 装上再点'}
    except Exception as e:
        return 200, {'ok': False, 'command': cmd, 'note': 'tmux new-session 没跑起来：' + str(e)}
    if started.returncode != 0:
        err = (started.stderr or started.stdout or '').strip()
        return 200, {'ok': False, 'command': cmd,
                     'note': 'tmux 启动失败：' + (err or '退出码 %d，无输出' % started.returncode)}
    report_note = ''
    if feishu:
        # 上报独立跑在一个 tmux 会话里，挂了不影响训练；重启时先清掉旧的
        rcmd = ('cd {repo} && python3 -m scripts.report_progress --plan {plan} --out-dir {out} '
                '--webhook {hook} --interval 120').format(
            repo=q(REPO), plan=q(os.path.join(REPO, 'plans', 'evolve', plan)),
            out=q(known[plan].get('out_dir') or ''), hook=q(feishu))
        try:
            subprocess.run(['tmux', 'kill-session', '-t', 'report'], capture_output=True, timeout=8)
        except Exception:
            pass
        try:
            rr = subprocess.run(['tmux', 'new-session', '-d', '-s', 'report', rcmd],
                                capture_output=True, text=True, timeout=10)
            report_note = ('；进度每 2 分钟推到飞书群' if rr.returncode == 0
                           else '；飞书上报没起来（' + (rr.stderr or '').strip()[:60] + '）')
        except Exception as e:
            report_note = '；飞书上报没起来（%s）' % e
    return 200, {'ok': True, 'command': cmd,
                 'note': '已在 tmux 会话 train 里开跑；tmux attach -t train 看屏幕，tail -f train.log 看日志'
                         + report_note + ck_note}


# ---------------- 机器状态（05 页的资源小图） ----------------

_CPU_LOCK = threading.Lock()
_CPU_PREV = {'total': None, 'idle': None}


def _sys_cpu():
    """两次读 /proc/stat 求差。第一次读不到差值就先睡 0.15 秒再读一遍。"""
    for attempt in (0, 1):
        try:
            with open('/proc/stat') as f:
                parts = f.readline().split()
            vals = [int(x) for x in parts[1:]]
        except Exception:
            return None, None, None
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        with _CPU_LOCK:
            prev_total, prev_idle = _CPU_PREV['total'], _CPU_PREV['idle']
            _CPU_PREV.update(total=total, idle=idle)
        if prev_total is not None and total > prev_total:
            dt, di = total - prev_total, idle - prev_idle
            pct = round(max(0.0, min(100.0, 100.0 * (dt - di) / dt)), 1) if dt else None
            try:
                load1 = round(os.getloadavg()[0], 2)
            except Exception:
                load1 = None
            return pct, os.cpu_count(), load1
        if attempt == 0:
            time.sleep(0.15)
    return None, os.cpu_count(), None


def _sys_mem():
    try:
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':', 1)
                info[k] = int(v.split()[0])
        total = info['MemTotal']
        avail = info.get('MemAvailable', info.get('MemFree', 0))
        used = total - avail
        return {'used_mb': used // 1024, 'total_mb': total // 1024,
                'pct': round(100.0 * used / total, 1) if total else None}
    except Exception:
        return None


def _sys_gpu():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    rows = []
    for line in (out.stdout or '').strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            continue
        try:
            used, total = int(parts[2]), int(parts[3])
            rows.append({'index': int(parts[0]), 'util': int(parts[1]),
                         'mem_used_mb': used, 'mem_total_mb': total,
                         'pct': round(100.0 * used / total, 1) if total else None,
                         'temp_c': int(parts[4])})
        except Exception:
            continue
    return rows or None


def _sys_disk():
    try:
        u = shutil.disk_usage(REPO)
        return {'used_gb': round(u.used / 2 ** 30, 1), 'total_gb': round(u.total / 2 ** 30, 1),
                'pct': round(100.0 * u.used / u.total, 1) if u.total else None}
    except Exception:
        return None


def sys_payload():
    pct, cores, load1 = _sys_cpu()
    return {'ok': True, 'ts': time.time(),
            'cpu': {'pct': pct, 'cores': cores, 'load1': load1},
            'mem': _sys_mem(), 'gpu': _sys_gpu(), 'disk': _sys_disk()}


# ---------------- 训练输出（05 页的 TRAIN OUT） ----------------

def _run_all(cmd, timeout=10):
    """跑一条命令拿整段 stdout（_run 只回第一行，这里要全量）。失败返回 ''。"""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout or ''
    except Exception:
        return ''


def _train_log_tail():
    """抓训练输出尾部。来源按优先级：
    1. 环境变量 BEATFCN_TRAIN_LOG 指的日志文件
    2. stages.json 里各阶段声明的 "log" 路径
    3. 约定位置：本目录 / 代码根下的 train.log、reports/train_console.log
    4. tmux 会话 train 的屏幕缓冲（页面本来就教同学在 tmux new -s train 里起训练）
    返回 (来源描述, 行列表, 文件年龄秒或 None)；什么都没有时来源为空串。"""
    candidates = []
    env_log = os.environ.get('BEATFCN_TRAIN_LOG')
    if env_log:
        candidates.append(env_log)
    cfg = os.path.join(HERE, 'stages.json')
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding='utf-8') as f:
                stages_cfg = json.load(f)
            candidates += [s['log'] for s in stages_cfg if s.get('log')]
        except Exception:
            pass
    candidates += [os.path.join(HERE, 'train.log'),
                   os.path.join(REPO, 'train.log'),
                   os.path.join(REPO, 'reports', 'train_console.log')]
    seen = set()
    for raw_path in candidates:
        path = os.path.abspath(os.path.expanduser(str(raw_path)))
        if path in seen:
            continue
        seen.add(path)
        try:
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                continue
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - TRAIN_READ_BYTES))
                lines = f.read().decode('utf-8', 'replace').replace('\r', '\n').splitlines()
            lines = [l for l in lines if l.strip()][-TRAIN_TAIL_LINES:]
            if lines:
                return ('日志 ' + path, lines, round(max(0, time.time() - os.path.getmtime(path)), 1))
        except Exception:
            continue
    raw = _run_all(['tmux', 'capture-pane', '-p', '-t', 'train', '-S', '-4000'], timeout=5)
    if raw:
        lines = [l.rstrip() for l in raw.replace('\r', '\n').splitlines()]
        lines = [l for l in lines if l.strip()][-TRAIN_TAIL_LINES:]
        if lines:
            return ('tmux 会话 train', lines, None)
    return ('', [], None)


_PAIR_RE = re.compile(
    r'([A-Za-z_][A-Za-z0-9_.]*)\s*[=:]\s*('
    r'-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[Tt]rue|[Ff]alse|[Nn]a[Nn]|[Ii]nf(?:inity)?)')


def _flatten(obj, prefix='', depth=0):
    """训练记录递归摊平成标量键值：子字典逐层用点号展开（如 fields.request.loss、
    eval.full_note_f1），最深 4 层；再深的或 list 压成短 JSON 串——参数一个不落，
    但别让单元格爆炸。非有限浮点转成字符串，浏览器 JSON 才收。"""
    out = {}
    for k, v in obj.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, bool):
            out[key] = v
        elif isinstance(v, (int, float)):
            out[key] = 'NaN' if v != v else ('inf' if v == float('inf')
                                             else ('-inf' if v == float('-inf') else v))
        elif isinstance(v, dict) and depth < 3:
            out.update(_flatten(v, key, depth + 1))
        else:
            text = (json.dumps(v, ensure_ascii=False, default=str)
                    if isinstance(v, (dict, list)) else str(v))
            out[key] = text[:80] + ('…' if len(text) > 80 else '')
    return out


_pg_cache = {'t': 0.0, 'v': {}}   # (时间戳, 结果)：页面 5 秒一轮，别每轮重新扫 11 个 out_dir


def _run_progress():
    """当前那次训练的 progress.json（各方案 out_dir 里 mtime 最新的那份），只读。
    训练器自己每 30 秒落盘，含 gpu_peak_mib / eta_s —— 页面上「这一跑到底吃了多少显存」就靠它。"""
    now = time.time()
    if now - _pg_cache['t'] < 4.0:
        return _pg_cache['v']
    newest = None
    try:
        for p in train_plans():
            d = p.get('out_dir')
            if not d:
                continue
            f = os.path.join(d, 'progress.json')
            try:
                m = os.path.getmtime(f)
            except OSError:
                continue
            if newest is None or m > newest[0]:
                newest = (m, f)
    except Exception:
        newest = None
    out = {}
    if newest is not None:
        try:
            with open(newest[1], encoding='utf-8') as fh:
                s = json.load(fh)
            if isinstance(s, dict):
                for k in ('phase', 'updates', 'target_updates', 'start_updates', 'gpu_peak_mib',
                          'rate_updates_per_s', 'recent_updates_per_s', 'eta_s', 'loss', 'lr',
                          'deadline_unix', 'written_unix'):
                    if k in s:
                        out[k] = s[k]
        except Exception:
            out = {}
    _pg_cache.update(t=now, v=out)
    return out


def _train_metrics(lines):
    """从尾部往上找第一条带指标的行，解析出全部参数。行里有 JSON 记录（本仓库训练器
    按行吐 {"event":"train","step":...}）就用 JSON；否则按 key=value / key: value 文本对
    解析，普通 print 的进度行也吃得下。找不到返回空表。"""
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        start = line.find('{')
        if start >= 0:
            try:
                rec = json.loads(line[start:])
            except Exception:
                rec = None
            if isinstance(rec, dict):
                flat = _flatten(rec)
                if len(flat) >= 2 and any(isinstance(v, (int, float)) for v in flat.values()):
                    return flat
        pairs = _PAIR_RE.findall(line)
        if len(pairs) >= 2:
            out = {}
            for k, v in pairs:
                low = v.lower()
                if low == 'true':
                    out[k] = True
                elif low == 'false':
                    out[k] = False
                elif low in ('nan', 'inf', 'infinity', '-inf', '-infinity'):
                    out[k] = low
                else:
                    try:
                        out[k] = int(v)
                    except ValueError:
                        out[k] = float(v)
            if len(out) >= 2:
                return out
    return {}


# ---------------- 验证侧指标（训练进度行不打 F1，验证事件里才有） ----------------

def _run_validations(run):
    """一个 run 记录里的全部验证记录。落盘形态（train_monitor.snapshot 的输出）会把
    validations 摘进 model_progress（latest_validation / baseline / history），内存/
    旧形态则留在 run['validations'][step]——两种都认，只收真实存在的记录。"""
    records = []
    vs = run.get('validations')
    if isinstance(vs, dict):
        records += [v for v in vs.values() if isinstance(v, dict)]
    mp = run.get('model_progress')
    if isinstance(mp, dict):
        for key in ('latest_validation', 'baseline'):
            if isinstance(mp.get(key), dict):
                records.append(mp[key])
        if isinstance(mp.get('history'), list):
            records += [v for v in mp['history'] if isinstance(v, dict)]
    return records


def _latest_validation():
    """挑一条最新验证记录：优先 reports/training_monitor.json 里"时间最新的 run 中
    step 最大"的 validation（结构见 scripts/train_monitor.py；文件本体是单 run 快照，
    rhythm 场景另有 paired_runs，旧/内存形态是 runs[run_id].validations[step]）；
    没有就退到 reports/*.jsonl 里最后的 {"event":"val"} 行（每个文件取最后一条，再比
    step/ts）。哪路都没有返回 None——宁缺勿造。"""
    try:
        with open(os.path.join(REPO, 'reports', 'training_monitor.json'), encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = None
    if isinstance(data, dict):
        runs = [data]
        if isinstance(data.get('runs'), dict):
            runs += [v for v in data['runs'].values() if isinstance(v, dict)]
        if isinstance(data.get('paired_runs'), list):
            runs += [v for v in data['paired_runs'] if isinstance(v, dict)]
        best = None
        for run in runs:
            recs = _run_validations(run)
            if not recs:
                continue
            metrics = run.get('metrics')
            run_time = max([run.get('last_event_at') or 0,
                            (metrics.get('recorded_at') if isinstance(metrics, dict) else 0) or 0]
                           + [r.get('ts') or 0 for r in recs])
            if best is None or run_time > best[0]:
                best = (run_time, max(recs, key=lambda r: (r.get('step') or 0, r.get('ts') or 0)))
        if best:
            return best[1]
    reports_dir = os.path.join(REPO, 'reports')
    try:
        names = sorted(n for n in os.listdir(reports_dir) if n.endswith('.jsonl'))
    except OSError:
        return None
    best = None
    for name in names:
        try:
            with open(os.path.join(reports_dir, name), 'rb') as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - TRAIN_READ_BYTES))
                tail = f.read().decode('utf-8', 'replace')
        except OSError:
            continue
        for line in reversed(tail.replace('\r', '\n').splitlines()):
            start = line.find('{')
            if start < 0:
                continue
            try:
                rec = json.loads(line[start:])
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get('event') == 'val':
                key = (rec.get('step') or 0, rec.get('ts') or 0)
                if best is None or key > best[0]:
                    best = (key, rec)
                break
    return best[1] if best else None


def _val_metrics():
    """验证侧指标 → "val." 前缀键值（val.step / val.full_note_f1 / val.onset_f1_50ms…），
    数值/布尔字段一个不落（沿用 _flatten 的摊平和非有限值转字符串规矩，字符串类字段
    如 event/run_id 不进表）。来源什么都没有就给空表，绝不编造。"""
    rec = _latest_validation()
    if not isinstance(rec, dict):
        return {}
    out = {}
    for k, v in _flatten(rec).items():
        if isinstance(v, (bool, int, float)) or v in ('NaN', 'inf', '-inf'):
            out['val.' + k] = v
    if isinstance(rec.get('step'), (bool, int, float)):
        out.setdefault('val.step', rec['step'])
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = 'beatfcn-console/1.0'

    def _authed(self):
        if not TOKEN:
            return True
        q = parse_qs(urlparse(self.path).query)
        got = (q.get('token') or [''])[0]
        if not got:
            got = self.headers.get('X-Token') or ''
        return got == TOKEN

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204, b'', 'text/plain')

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            for name in HTML_NAMES:
                p = os.path.join(HERE, name)
                if os.path.isfile(p):
                    with open(p, 'rb') as f:
                        return self._send(200, f.read(), 'text/html; charset=utf-8')
            return self._send(404, {'ok': False, 'note': 'HTML 文件不在这个目录'})
        if path == '/api/ping':
            return self._send(200, {'ok': True, 'authed': self._authed(),
                                    'host': os.uname().nodename, 'ts': time.time()})
        if path in ('/api/env', '/api/status', '/api/trainlog', '/api/train-plan', '/api/sys'):
            if not self._authed():
                return self._send(401, {'ok': False, 'note': 'wrong token: 网址后加 ?token=访问码再回车'})
            if path == '/api/env':
                return self._send(200, {'ok': True, 'items': get_env()})
            if path == '/api/train-plan':
                return self._send(200, {'ok': True, 'plans': train_plans(), 'repo': REPO})
            if path == '/api/sys':
                return self._send(200, sys_payload())
            if path == '/api/trainlog':
                src, lines, age = _train_log_tail()
                # 解析用完整行，回显的单行截到 400 字符，别让一条超长行撑爆手机流量
                shown = [(l[:400] + '…') if len(l) > 400 else l for l in lines]
                metrics = _train_metrics(lines)
                metrics.update(_val_metrics())   # 训练行里没有的验证指标（F1 等）以 val. 前缀并入
                return self._send(200, {'ok': True, 'source': src, 'age_s': age,
                                        'lines': shown, 'metrics': metrics,
                                        'progress': _run_progress(),
                                        'ts': time.time()})
            return self._send(200, {'ok': True, **get_status()})
        return self._send(404, {'ok': False, 'note': 'no such endpoint'})

    def do_POST(self):
        path = self.path.split('?')[0]
        if path != '/api/action':
            return self._send(404, {'ok': False, 'note': 'no such endpoint'})
        if not self._authed():
            return self._send(401, {'ok': False, 'note': 'wrong token: 网址后加 ?token=访问码'})
        try:
            n = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(min(n, 10240)) or b'{}')
        except Exception:
            return self._send(400, {'ok': False, 'note': 'bad json'})
        action = str(body.get('id') or '')
        if action == 'env-recheck':
            _cache['env'] = (0.0, None)
            _cache['torch'] = (0.0, None)
            return self._send(200, {'ok': True, 'items': get_env()})
        if action == 'train-start':
            code, resp = _train_start(body)
            return self._send(code, resp)
        # 阶段动作：只允许 stages.json 里声明过的 start/stop，且命令逐字来自声明
        path_cfg = os.path.join(HERE, 'stages.json')
        if os.path.isfile(path_cfg):
            try:
                for s in json.load(open(path_cfg, encoding='utf-8')):
                    if s.get('id') == action and body.get('op') in ('start', 'stop'):
                        cmd = s.get(body['op'])
                        if cmd:
                            subprocess.Popen(['bash', '-lc', cmd], cwd=s.get('cwd') or HERE,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                             start_new_session=True)
                            return self._send(200, {'ok': True, 'started': body['op'], 'stage': action})
            except Exception as e:
                return self._send(500, {'ok': False, 'note': str(e)})
        return self._send(400, {'ok': False, 'note': 'unknown or disabled action: ' + action})

    def log_message(self, fmt, *args):
        sys.stderr.write('[console] %s %s\n' % (self.address_string(), fmt % args))


def _usable_ip(ip):
    if not ip:
        return False
    if ip.startswith('127.') or ip.startswith('169.254.'):
        return False
    return bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip))


def _lan_ips():
    """能给别人用的地址：(Windows 局域网地址, 本机地址)。

    WSL 里 `hostname -I` 给的是虚拟网卡（172.x），手机根本连不上，
    所以额外问一次 Windows 的 ipconfig，把真正的局域网 IP 捞出来。
    """
    ips = [ip for ip in (_run('hostname -I', timeout=5) or '').split() if _usable_ip(ip)]
    win = []
    try:
        ver = open('/proc/version', encoding='utf-8', errors='ignore').read().lower()
    except Exception:
        ver = ''
    is_wsl = 'microsoft' in ver or bool(os.environ.get('WSL_DISTRO_NAME'))
    if is_wsl:
        ic = '/mnt/c/Windows/System32/ipconfig.exe'
        if os.path.exists(ic):
            for line in _run_all([ic], timeout=8).splitlines():
                if 'IPv4' in line or 'ipv4' in line:
                    m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line)
                    if m and _usable_ip(m.group(1)) and m.group(1) not in win:
                        win.append(m.group(1))
    # 家里的网段优先，WSL 自己的 172.x 排最后
    win.sort(key=lambda ip: 0 if ip.startswith('192.168.') else (1 if ip.startswith('10.') else 2))
    return win, ips, is_wsl


def main():
    global TOKEN
    args = sys.argv[1:]
    port = 7777
    if args and args[0].isdigit():
        port = int(args.pop(0))
    if '--token' in args:
        i = args.index('--token')
        if i + 1 < len(args):
            TOKEN = args[i + 1]
    srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    win, ips, is_wsl = _lan_ips()
    print(f'[console] 后端已启动: http://localhost:{port}')
    print(f'[console] 访问码: ' + (TOKEN or '未设置（仅限局域网/本机用）'))
    for ip in (win or ips):
        print(f'[console] 手机/其他设备: http://{ip}:{port}')
    if win and ips:
        print('[console] 上面是 Windows 的局域网地址；WSL 里的 ' + '、'.join(ips) + ' 手机连不上，别用')
    elif is_wsl:
        print('[console] 注意：上面可能是 WSL 虚拟网卡，手机连不上。手机要用 Windows 的')
        print('[console] 局域网 IP（Windows 里 ipconfig 看「WLAN」那行的 IPv4），格式 http://那个IP:'
              + str(port))
    threading.Thread(target=_torch_probe, daemon=True).start()  # 预热：首次 import 别让页面等
    print('[console] Ctrl+C 停止。页面会自动退回手动模式，不影响训练。')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n[console] 已停止。')


if __name__ == '__main__':
    main()
