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
    POST /api/action    预留安全操作；目前仅支持 {"id":"env-recheck"}

设计约束:
    * 只用标准库，装完系统就能跑，先于 torch 存在
    * 只暴露只读探测 + 白名单动作，绝不提供删除类操作
    * 子进程全部带超时；torch 检测走子进程，本进程永不 import torch
"""
import json
import os
import re
import shutil
import subprocess
import sys
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
    """层04：数据在不在盘上——直接数文件，不看人嘴上说的"导好了"。"""
    if not os.path.isdir(REPO):
        return [_item('DATA', '数据目录', '', 'none',
                      f'没找到 {REPO}：代码还没拉下来，数据无从检查', 'data')]
    feat = tape = scored = 0
    seen = 0
    for root in (os.path.join(REPO, 'inputs'), os.path.join(REPO, 'reports')):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            seen += 1
            if seen > 3000:      # 盘大了别扫到底，够判断就行
                break
            names = set(filenames)
            if 'L12.npy' in names and 'L23.npy' in names:
                feat += 1
            tape += sum(1 for f in filenames if f.endswith('.jsonl'))
        if seen > 3000:
            break
    items = []
    items.append(_item('FEAT', '特征目录', f'{feat} 首', 'pass' if feat else 'bad',
                       '' if feat else 'inputs 里没有 L12.npy + L23.npy 齐的目录', 'data'))
    items.append(_item('TAPE', '清单 jsonl', f'{tape} 份', 'pass' if tape else 'bad',
                       '' if tape else 'inputs / reports 里没有 jsonl 清单', 'data'))
    if not feat and not tape and seen:
        items.append(_item('DATA0', '数据', '空的', 'bad', '目录在但里面没数据', 'data'))
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
                           '数比 1T 的盘小很多，可能看的不是那块盘' if st == 'warn' else '', 'disk'))
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
        if path in ('/api/env', '/api/status'):
            if not self._authed():
                return self._send(401, {'ok': False, 'note': 'wrong token: 网址后加 ?token=访问码'})
            if path == '/api/env':
                return self._send(200, {'ok': True, 'items': get_env()})
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
    ips = []
    try:
        out = _run("hostname -I", timeout=5) or ''
        ips = out.split()
    except Exception:
        pass
    print(f'[console] 后端已启动: http://localhost:{port}')
    print(f'[console] 访问码: ' + (TOKEN or '未设置（仅限局域网/本机用）'))
    for ip in ips:
        print(f'[console] 手机/其他设备: http://{ip}:{port}')
    threading.Thread(target=_torch_probe, daemon=True).start()  # 预热：首次 import 别让页面等
    print('[console] Ctrl+C 停止。页面会自动退回手动模式，不影响训练。')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n[console] 已停止。')


if __name__ == '__main__':
    main()
