"""语料根目录的统一解析——本目录脚本不再出现任何机器/会话相关的绝对路径。

WSL 约定（见仓库 SETUP.md）：所有产物走 Linux 侧，默认根 /data；不要放 /mnt/e、/mnt/d。
优先级：
  1. PHIRA_WORK / PHIRA_OLD —— 单独覆盖这一份语料
  2. BEATFCN_DATA_ROOT 下的 phira_full / charts_by_rating（默认根 /data）
  3. 仓库下的同名目录

换机器、换盘、把语料挪走，都只改环境变量：

    export BEATFCN_DATA_ROOT=/data            # 一个变量管全部数据
    export PHIRA_WORK=/your/path/phira_full   # 可选：只单独指定 phira 语料
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_ROOT = os.environ.get('BEATFCN_DATA_ROOT') or '/data'


def _resolve(env_key, dirname):
    explicit = os.environ.get(env_key)
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    for base in (DATA_ROOT, REPO):
        candidate = os.path.join(base, dirname)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.join(DATA_ROOT, dirname)


WORK = _resolve('PHIRA_WORK', 'phira_full')
OLD = _resolve('PHIRA_OLD', 'charts_by_rating')


def require(path, env_key, what='语料根目录'):
    """脚本开跑前调用：目录不在就给出能照做的提示，而不是抛一长串 FileNotFoundError。"""
    if not os.path.isdir(path):
        raise SystemExit(
            f'{what}不存在：{path}\n'
            f'请先 export {env_key}=<该目录的绝对路径>，再重新运行。')
    return path
