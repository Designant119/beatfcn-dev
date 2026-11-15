#!/usr/bin/env bash
# 插上硬盘就能跑：所有路径都从本脚本所在目录算，不认死 /home/user。
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT="${PORT:-7777}"
PY="${PYTHON:-python3}"
mkdir -p "$ROOT/.harness/logs" "$ROOT/data"

# .venv 里的包用 PYTHONPATH 带进来：硬盘换挂载点、venv 的 shebang 失效时也能用
SITE="$(ls -d "$ROOT"/.venv/lib/python3.*/site-packages 2>/dev/null | head -1 || true)"
if [ -n "$SITE" ]; then
  export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
fi
if [ -x "$ROOT/.venv/bin/python3" ] && "$ROOT/.venv/bin/python3" -c "import sys" >/dev/null 2>&1; then
  PY="$ROOT/.venv/bin/python3"
fi

export BEATFCN_DATA_ROOT="${BEATFCN_DATA_ROOT:-$ROOT/data}"
export BEATFCN_TRAIN_LOG="$ROOT/.harness/logs/phira-train.log"
export PYTHONUNBUFFERED=1

echo "仓库：$ROOT"
echo "数据：$BEATFCN_DATA_ROOT"
echo "打开：http://127.0.0.1:$PORT/"
exec "$PY" console_server.py "$PORT"
