#!/usr/bin/env bash
# 插上硬盘就能跑：所有路径都从本脚本所在目录算，不认死 /home/user。
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT="${PORT:-7777}"
PY="${PYTHON:-python3}"
mkdir -p "$ROOT/.harness/logs" "$ROOT/data"

# 自带的 Python 优先：整盘带走时连解释器都不用装
USE_RUNTIME=0
if [ -x "$ROOT/runtime/python/bin/python3" ] && "$ROOT/runtime/python/bin/python3" -c "import sys" >/dev/null 2>&1; then
  PY="$ROOT/runtime/python/bin/python3"
  USE_RUNTIME=1
  export PYTHONNOUSERSITE=1     # 别借同学 ~/.local 里的包，版本会打架
  unset PYTHONPATH
  export PATH="$ROOT/runtime/bin:$PATH"   # 盘里的 ffmpeg
fi

# 没自带解释器时，再用 .venv 里的包（PYTHONPATH 带，换挂载点也不怕）
if [ "$USE_RUNTIME" = 0 ]; then
  SITE="$(ls -d "$ROOT"/.venv/lib/python3.*/site-packages 2>/dev/null | head -1 || true)"
  if [ -n "$SITE" ]; then
    export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
  fi
  if [ -x "$ROOT/.venv/bin/python3" ] && "$ROOT/.venv/bin/python3" -c "import sys" >/dev/null 2>&1; then
    PY="$ROOT/.venv/bin/python3"
  fi
fi

export BEATFCN_DATA_ROOT="${BEATFCN_DATA_ROOT:-$ROOT/data}"
export BEATFCN_TRAIN_LOG="$ROOT/.harness/logs/phira-train.log"
export HF_HOME="${HF_HOME:-$ROOT/data/cache/hf}"          # 模型权重也留在盘上，不写同学 ~/.cache
export TORCH_HOME="${TORCH_HOME:-$ROOT/data/cache/torch}"
export PYTHONUNBUFFERED=1

echo "仓库：$ROOT"
echo "数据：$BEATFCN_DATA_ROOT"
echo "打开：http://127.0.0.1:$PORT/"
exec "$PY" console_server.py "$PORT"
