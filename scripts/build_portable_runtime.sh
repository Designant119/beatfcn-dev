#!/usr/bin/env bash
# 把 Python + torch + transformers 全部塞进硬盘的 runtime/，同学机器不用装任何东西。
# 用法：
#   bash scripts/build_portable_runtime.sh                 # CPU 版（任何机器都能跑演示）
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/build_portable_runtime.sh   # 4090
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${RUNTIME_DIR:-$ROOT/runtime}"
PY_VER="${PY_VER:-3.12.7}"
BUILD="${PY_BUILD:-20241016}"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${BUILD}/cpython-${PY_VER}%2B${BUILD}-x86_64-unknown-linux-gnu-install_only.tar.gz"

echo "目标：$DEST（Python $PY_VER）"
mkdir -p "$DEST"
if [ ! -x "$DEST/python/bin/python3" ]; then
  echo "下载独立 Python…"
  curl -fL "$URL" | tar -xz -C "$DEST"
fi
PY="$DEST/python/bin/python3"
"$PY" -m pip install -U pip wheel

echo "装 torch / torchaudio…"
"$PY" -m pip install torch torchaudio --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

echo "装其余依赖…"
"$PY" -m pip install -r "$ROOT/requirements.txt"

"$PY" - <<'PY'
import torch, transformers, numpy
print('runtime OK:', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__)
PY
echo "完成：$DEST"
