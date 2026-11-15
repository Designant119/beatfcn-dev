#!/usr/bin/env bash
# 把 Python + torch + transformers 全部塞进硬盘的 runtime/，同学机器不用装任何东西。
# 用法：
#   bash scripts/build_portable_runtime.sh                 # CPU 版（任何机器都能跑演示）
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/build_portable_runtime.sh   # 4090
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# 关键：别让本机的 ~/.local 借包给新解释器，否则“装成功了”其实没进盘
export PYTHONNOUSERSITE=1
unset PYTHONPATH
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

echo "装 ffmpeg（音频那步要用）…"
if [ ! -x "$DEST/bin/ffmpeg" ]; then
  TMP_FF="$(mktemp -d)"
  curl -fL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz" | tar -xJ -C "$TMP_FF"
  mkdir -p "$DEST/bin"
  cp "$(find "$TMP_FF" -type f -name ffmpeg | head -1)" "$DEST/bin/ffmpeg"
  cp "$(find "$TMP_FF" -type f -name ffprobe | head -1)" "$DEST/bin/ffprobe"
  chmod +x "$DEST/bin/ffmpeg" "$DEST/bin/ffprobe"
  rm -rf "$TMP_FF"
fi

echo "装其余依赖…"
"$PY" -m pip install -r "$ROOT/requirements.txt"

env -i HOME=/tmp PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import torch, transformers, numpy, sys
assert '/runtime/' in torch.__file__ or str(sys.prefix) in torch.__file__, torch.__file__
print('runtime OK:', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__, '|', torch.__file__)
PY
echo "完成：$DEST"
