#!/usr/bin/env bash
# 在硬盘上建一套独立 Python 环境（.venv），换电脑不用装系统的包。
# CPU 机器：bash scripts/portable_env.sh
# NVIDIA：  TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/portable_env.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
"$PY" -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/pip install torch torchaudio --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
./.venv/bin/pip install -r requirements.txt
echo "装好了：$ROOT/.venv"
