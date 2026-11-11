#!/usr/bin/env bash
# 操作台后端一键启动/回到后台
# 用法（WSL 里）:  bash 启动控制台.sh
# 已在跑就会直接接回那个窗口（不会起第二个）。

cd "$(dirname "$0")"
exec tmux new -A -s console 'python3 console_server.py 7777'
