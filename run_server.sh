#!/usr/bin/env bash
# 启动训练操作台。只有一个地方要改：FEISHU。
#
# FEISHU 填飞书群机器人地址（群设置 -> 群机器人 -> 自定义机器人 -> 复制 webhook），
# 填了以后每次「开始训练」都会自动每 2 分钟把进度推到那个群；留空就不推。
# 地址只写在这台机器上，网页里看不到，同学也不会看到。

FEISHU=""

PORT="${1:-7777}"
cd "$(dirname "$0")" || exit 1

export BEATFCN_REPO="$(pwd)"
[ -n "$FEISHU" ] && export BEATFCN_FEISHU_WEBHOOK="$FEISHU"

exec python3 console_server.py "$PORT"
