#!/usr/bin/env bash
# 启动训练操作台。要改的都在下面几行（FEISHU / DATA_ROOT），其余不用动。
#
# FEISHU 填飞书群机器人地址（群设置 -> 群机器人 -> 自定义机器人 -> 复制 webhook），
# 填了以后每次「开始训练」都会自动每 2 分钟把进度推到那个群；留空就不推。
# 地址只写在这台机器上，网页里看不到，同学也不会看到。

FEISHU=""

# 数据目录。方案里写死的是 /data/evolve/...，如果盘挂在别的地方，把这行改成实际路径，
# 网页点「开始训练」时会自动把方案里的 /data 换成这里（原方案文件不动）。
# 留空 = 用 /data。
DATA_ROOT=""

PORT="${1:-7777}"
cd "$(dirname "$0")" || exit 1

export BEATFCN_REPO="$(pwd)"
[ -n "$FEISHU" ] && export BEATFCN_FEISHU_WEBHOOK="$FEISHU"
[ -n "$DATA_ROOT" ] && export BEATFCN_DATA_ROOT="$DATA_ROOT"

exec python3 console_server.py "$PORT"
