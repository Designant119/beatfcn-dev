# BeatFCN 路径配置样板（WSL2）
#
# 用法：把需要的几行复制到 ~/.bashrc，或每次开终端前 `source env.example.sh`
# 约定（见 SETUP.md）：所有产物走 Linux 侧，别放 /mnt/e、/mnt/d —— NTFS 翻译层读写慢。
#
# 只有 BEATFCN_DATA_ROOT 是必须的，其余都是可选覆盖。

# ── 必须 ─────────────────────────────────────────────
export BEATFCN_DATA_ROOT=/data          # 数据根：tape / 特征 / checkpoint / phira_full 都在这下面

# ── 可选：代码根（默认跟 console_server.py 所在目录走，一般不用设）
# export BEATFCN_REPO=$HOME/beatfcn-dev

# ── 可选：phira 语料（不设就取 $BEATFCN_DATA_ROOT/phira_full）
# export PHIRA_WORK=$BEATFCN_DATA_ROOT/phira_full
# export PHIRA_OLD=$BEATFCN_DATA_ROOT/charts_by_rating

# ── 可选：时间轴表 / 训练日志 / 评测 floor
# export BEATFCN_TIMING=$BEATFCN_DATA_ROOT/chart_timing-v2.json
# export BEATFCN_TRAIN_LOG=$HOME/beatfcn-dev/train.log
# export BENCHMARK_V3_FLOOR=$HOME/beatfcn-dev/reports/evolve-20261104/benchmark_v3/floor.json

# ── 可选：控制台监听地址（默认只开本机；局域网访问显式给 0.0.0.0）
# export PHIRA_BIND=127.0.0.1
