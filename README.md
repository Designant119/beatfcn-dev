# BeatFCN

给一首歌，自动生成 Phigros / Phira / osu!mania 可用的**完整事件谱**（时间点 + 类型 + 位置布局）。
核心是 `beatfcn/` 包：从音频特征（PupuM2D / MERT2）出发，用事件级模型直接出规范事件表示，再导出成各平台谱面。

> 旧的 v22 / v23 / v24「节拍骨架线」（`model.py`、`beat_fcn_train_v22~v24*.py`、`infer_v22.py`、
> `reval_v22.py`、`gen/`、`archive/` 等）已在 2026-11-11 整体移除。该线只出按键时间点、不做位置与判定线，
> 与当前事件谱系统没有代码往来；需要时从 git 历史取回。

## 插盘就跑（整盘带走）

代码、数据、Python 环境都放在同一块硬盘上，换台机器插上就能起，不依赖本机的 `/data` 或 `~/.local`：

```bash
bash scripts/build_portable_runtime.sh   # 只跑一次：把 Python + torch + transformers 全塞进盘里的 runtime/
# 4090 想要 CUDA 版：TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/build_portable_runtime.sh
bash run.sh                       # 起控制台，默认 http://127.0.0.1:7777/
```

- 解释器在 `<仓库>/runtime/python`，包在它自己的 site-packages；同学机器不用装 Python
- 音频那步要的 ffmpeg 也在 `runtime/bin`，启动脚本自动加进 PATH
- 所有数据默认落在 `<仓库>/data`；想换位置设 `BEATFCN_DATA_ROOT`
- 模型缓存也留在盘上：`data/cache/hf`、`data/cache/torch`
- 路径记忆存的是**相对仓库**的路径，硬盘挂载点从 `/mnt/e` 变成 `/media/usb` 也不会指丢
- Windows 下双击 `run.bat`（走 WSL）
- 硬件的 `/data` 存在时仍优先用它（4090 那台习惯）

## ⚠️ 安全提示：仓库里有 API Key

`phira_chat.json` 里的 `key_b64` 只是 **base64 编码，不是加密**——任何人拿到仓库都能还原成明文。
这个仓库必须保持 **private**，不要 fork 成公开、不要把文件粘到 issue / 截图里。
要轮换：去智谱后台重新生成 key，改 `phira_chat.json` 与 `~/.config/phira/chat.json` 后重启控制台。

## 仓库结构

| 路径 | 作用 |
|---|---|
| `beatfcn/events/` | 数据契约层：规范事件表示（整数微秒 + 精确分数拍）、osu!mania 4K / Phira RPE 解析与导出、规则式双手布局 |
| `beatfcn/model/` | 事件模型（P1 / HN2 原生元组状态 / 事件画布等）与编解码、评测 |
| `beatfcn/train/` | 训练环：暴露装载、EMA、LR 曲线、KD、LayerDrop、损失面板、实验战役 |
| `beatfcn/train/peer_rl.py` | peer RL：自采样 REINFORCE，奖励只看可玩性规则（违反记 -1），不用人类谱 |
| `beatfcn/audio/` | 音频特征：本地 / CUDA mel、MERT2 批量编码、PupuM2D、缓存 |
| `beatfcn/ingest/` | 摄取：谱面下载解析、offset、timing、覆盖统计 |
| `beatfcn/spatial/` | 空间侧：RPE 坐标、上下文 |
| `beatfcn/cloud/` | Modal 云作业（摄取 / 编码 / 训练 / 进化轮）。需 Owner 显式授权才启动 |
| `scripts/` | 运行入口与运维：`evolve_finetune.py`（有界微调）、`evolve_surgery.py`（改深/改 FFN 的外科手术）、`draft_song.py`（出谱）、`benchmark_v3.py`（W/H/M 评分）、`ops_autodl/`（租卡无人值守，到点自动关机） |
| `tests/` | 274 个回归测试（205 个直接 import `beatfcn`），与 scripts 一一对应 |
| `editor/` | 编辑器推理入口：`infer.py`（出谱 + 出口二选一）、`ms_to_beats.py`（毫秒 → 拍分数精确换算，纯标准库自测） |
| `plans/evolve/` | 11 个预注册实验计划（C1–C9 + peer RL R1） |
| `bench/` | 评测口径与历史指标数据；`beatfcn/cloud/*` 有引用 |
| `phira_all_charts.jsonl` | Phira 谱面目录，`beatfcn/cloud/ingest_app.py` 使用 |
| `同学训练操作页.html` + `console_server.py` | 给同学用的一键操作台：网页看机器/环境/数据/开跑状态，后端只读探活、不 import torch |
| `双击安装训练系统.bat` / `启动控制台.sh` | Windows 双击装 WSL 系统并起后端；Linux 侧起后端（tmux 会话 `console`） |

## r5 现状

r5 起模型只做**采音**：mania 轨道输出（lane）全链与输入端 lane 特征已删除，位置交给下游规则/编辑器。
同代清理也移除了云端旧模块与一批 probe 脚本，只保留 `beatfcn/train/cohorts.py`（队列/cohort 工具）与
`beatfcn/train/deadlines.py`（截止时间工具）这两个独立工具。

```bash
# 有界微调（累积 --updates，每段都是同一轨迹的前缀，可 --resume）
BEATFCN_ALLOW_MISSING_MERT=1 python -m scripts.evolve_finetune --plan PLAN.json --updates 300 --deadline-unix T

# 候选外科手术：少几层 decoder / 窄 FFN
python -m scripts.evolve_surgery --in ckpt.pt --out CAND.pt --depth 8 --ffn 1792

# 出谱（编辑器集成默认：unconditioned 自然采音 + 不做可玩性清理 + 单采样）
python -m scripts.draft_song --audio S --out-dir D --mode phira --profile none --consensus 1

# 编辑器推理：换算版 / 纯毫秒版（无 torch 时可用 --from-draft 只测第二级）
python editor/infer.py --audio song.mp3 --bpm-map bpm.json --model ckpt.pt --out-dir out/
python editor/infer.py --audio song.mp3 --raw --model ckpt.pt --out-dir out/
```

## 给同学用的一键操作台

Windows 上双击 `双击安装训练系统.bat`：装 WSL（要重启一次）→ 从训练盘导入系统 → 起后端 → 自动打开页面。
Linux / WSL 里手动起：`bash 启动控制台.sh`，然后开 `http://localhost:7777`（手机连同一 WiFi 用 `http://本机IP:7777`）。

后端默认认 `~/beatfcn-dev` 为代码目录（换过路径设环境变量 `BEATFCN_REPO`），公网暴露时用 `--token 口令` 给 `/api/*` 加访问码。
页面不是训练必需项：关掉后端页面就退回手动模式，训练进程不受影响。

## 环境

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-fc.txt      # torch / numpy / safetensors / jsonschema / pyyaml / modal
sudo apt install -y ffmpeg curl
python3 -m unittest discover -s tests -q
```

仓库**不含**音频、特征 npz（约 95GB）和大部分权重；按产出线重新生成。云端作业只在 Owner 授权后启动。
