# 从零到出谱面：环境与全流程

这份文档描述的是**这台机器上真实跑通过的路径**，每一步都有对应的产物可以检查。
不是理想化的说明，是踩过坑之后写下来的。

## 一、机器要求

| | 最低 | 推荐 |
|---|---|---|
| 系统 | Linux / WSL2 | WSL2 |
| Python | 3.11+ | 3.12 |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 200 GB（100 包试跑 + 特征） | 1 TB（全量 9760 包） |
| 显卡 | 不需要 | NVIDIA（提特征快 10 倍以上） |

**WSL2 的关键一条**：所有产物（谱面、音频、特征、checkpoint）必须放在 Linux 侧
（`/data`、`~/data` 这类路径），**不要放 `/mnt/e`、`/mnt/d`**。NTFS 经 WSL 翻译层读写
慢 3～5 倍，提特征这种大量小文件 IO 会被拖到没法用。

## 二、装依赖

```bash
cd beatfcn-dev
pip install -r requirements.txt
```

装完检查三样：

```bash
python3 -c "import torch, torchaudio, transformers; print(torch.__version__, torchaudio.__version__, transformers.__version__)"
ffmpeg -version | head -1
```

`torchaudio` 和 `torch` 大版本对不齐很常见（本机是 torch 2.15.0+cpu / torchaudio 2.11.1+cpu），
只要三个都能 import 就行，不影响使用。

**ffmpeg 必须有**。整套流程解码音频全靠它（`ffmpeg -i ... -f f32le`），不依赖 torchaudio 的解码。

## 三、下载 MERT 权重

**权重是强绑定的**，不是随便一个 MERT：

```
repo      m-a-p/MERT-v2-FullSong
revision  7001f7657276238629e61c7746ce70735e0604c0
文件       config.json
          model.safetensors
          configuration_mert2.py
          modeling_mert2.py
```

```bash
python3 scripts/download_mert.py --out /data/weights/mert
# 国内网络慢的话：
python3 scripts/download_mert.py --out /data/weights/mert --mirror https://hf-mirror.com
```

脚本会把四个文件逐一下载并校验 sha256，任何一个对不上就报错。
**不要试图下 `MERT-v1-95M` 之类的其他版本**——文件名和指纹都不一样，模型会拒绝加载。

## 四、准备谱面数据

```bash
# 全量下载（9760 包约 60 GB）；只想要高难可以加 --levels IN,AT
python3 scripts/phira_local_download.py --dest /data/phira_charts

# 或者只下 IN/AT（约 49 GB，省 11.5 GB）
python3 scripts/phira_local_download.py --dest /data/phira_charts --levels IN,AT

# 先看看有多少、多大，不真下（清单 selection.json 仍会写到 --dest 目录）
python3 scripts/phira_local_download.py --dest /data/phira_charts --list-only
```

## 五、五步流水线

```bash
DATA=/data/phira_charts
DB=/data/phira_db
AUDIO=/data/phira_audio
TAPE=/data/tape_min
FEAT=/data/features

# ① 建库：zip → canon + targets
python3 -m scripts.ingest_phira_dataset \
    --zips $DATA --meta phira_charts_meta.json --out $DB

# ② 摊平音频：zip 里的音轨 → <sha256>.<ext>
python3 -m scripts.extract_phira_audio \
    --zips $DATA --output $AUDIO

# ③ 最小 tape：canon → 提特征要的 <sha>-chart.json
python3 -m scripts.build_feature_tape \
    --canon $DB/canon/phira --output $TAPE

# ④ 提特征（最慢的一步）
python3 -m scripts.batch_extract_features \
    --tape-dir $TAPE --audio-dir $AUDIO --features-dir $FEAT \
    --mert-source /data/weights/mert --device cuda \
    --decode-workers 6 --log /data/extract.jsonl

# ⑤ tape → pool → exposure
python3 -m scripts.build_full_native_tape \
    --targets-root $DB/targets --features-root $FEAT/features \
    --output /data/tape --workers 8

python3 -m scripts.build_phira_exposure \
    --tape-dir /data/tape --canon $DB/canon/phira \
    --seed 20261108 --steps 2400 \
    --output-pool /data/evolve/pool.json \
    --output-exposure /data/evolve/exposure-20261108-2400u.json
```

## 六、训练与推理

```bash
# 极小闭环：2 步就能出一个能被推理加载的 ckpt
python3 -m scripts.train_phira_tiny \
    --tape /data/tape --features $FEAT/features \
    --steps 2 --out /data/ckpt/tiny.pt

# 推理：音频 + ckpt → Phira 谱面 zip
python3 -m scripts.draft_song \
    --audio $AUDIO/<sha256>.ogg \
    --out-dir /data/gen \
    --model /data/ckpt/tiny.pt \
    --mode phira --device cpu \
    --mert-source /data/weights/mert \
    --mel-stats <mel 归一化统计>
```

`--mel-stats` 是一个 json，格式：

```json
{"schema": "train-mel-channel-v1",
 "train_audios": ["<训练集里的 sha256>", "..."],
 "mean": [128 个浮点数],
 "std":  [128 个浮点数]}
```

它可以从任一 `features/<sha>/mel10_complete.npy` 按通道统计出来。

## 七、实测数字

| 环节 | 实测 | 备注 |
|---|---|---|
| ① ingest | 82 建成 / 18 失败，1.1 分钟 / 100 包 | 失败的 18% 是 `.json` 扩展名里装 PEC 文本 |
| ② 摊平音频 | 100 份 / 498 MB | 无扩展名的音频存成 `.bin`，ffmpeg 照样解 |
| ③ 最小 tape | 82 份 / 344 KB | 秒级 |
| ④ 提特征 | 27～230 秒 / 包（CPU） | 9760 包约 5～6 天；有 GPU 几小时 |
| ⑤ exposure | 205 请求 → 300 步，31 万 tokens | 秒级 |
| 训练 | 1.59M 参数，2 步 loss 0.1347→0.1116 | CPU 约 1 分钟 / 步 |
| 推理 | 生成 4.8 秒 / 总 86.6 秒 | 输出的 zip 可直接放进 Phira |

## 八、已知的坑

**1. 20% 的包建不起来**
Phira 里有一部分包，`.json` 扩展名下装的是 PEC 文本（`bp ...` / `cp ...` / `cm ...`），
`info.yml` 的 `format` 是 null。代码会明确拒绝：
`UnsupportedFormat('non-JSON chart (PEC/PBC?): needs a verified converter, not implemented in P0')`。
这批包目前直接跳过，失败清单在 `ingest_report.json` 里。

**2. 特征层号和目录结构**
`batch_extract_features` 产出的是 `features/<sha>/L12.npy`、`L23.npy`、`mel10_complete.npy`。
仓库里历史实验的 panel 读的是 `inputs/r09_positive_pairs/taps/v4_i8_L9-16-24` 和
`inputs/d65_local`（L9/L16/L24）——**两套不是一回事**，别混用。

`scripts/train_phira_tiny.py` 用的是新布局，把 1024 维按 `bands=1` 装进模型的五维契约，
所以模型代码零改动。老的 learning curve 脚本（`run_local_learning_curve.py`）仍然要 MPS + 旧 panel，
在 Linux 上跑不了。

**3. `split` 和 `stratum` 原本不写进 receipt**
`phira_charts_meta.json` 里没有 split 字段，ingest 过去就原样写 `None`，
采样器过滤 `split!='train'` 后一条都挑不出来。现在按音频指纹切 80/20，同一首歌永远同一边。

stratum 同理，`build_phira_exposure` 加了 `--canon` 按 `chart_id` 回查 `difficulty_label` 重算。

**4. 采样器要求双 mode**
`LocalRequestSampler` 过去写死 `phira 3 + mania_4k 1`，且要求两个 mode 都有歌。
纯 Phira 语料下起不来。现在按实际有数据的 mode 组成批次：只有 phira 就整批 phira。

**5. 叉积张量别整曲塞**
`local` 只该取当前 request 的 8 秒窗口。整曲（4000+ 帧 × 1024 维 × 2 层）过一次
`local_projection` 是上万亿次浮点运算，CPU 会卡死。

## 九、目录结构

```
/data
├── phira_charts/        zip 原始文件（只读）
├── phira_db/            canon + targets + receipt
├── phira_audio/         <sha256>.<ext> + audio_index.json
├── features/features/   <sha256>/L12.npy, L23.npy, mel10_complete.npy, receipt.json
├── tape/                *-chart.json, *-targets.npz, *-tokens.npy
├── evolve/              pool.json, exposure-*.json
├── weights/mert/        MERT 权重 4 个文件
└── ckpt/                tiny.pt
```
