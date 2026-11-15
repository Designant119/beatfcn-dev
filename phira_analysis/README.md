# Phira 全量数据集分析档案

9760 个谱面 zip（60.60 GiB）的抓取、去重与统计过程留档。**zip 本身不入库**（61 GB），这里只留结论数据和可复现的分析脚本。

## 运行脚本前：设置语料路径

脚本里不再写死任何绝对路径，统一由 `paths.py` 解析。WSL2 按 `SETUP.md` 的约定走 **Linux 侧**（默认 `/data`，别放 `/mnt/e`、`/mnt/d`，NTFS 翻译层读写慢）：

```bash
export BEATFCN_DATA_ROOT=/data                        # 一个变量管全部数据（默认就是 /data）
export PHIRA_WORK=$BEATFCN_DATA_ROOT/phira_full       # 可选：只单独指定 phira 语料根
export PHIRA_OLD=$BEATFCN_DATA_ROOT/charts_by_rating  # 可选：旧分档目录，只有 dl_full.py 用到
```

解析顺序：`PHIRA_WORK` → `BEATFCN_DATA_ROOT/phira_full` → 仓库下 `phira_full`。换盘换机器只改环境变量，脚本一行不用动。

## 结论速览

| 项目 | 数值 |
|---|---|
| 全量谱面 | 9760 个 / 60.60 GiB |
| 唯一音轨 | 9189（按各包主音频）/ 9217（按文件名） |
| 可折叠重复音轨 | 568 条 |
| 去重后可省 | 约 12 GB，占 6% |
| IN/AT 子集 | 7930 个 / 49.09 GiB |
| 过评分门槛（rating≥0.75、票数≥50） | 7300 条 |

难度构成（包体量）：

```
IN      5660 个   34.18 GiB
AT      2270 个   14.91 GiB
SP       265 个    1.69 GiB
HD       179 个    1.01 GiB
EZ        78 个    0.41 GiB
```

## 关键结论

**同名不等于同音轨。** 按歌名合并只能省 6%，远低于按名字统计能到 28.7% 的预期。原因是 Phira 上同名的多是不同录音/不同版本——`Latent Kingdom` 有四种长度且包络相关仅 0.08，`atlasrush` 0.672，`ж` 0.662，`ツムギボシ` 0.490。

**SP 采音密度跟 IN 几乎一样。** SP 全量 264 个包中 244 个可解析，中位 7.26 notes/sec；对照 IN 中位 7.25、AT 中位 9.43。77.5% 落在正常区间，只有 3.7% 是画图级。按标签一刀切会误杀 240 条采音正常的 SP。

**低难度数据不足以当预设。** `EZ Lv.8` 只有 9 条、`HD Lv.8` 8 条、`HD Lv.12` 34 条；而 `IN Lv.15` 1819 条、`AT Lv.16` 699 条。写谱只写高难，低难度不补。

## 文件说明

### 结论数据

| 文件 | 内容 |
|---|---|
| `baseline_zips.json` | 9760 源包基线：{id: {zip, size, n_audio, audio:[[name, crc8, bytes]]}}，验伤用 |
| `audio_alias.json` | 音轨别名表（按内容验证过的合并关系） |
| `audio_alias_stats.json` | 别名统计汇总 |
| `variant_report.json` | 同名变体复核报告（每组时长/音轨差异） |
| `variant_summary.json` | 变体报告摘要 |
| `verify_similarity.json` | 包络相关复核结果（352 组） |
| `mel_similarity.json` | mel 频谱补核（50 组灰区） |
| `mel_targets.json` | mel 复核目标清单 |
| `probe_durations.json` | 音频时长探针（覆盖不全） |
| `audio_match.json` | 音频级匹配结果 |
| `song_groups_dry.json` | 按歌名聚合的干跑结果 |
| `dry_run_report.json` | 去重干跑明细（1.2 MB） |
| `dry_run_新口径报告.md` | 新口径去重干跑报告 |
| `plan/*.csv` | 合并计划：`merge_ok`（可合）/ `merge_review`（待复核）/ `merge_blocked`（禁止合） |
| `shard_a_ids.txt` / `shard_b_ids.txt` | A/B 两片各 4880 的分片清单 |

### 分析脚本（当时跑出上述结论用的）

```
fetch_pages.py          抓 api.phira.cn/chart?page=N 全量分页
build_meta.py           合并分页为 charts_meta.json
split_shards.py         切 A/B 两片
dl_full.py              分片并发下载 + testzip 校验
analyze_variants.py     同名变体复核（时长 + 音轨指纹）
audio_match.py          音频级匹配
build_alias.py          生成别名表
verify_similarity.py    包络相关复核
mel_similarity.py       mel 频谱补核
probe_durations.py      音频时长探针
build_plan.py           生成合并计划 CSV
aggregate_stats.py      统计汇总
verify.py / verify_b.py 分片完整性校验
```

脚本里的路径是按当时的目录结构写死的，在这个仓库里不能直接跑。留在这里是为了记录每个数字是怎么算出来的，不是给人一键复现用的。
