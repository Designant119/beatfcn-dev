#!/bin/bash
# ============================================================================
# 已废弃，不要用。这个脚本指向 /workspace/phira/extract_full.py 和
# bench/offset_clean.py，这两个文件在当前仓库里都不存在，一跑就 exit 1。
# 现在的正确链路是（五步，详见 包说明-先看我.txt）：
#   1 scripts/phira_local_download.py          下谱面 zip
#   2 scripts/ingest_phira_dataset.py          zip -> canon + targets
#   3 scripts/build_feature_tape.py            canon -> 最小 tape
#   4 scripts/batch_extract_features.py        tape -> MERT 特征
#   5 scripts/build_full_native_tape.py        特征 -> 完整 tape
# 保留文件只为留档，防止有人照着旧文档又去跑它。
# ============================================================================
# Extract newly downloaded charts into a staging dir, offset-clean them, then merge into
# grid_npz_full128 / grid_npz_clean without touching existing files. Run from /workspace/bench.
set -e
P=/workspace/phira; PY=/workspace/venv/bin/python
awk -F, '$9=="success"{print $1}' manifest_expand.csv > ids_expand_ok.txt
timeout 10h nice -n 10 $PY $P/extract_full.py charts grid_npz_new grid 3 ids_expand_ok.txt 32
timeout 4h nice -n 10 $PY $P/bench/offset_clean.py grid_npz_new grid_npz_new_clean offset_report_expand.json
$PY - <<'P'
import json,os,shutil
r=json.load(open('offset_report_expand.json'))
drop={k for k,v in r.items() if v.get('drop')}
n=0
for f in os.listdir('grid_npz_new'):
    if not os.path.exists('grid_npz_full128/'+f): shutil.move('grid_npz_new/'+f,'grid_npz_full128/'+f)
for f in os.listdir('grid_npz_new_clean'):
    k=f[:-4]
    if k in drop or os.path.exists('grid_npz_clean/'+f): continue
    shutil.move('grid_npz_new_clean/'+f,'grid_npz_clean/'+f); n+=1
print('new extracted',len(r),'kept',n,'dropped',len(drop))
P
