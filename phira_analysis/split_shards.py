"""按 id 排序把 9760 条元数据对半切成两个互斥分片，供两个下载 agent 并行。"""
import json, os

WORK = "/home/user/Doubao/chats/38453554213740802/phira_full"
meta = json.load(open(os.path.join(WORK, "charts_meta.json")))
ids = sorted(int(r["id"]) for r in meta)
n = len(ids)
half = n // 2
shards = {"a": ids[:half], "b": ids[half:]}
out = {}
for tag, lst in shards.items():
    p = os.path.join(WORK, f"shard_{tag}_ids.txt")
    open(p, "w").write("\n".join(str(i) for i in lst))
    out[tag] = {"path": p, "n": len(lst), "min": lst[0], "max": lst[-1],
                "dest": f"/home/user/Doubao/chats/38453554213740802/phira_full/zips/{tag}"}
json.dump({"total": n, "shards": out}, open(os.path.join(WORK, "shards.json"), "w"), ensure_ascii=False, indent=2)
print(json.dumps({"total": n, "shards": out}, ensure_ascii=False, indent=2))
