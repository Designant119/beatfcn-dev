"""Modal CLI entrypoint for detached 0.6B training."""
import json

import modal

from beatfcn.cloud import train_stage0_app as training

app = training.app


@app.local_entrypoint()
def main(plan: str):
    handle = training.train_large_plan.spawn(plan)
    print(json.dumps(dict(spawned=True, handle=str(handle)), ensure_ascii=False, indent=1))
