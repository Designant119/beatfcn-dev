"""Extract the counterfactual tape tar on the Volume."""
import json

from beatfcn.cloud import train_stage0_app as training

with training.app.run(detach=True):
    result = training.extract_counterfactual_tar.remote()
    print(json.dumps(result, ensure_ascii=False))
    if isinstance(result, dict) and result.get('returncode'):
        raise SystemExit(result['returncode'])
