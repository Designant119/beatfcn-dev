"""生成“322M 从0 + 生产配方”的冻结 plan/exposure，再交给 run_local_song_baseline。

这里不自己改 loss/EMA/数据契约：直接复用生产入口
`prepare_native_counterfactual_training`（counterfactual exposure、own-target 条件、
teacher-forced/EMA/checkpoint、validation panel），只在外面套一层 recipe，方便网页
把学习率、warmup、accum、microbatch、EMA、验证集和 checkpoint 间隔传进来。

  python -m scripts.build_l322m_production --recipe plans/evolve/l322m-production.json

recipe 里必须给 base_plan（通常是 /data/round5-level/plan.json）和 tape/mert_features。
base_plan 只当结构模板；生成时会删掉 initialize_from / resume_checkpoint / init_weights，
保证真的从0初始化。
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_native_counterfactual_training as prep


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--plan-output', default=None)
    ap.add_argument('--exposure', default=None)
    args = ap.parse_args()
    recipe_path = Path(args.recipe)
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    base_path = Path(recipe['base_plan'])
    if not base_path.is_file():
        raise SystemExit('生产配方需要 base_plan（结构模板）：%s' % base_path)
    base = json.loads(base_path.read_text(encoding='utf-8'))
    tape = recipe.get('tape') or base.get('tape')
    mert = recipe.get('mert_features') or base.get('mert_features')
    norm = base.get('mel_normalization')
    base_mel = norm.get('path') if isinstance(norm, dict) else None
    mel = recipe.get('mel_stats') or base_mel
    if not tape or not mert or not mel:
        raise SystemExit('recipe/base_plan 缺少 tape / mert_features / mel_stats')
    if not Path(tape).exists():
        raise SystemExit('tape 不存在：%s' % tape)
    if not Path(mert).is_dir():
        raise SystemExit('MERT 特征目录不存在：%s' % mert)
    if not Path(mel).is_file():
        raise SystemExit('mel statistics 不存在：%s' % mel)

    plan_output = Path(args.plan_output or recipe['plan_output'])
    exposure = Path(args.exposure or recipe['exposure'])
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    exposure.parent.mkdir(parents=True, exist_ok=True)
    scale = Path(tempfile.mkdtemp(prefix='l322m-scale-')) / 'scale.json'
    scale.write_text(json.dumps({'config': recipe['config']}, ensure_ascii=False), encoding='utf-8')

    argv = [
        '--tape', str(tape),
        '--mert-features', str(mert),
        '--mel-stats', str(mel),
        '--scale-config', str(scale),
        '--name', str(recipe.get('name') or 'l322m-production'),
        '--optimizer-updates', str(int(recipe.get('optimizer_updates', 2400))),
        '--accum-steps', str(int(recipe.get('accum_steps', 4))),
        '--physical-microbatch', str(int(recipe.get('physical_microbatch', 1))),
        '--learning-rate', str(float(recipe.get('learning_rate', 1e-4))),
        '--warmup-updates', str(int(recipe.get('warmup_updates', 100))),
        '--minimum-lr', str(float(recipe.get('minimum_lr', 1e-5))),
        '--ema-decay', str(float(recipe.get('ema_decay', .999))),
        '--validation-count', str(int(recipe.get('validation_count', 200))),
        '--checkpoint-every-updates', str(int(recipe.get('checkpoint_every_updates', 500))),
        '--light-checkpoint-every-updates', str(int(recipe.get('light_checkpoint_every_updates', 250))),
        '--max-train-wall-s', str(int(recipe.get('max_train_wall_s', 43200))),
        '--seed', str(int(recipe.get('seed', 20261115))),
        '--benchmark-every-updates', str(int(recipe.get('benchmark_every_updates', 250))),
        '--benchmark-max-wall-s', str(float(recipe.get('benchmark_max_wall_s', 600))),
        '--efficiency-profile', 'none',
        '--base-plan', str(base_path),
        '--exposure', str(exposure),
        '--plan-output', str(plan_output),
    ]
    if recipe.get('treatment_mix') is not None:
        argv += ['--treatment-mix', json.dumps(recipe['treatment_mix'], ensure_ascii=False)]
    if recipe.get('condition_dropout') is not None:
        argv += ['--condition-dropout', json.dumps(recipe['condition_dropout'], ensure_ascii=False)]
    if recipe.get('health'):
        argv += ['--health', str(recipe['health'])]
    if recipe.get('features'):
        argv += ['--features', str(recipe['features'])]
    prep.main(argv)

    plan = json.loads(plan_output.read_text(encoding='utf-8'))
    # 从0：base_plan 只贡献结构/字段，不允许继承任何旧权重或续跑状态。
    for key in ('initialize_from', 'resume_checkpoint', 'init_weights', 'round4', 'relaunch',
                'context_training_source_plan', 'memory_probe', 'cuda_graph', 'compile_loss',
                'grouped_max_tokens', 'efficiency_profile', 'yield_control_file'):
        plan.pop(key, None)
    plan['config']['grad_checkpointing'] = True
    plan['device'] = 'cuda'
    plan['loader_workers'] = int(recipe.get('loader_workers', 4))
    plan['loader_prefetch'] = int(recipe.get('loader_prefetch', 2))
    plan['multiprocess_loader'] = True
    plan['checkpoint_keep'] = int(recipe.get('checkpoint_keep', 1))
    plan['output'] = str(recipe.get('output') or f'/data/evolve/{plan["id"]}-training.json')
    plan['checkpoint_dir'] = str(recipe.get('checkpoint_dir') or f'/data/evolve/{plan["id"]}-checkpoints')
    plan['generation_output'] = str(recipe.get('generation_output') or f'/data/evolve/{plan["id"]}-generation.json')
    plan['generation_dir'] = str(recipe.get('generation_dir') or f'/data/evolve/{plan["id"]}-generation')
    plan['note'] = '322M from-scratch production recipe: native counterfactual exposure, own-target conditions, EMA, teacher validation, production runner.'
    plan['production_recipe'] = str(recipe_path)
    plan['recipe_sha256'] = digest(recipe_path)
    plan.setdefault('code_sha256', {})[str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    plan_output.write_text(json.dumps(plan, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(plan=str(plan_output), exposure=str(exposure),
                          optimizer_updates=plan['optimizer_updates'], steps=plan['steps'],
                          records=plan.get('validation_selection'),
                          config=plan['config']), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
