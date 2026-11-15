"""Explicit recovery after zero-update clock bug; each later candidate once."""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/local-portfolio-v1/repaired-sequence-v2.json'


def _wait_or_kill(child, timeout):
    """Wait for a stage child; on wall-clock timeout terminate, then kill, then re-raise."""
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()
        raise


def main():
    if REPORT.exists():raise FileExistsError('Preserve repaired sequence')
    for name in ('final-candidates-queue-v1','lattice-queue-v1','more-candidates-queue-v1'):
        state=json.loads((ROOT/f'reports/local-portfolio-v1/{name}.json').read_text())
        if state['status']!='stopped':raise ValueError('Old queues must be terminal before explicit recovery')
    report=dict(status='running',stages=[],cloud_compute_calls=0);write(REPORT,report);started=time.monotonic()
    try:
        for name,module,args,target,wall in (
            ('grouped-chart-v2','scripts.run_grouped_chart_experiment',['--repair'],'reports/local_grouped_chart_v2.json',7400),
            ('lattice-complete-v1','scripts.run_local_lattice_experiment',[],'reports/local_lattice_complete_v1.json',5100),
            ('event-code-flow-v1','scripts.run_event_code_flow',[],'reports/local_event_code_flow_v1.json',8000),
            ('ssm-event-package-v1','scripts.run_ssm_event_package',[],'reports/local_ssm_event_package_v1.json',7500)):
            report['status']='running_'+name;write(REPORT,report)
            with (ROOT/f'artifacts/local-{name}.log').open('ab') as out:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',module]+args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                report['pid']=child.pid;write(REPORT,report);code=_wait_or_kill(child, wall)
            path=ROOT/target;result=json.loads(path.read_text()) if path.exists() else {}
            report['stages'].append(dict(id=name,exit_code=code,status=result.get('status','no_report'),report=target));write(REPORT,report)
            if code or result.get('status') not in ('completed','health_failed'):raise RuntimeError('Explicit failure review required: '+name)
        report.update(status='completed',wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
