"""Modal definitions for bounded full-database MERT preparation.

There is intentionally no local entrypoint and no automatic upload/spawn. A
separate Owner-authorized launcher must reserve the lifecycle before importing
this module outside local static tests. GPU efficiency follows the legacy
stable-period monitor: GpuUtil interval means plus the last-eight rolling mean.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import threading
import time

import modal
import numpy as np
from .feature_contract import (GPU_UTILIZATION_REQUIRED, GPU_TAKE_INTERVAL_S,
    GPU_ROLLING_INTERVALS, GPU_CHECK_AFTER_S, FEATURE_FILES, StableGpuMonitor,
    verify_feature_folder)


DATA = '/data'
REMOTE_ROOT = 'full-database-v1'
CHECKPOINT = '/data/full-database-v1/mert2-checkpoint'
CPU_WORKERS = 8
app = modal.App('beatfcn-full-database-mert')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg')
         .pip_install('torch==2.6.0', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('torchaudio==2.6.0', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('transformers==4.53.2', 'numpy==2.2.6', 'safetensors==0.5.3')
         .add_local_python_source('beatfcn'))
_ENCODER = None


def encoder():
    global _ENCODER
    if _ENCODER is None:
        from beatfcn.audio.mert2 import FrozenMert2
        _ENCODER = FrozenMert2(device='cuda', local_source=CHECKPOINT)
    return _ENCODER


def verify_beatfcn_sources(contract):
    """Verify uploaded beatfcn source hashes against the frozen pilot contract."""
    import beatfcn
    package_root = Path(beatfcn.__file__).resolve().parent.parent
    for name, expected in contract['source_sha256'].items():
        if not name.startswith('beatfcn/'):
            continue
        path = package_root / name
        if not path.is_file():
            raise ValueError('Uploaded source missing: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Uploaded source differs from frozen contract: ' + name)


def _prefetch(job, output_root_path, temporary_root):
    from beatfcn.audio.local_mel_cache import complete_song_mel
    from beatfcn.train.full_database import decoded_audio
    audio, original, wave = decoded_audio(job, DATA, temporary_root)
    mel = complete_song_mel(wave)
    return dict(job=job, audio=audio, original=original, wave=wave, mel=mel)


def _write_feature(prefetched, output_root_path, temporary_root, features, expected):
    from beatfcn.train.full_database import commit_feature
    started = time.monotonic()
    result = commit_feature(prefetched['audio'], Path(prefetched['original']), output_root_path,
        temporary_root, features, prefetched['mel'], expected)
    result['wall_s'] = time.monotonic() - started
    return result


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(8, 8),
              memory=(32768, 32768), timeout=12 * 3600, retries=0,
              max_containers=1, scaledown_window=120)
def encode(run_id, contract_json, dispatch_deadline_unix):
    """Pipeline-decode/write around GPU calls; commit and remotely reopen each receipt."""
    from beatfcn.audio.mert2 import validate_local_source
    from beatfcn.train.full_database import feature_payloads
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    validate_local_source(CHECKPOINT)
    output_root = contract['output_root']
    if not output_root.startswith(REMOTE_ROOT + '/'):
        raise ValueError('Output root must stay inside full-database-v1')
    jobs = contract['jobs']
    if not jobs or len({job['audio_sha256'] for job in jobs}) != len(jobs):
        raise ValueError('Jobs must be nonempty and unique by audio hash')
    model = encoder()
    if model.spec.layers != (12, 23):
        raise ValueError('Full database requires frozen MERT layers 12/23')
    output_root_path = Path(DATA) / output_root
    temporary_root = Path('/tmp') / 'full-database-audio'
    temporary_root.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    stopped = False
    monitor = StableGpuMonitor()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=CPU_WORKERS, thread_name_prefix='full-db-cpu') as pool:
        ready = []
        # Existing receipts and their remote recovery checks are cold setup, not stable GPU work.
        for job in jobs:
            folder = output_root_path / 'features' / job['audio_sha256']
            if (folder / 'receipt.json').exists():
                verified = verify_feature_folder(folder, job, model.revision)
                records.append(dict(audio_sha256=job['audio_sha256'], complete=True,
                    wall_s=0., files=verified['files'], existing=True))
            else:
                ready.append(job)
        if not ready:
            utilization = monitor.summary(applicable=False)
            return dict(run_id=run_id, output_root=output_root,
                attempted=len(records), planned=len(jobs), successes=records, failures=[],
                stopped=False, wall_s=time.monotonic() - started,
                recovery_verified=[record['audio_sha256'] for record in records],
                gpu_utilization=utilization, all_existing=True)
        monitor.start()
        new_audio_seconds = 0.
        try:
            prefetch_future = pool.submit(_prefetch, ready[0], output_root_path, temporary_root)
            writer_futures = []
            index = 0
            while index < len(ready):
                if time.time() >= dispatch_deadline_unix or monitor.stop_reason:
                    stopped = True
                    break
                job = ready[index]
                try:
                    prefetched = prefetch_future.result()
                    if index + 1 < len(ready):
                        prefetch_future = pool.submit(_prefetch, ready[index + 1],
                            output_root_path, temporary_root)
                    features, _, expected = feature_payloads(prefetched['wave'], model,
                        mel=prefetched['mel'])
                    writer_futures.append((job, pool.submit(_write_feature, prefetched,
                        output_root_path, temporary_root, features, expected)))
                except Exception as exc:
                    failures.append(dict(audio_sha256=job['audio_sha256'],
                        error=f'{type(exc).__name__}: {exc}'))
                    stopped = True
                    break
                index += 1
                data.commit()
                if index % 5 == 0 or index == len(ready):
                    print(dict(stage='full-database-mert', gpu_submitted=index,
                        successes=len(records) + len(writer_futures), failures=len(failures),
                        wall_s=time.monotonic() - started, gpu_monitor=monitor.summary()), flush=True)
            for job, future in writer_futures:
                try:
                    result = future.result()
                    records.append(dict(audio_sha256=job['audio_sha256'],
                        complete=result['complete'], wall_s=result['wall_s'],
                        files=result['files'], existing=False))
                    new_audio_seconds += float(job['audio_sec'])
                except Exception as exc:
                    failures.append(dict(audio_sha256=job['audio_sha256'],
                        error=f'{type(exc).__name__}: {exc}'))
            data.commit()
        finally:
            monitor.close()
        utilization = monitor.summary()
        # Re-read from the remote Volume after commit; no feature array goes to the Mac launcher.
        data.reload()
        recovery = list(pool.map(lambda record: verify_feature_folder(
            output_root_path / 'features' / record['audio_sha256'],
            next(job for job in jobs if job['audio_sha256'] == record['audio_sha256']),
            model.revision), records))
    stable_s = utilization.get('stable_observed_s') or 0.
    throughput = dict(new_audio_seconds=new_audio_seconds,
        stable_pipeline_seconds=stable_s,
        audio_seconds_per_second=new_audio_seconds / stable_s if stable_s > 0 else None,
        new_jobs_per_second=sum(not record['existing'] for record in records) / stable_s
            if stable_s > 0 else None)
    return dict(run_id=run_id, output_root=output_root,
        attempted=len(records) + len(failures), planned=len(jobs),
        successes=records, failures=failures,
        stopped=stopped or bool(utilization['rolling_stop_reason']),
        recovery_verified=[record['audio_sha256'] for record in recovery],
        throughput=throughput, wall_s=time.monotonic() - started,
        gpu_utilization=utilization, all_existing=False)
