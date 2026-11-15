"""Modal-free shared pieces for the full-database pipeline.

These live outside the Modal app modules on purpose: local audits and tests need
the feature-folder contract and the legacy GPU monitor, and importing
``beatfcn.cloud.full_database_app`` would drag in the whole Modal runtime.
"""
from __future__ import annotations
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

GPU_UTILIZATION_REQUIRED = 85.0
GPU_TAKE_INTERVAL_S = 30.0
GPU_ROLLING_INTERVALS = 8
GPU_CHECK_AFTER_S = 480.0
FEATURE_FILES = {'L12.npy', 'L23.npy', 'times_s.npy', 'intervals_s.npy', 'mel10_complete.npy'}


class StableGpuMonitor:
    """Legacy GpuUtil/ResourceGuard semantics with the Owner's 85% threshold."""
    def __init__(self):
        self.sampler = None
        self.stop = threading.Event()
        self.thread = None
        self.lock = threading.Lock()
        self.interval_means = []
        self.sample_counts = []
        self.interval_durations_s = []
        self.error_intervals = 0
        self.stop_reason = None
        self.started_monotonic = None
        self.last_take_monotonic = None
        self.closed_monotonic = None

    def start(self):
        from beatfcn.cloud.gpu_util import GpuUtil
        if self.thread is not None:
            raise RuntimeError('Stable GPU monitor cannot be started twice')
        # Construct and drain exactly at the real pipeline boundary, as the legacy runtime did.
        self.sampler = GpuUtil(period_ms=500)
        self.sampler.take()
        self.started_monotonic = self.last_take_monotonic = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _take_interval(self):
        now = time.monotonic()
        sample = self.sampler.take()
        value = sample.get('util_mean')
        if value is None or not 0 <= value <= 100 or sample.get('samples', 0) <= 0:
            with self.lock:
                self.error_intervals += 1
                self.stop_reason = self.stop_reason or 'gpu_observation_unavailable'
            return False
        with self.lock:
            self.interval_means.append(float(value))
            self.sample_counts.append(int(sample['samples']))
            self.interval_durations_s.append(now - self.last_take_monotonic)
            means = list(self.interval_means)
        self.last_take_monotonic = now
        elapsed = now - self.started_monotonic
        if elapsed >= GPU_CHECK_AFTER_S:
            rolling = means[-GPU_ROLLING_INTERVALS:]
            if sum(rolling) / len(rolling) < GPU_UTILIZATION_REQUIRED:
                self.stop_reason = 'stable_rolling_gpu_mean_below_85'
                return False
        return True

    def _run(self):
        while not self.stop.wait(GPU_TAKE_INTERVAL_S):
            if not self._take_interval():
                return

    def close(self):
        if self.thread is None:
            return
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            with self.lock:
                self.error_intervals += 1
                self.stop_reason = self.stop_reason or 'gpu_monitor_thread_did_not_stop'
        else:
            # Include the final partial decode/GPU/write interval in both means.
            self._take_interval()
        self.sampler.close()
        self.closed_monotonic = time.monotonic()

    def summary(self, *, applicable=True):
        with self.lock:
            means = list(self.interval_means)
            counts = list(self.sample_counts)
            durations = list(self.interval_durations_s)
            errors = self.error_intervals
            reason = self.stop_reason
        mean = lambda values: sum(values) / len(values) if values else None
        endpoint = self.closed_monotonic if self.closed_monotonic is not None else time.monotonic()
        elapsed = None if self.started_monotonic is None else endpoint - self.started_monotonic
        rolling = means[-GPU_ROLLING_INTERVALS:]
        stable_mean = mean(means)
        rolling_mean = mean(rolling)
        result = dict(
            method='legacy_GpuUtil_take_interval_means_with_last8_resource_guard',
            applicable=applicable, required_stable_mean_percent=GPU_UTILIZATION_REQUIRED,
            sampler_period_ms=500, take_interval_s=GPU_TAKE_INTERVAL_S,
            rolling_intervals=GPU_ROLLING_INTERVALS, check_after_s=GPU_CHECK_AFTER_S,
            stable_observed_s=elapsed, interval_count=len(means),
            raw_sample_count=sum(counts), error_interval_count=errors,
            raw_interval_means=means, raw_sample_counts=counts,
            raw_interval_durations_s=durations,
            stable_period_mean_utilization_percent=stable_mean,
            last_eight_interval_mean_utilization_percent=rolling_mean,
            minimum_interval_mean_percent=min(means) if means else None,
            maximum_interval_mean_percent=max(means) if means else None,
            rolling_stop_reason=reason,
            note='Cold model/checkpoint setup and post-pipeline recovery are excluded; sampler is drained at the first real job and the final decode/GPU/write/commit interval is included.')
        result['passed'] = bool(applicable and elapsed is not None
            and elapsed >= GPU_CHECK_AFTER_S and len(means) >= GPU_ROLLING_INTERVALS
            and not errors and all(count > 0 for count in counts)
            and stable_mean is not None and stable_mean >= GPU_UTILIZATION_REQUIRED
            and rolling_mean is not None and rolling_mean >= GPU_UTILIZATION_REQUIRED
            and reason is None)
        if not applicable:
            result['passed'] = False
            result['not_applicable_reason'] = 'No non-existing feature job ran; stable GPU period was never entered.'
        return result


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_feature_folder(folder, job, revision):
    """Remote Volume reopen: exact receipt/files, shapes, finiteness and clocks."""
    folder = Path(folder)
    receipt = json.loads((folder / 'receipt.json').read_text())
    decoded_samples = receipt.get('decoded_samples')
    if type(decoded_samples) is not int or decoded_samples <= 0:
        raise ValueError('Remote feature lacks a positive decoded-sample clock')
    frames = (decoded_samples + 959) // 960
    if (not receipt.get('complete') or receipt.get('audio_sha256') != job['audio_sha256']
            or receipt.get('metadata', {}).get('revision') != revision
            or receipt.get('metadata', {}).get('real_samples') != decoded_samples
            or receipt.get('frames') != frames
            or job['expected_mel_frames'] != 4 * (decoded_samples // 960)
            or set(receipt.get('files', {})) != FEATURE_FILES):
        raise ValueError('Remote feature receipt identity or closure differs: ' + job['audio_sha256'])
    shapes = {}
    dtypes = {}
    arrays = {}
    for name, info in receipt['files'].items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != info['bytes'] or _digest(path) != info['sha256']:
            raise ValueError('Remote feature content differs: ' + job['audio_sha256'] + '/' + name)
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if not np.isfinite(value).all() or not value.flags.c_contiguous:
            raise ValueError('Remote feature is nonfinite or non-contiguous: ' + name)
        arrays[name] = value
        shapes[name] = list(value.shape)
        dtypes[name] = str(value.dtype)
    expected_shapes = {
        'L12.npy': [frames, 1024], 'L23.npy': [frames, 1024],
        'times_s.npy': [frames], 'intervals_s.npy': [frames, 2],
        'mel10_complete.npy': [frames * 4, 128],
    }
    expected_dtypes = {'L12.npy': 'float16', 'L23.npy': 'float16',
        'times_s.npy': 'float64', 'intervals_s.npy': 'float64',
        'mel10_complete.npy': 'float16'}
    if shapes != expected_shapes or dtypes != expected_dtypes:
        raise ValueError('Remote feature shape/clock/dtype differs: ' + job['audio_sha256']
            + ' ' + str(dict(shapes=shapes, dtypes=dtypes)))
    ids = np.arange(frames, dtype=np.int64) * 960
    expected_times = (ids + 360) / 24000
    expected_intervals = np.stack((ids, np.minimum(ids + 960, decoded_samples)), axis=-1) / 24000
    if (not np.array_equal(arrays['times_s.npy'], expected_times)
            or not np.array_equal(arrays['intervals_s.npy'], expected_intervals)):
        raise ValueError('Remote feature physical clock values differ: ' + job['audio_sha256'])
    return dict(audio_sha256=job['audio_sha256'], files=receipt['files'], shapes=shapes,
        dtypes=dtypes, decoded_samples=decoded_samples, complete_tail_frames=frames,
    legacy_truncated_mel_frames=job['expected_mel_frames'],
    remote_reopened=True, finite=True)
