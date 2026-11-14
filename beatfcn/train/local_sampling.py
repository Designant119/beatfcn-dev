"""Reproducible mode/song/chart/request sampling for local learning curves."""
from collections import Counter, defaultdict
import hashlib
import json
import random


class LocalRequestSampler:
    def __init__(self, records, *, seed):
        self.records = records
        self.pool_identity = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.rng = random.Random(seed)
        self.queues = {}
        self.pools = {}
        songs, charts, requests = defaultdict(set), defaultdict(set), defaultdict(list)
        for index, row in enumerate(records):
            if row['split'] != 'train':
                continue
            mode, song, chart = row['mode'], row['component'], row['chart_id']
            songs[mode].add(song)
            charts[(mode, song)].add(chart)
            requests[(mode, song, chart)].append(index)
        # 批次配比按实际有数据的 mode 来定：过去写死 phira 3 + mania_4k 1，
        # 没有 mania 语料时这条线根本起不来。现在只有 phira 就整批 phira，
        # 两者都有时仍保持原来的 3:1。
        self.modes = tuple(m for m in ('phira', 'mania_4k') if songs[m])
        if not self.modes:
            raise ValueError('No admitted training mode has any song')
        for mode in self.modes:
            self.pools[('song', mode)] = sorted(songs[mode])
        self.batch_modes = (('phira',) * 4 if self.modes == ('phira',)
                            else ('phira', 'phira', 'phira', 'mania_4k'))
        for (mode, song), values in charts.items():
            self.pools[('chart', mode, song)] = sorted(values)
        for (mode, song, chart), values in requests.items():
            self.pools[('request', mode, song, chart)] = values
        self.exposure = {k: Counter() for k in ('mode', 'song', 'chart', 'stratum', 'request')}

    def _take(self, key):
        if not self.queues.get(key):
            queue = list(self.pools[key])
            self.rng.shuffle(queue)
            self.queues[key] = queue
        return self.queues[key].pop()

    def next_batch(self):
        indices = []
        for mode in self.batch_modes:
            song = self._take(('song', mode))
            chart = self._take(('chart', mode, song))
            index = self._take(('request', mode, song, chart))
            row = self.records[index]
            for field, value in [('mode', mode), ('song', song), ('chart', chart),
                                 ('stratum', mode+'/'+row['stratum']), ('request', str(index))]:
                self.exposure[field][value] += 1
            indices.append(index)
        self.rng.shuffle(indices)
        return indices

    def state_dict(self):
        return dict(pool_identity=self.pool_identity, random_state=self.rng.getstate(), queues=self.queues,
                    exposure={k: dict(v) for k, v in self.exposure.items()})

    def load_state_dict(self, state):
        if state['pool_identity'] != self.pool_identity or not set(state['queues']) <= self.pools.keys():
            raise ValueError('Sampler checkpoint belongs to another request pool')
        for key, values in state['queues'].items():
            if len(set(values)) != len(values) or not set(values) <= set(self.pools[key]):
                raise ValueError('Invalid sampler queue checkpoint')
        self.rng.setstate(state['random_state'])
        self.queues = {k: list(v) for k, v in state['queues'].items()}
        self.exposure = {k: Counter(v) for k, v in state['exposure'].items()}
