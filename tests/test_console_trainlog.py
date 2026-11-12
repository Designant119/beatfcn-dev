import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import console_server as cs


class FlattenTest(unittest.TestCase):
    def test_nested_dict_recursive_and_value_guards(self):
        flat = cs._flatten(dict(a=1, b=True, c=float('nan'), d=float('inf'),
                                fields=dict(request=dict(loss=0.71), timing=dict(loss=0.60)),
                                deep=dict(a=dict(b=dict(c=dict(d=1)))), lst=[1, 2], s='x'*200))
        self.assertTrue(flat['b'] is True)                      # bool 先于 int 判
        self.assertEqual(flat['c'], 'NaN'); self.assertEqual(flat['d'], 'inf')
        self.assertEqual(flat['fields.request.loss'], 0.71)     # 两层嵌套逐层摊开
        self.assertEqual(flat['fields.timing.loss'], 0.6)
        self.assertEqual(flat['deep.a.b.c'], json.dumps({'d': 1}))   # 超过 4 层压成短 JSON，不摊
        self.assertEqual(flat['lst'], json.dumps([1, 2]))       # list 不摊，压成短 JSON
        self.assertLessEqual(len(flat['s']), 81); self.assertTrue(flat['s'].endswith('…'))


class TrainMetricsTest(unittest.TestCase):
    def test_latest_json_line_wins_and_keeps_every_field(self):
        lines = [
            '{"phase": "build_panel"}',
            '{"phase": "train", "updates": 110, "target": 300, "loss": 1.35, "grad": 1.2, "lr": 0.00095,'
            ' "data_wait": 0.02, "fields": {"timing": 0.62, "request": 0.73}}',
            '{"phase": "train", "updates": 120, "target": 300, "loss": 1.31, "grad": 1.1, "lr": 0.0009,'
            ' "data_wait": 0.01, "fields": {"timing": 0.6, "request": 0.71}, "eval": {"full_note_f1": 0.83}}',
        ]
        m = cs._train_metrics(lines)
        self.assertEqual(m['updates'], 120)
        self.assertEqual(m['lr'], 0.0009)
        self.assertEqual(m['loss'], 1.31)
        self.assertEqual(m['fields.timing'], 0.6)
        self.assertEqual(m['eval.full_note_f1'], 0.83)          # f1 一出现就不落
        self.assertNotEqual(m.get('updates'), 110)

    def test_plain_key_value_progress_line_parses(self):
        self.assertEqual(cs._train_metrics(['noise line', 'epoch=3 lr=1.5e-4 loss=0.42 f1=0.87']),
                         dict(epoch=3, lr=1.5e-4, loss=0.42, f1=0.87))

    def test_lines_without_metrics_yield_empty(self):
        self.assertEqual(cs._train_metrics(['nothing here', 'x: 1', '']), {})


class ValMetricsTest(unittest.TestCase):
    """验证侧指标（val.* 前缀）：F1 在验证事件里，训练进度行永远打不出来。"""

    def _write_reports(self, folder, files):
        reports = os.path.join(folder, 'reports')
        os.makedirs(reports)
        for name, content in files.items():
            with open(os.path.join(reports, name), 'w', encoding='utf-8') as f:
                f.write(content if isinstance(content, str) else json.dumps(content))

    def test_monitor_snapshot_shape_yields_val_fields(self):
        # 落盘形态：单 run 快照，验证记录在 model_progress.latest_validation（snapshot 会 pop validations）
        snap = {'run_id': 'r1', 'last_event_at': 500, 'status': 'training',
                'model_progress': {'latest_validation': {'event': 'val', 'run_id': 'r1',
                    'step': 400, 'ts': 490, 'full_note_f1': 0.831,
                    'onset_f1_50ms': 0.77, 'note_f1_50ms': 0.66}}}
        with tempfile.TemporaryDirectory() as folder:
            self._write_reports(folder, {'training_monitor.json': snap})
            with mock.patch.object(cs, 'REPO', folder):
                m = cs._val_metrics()
        self.assertEqual(m, {'val.step': 400, 'val.ts': 490, 'val.full_note_f1': 0.831,
                             'val.onset_f1_50ms': 0.77, 'val.note_f1_50ms': 0.66})  # event/run_id 字符串不进表

    def test_runs_shape_picks_latest_run_then_max_step(self):
        # 内存/旧形态：runs[run_id].validations[step]；时间最新的 run 里取 step 最大的验证
        data = {'runs': {
            'old': {'last_event_at': 10, 'validations': {'100': {'event': 'val', 'step': 100, 'ts': 9, 'full_note_f1': 0.5}}},
            'new': {'last_event_at': 50, 'validations': {
                '200': {'event': 'val', 'step': 200, 'ts': 40, 'full_note_f1': 0.9},
                '300': {'event': 'val', 'step': 300, 'ts': 45, 'note_f1_50ms': 0.7}}}}}
        with tempfile.TemporaryDirectory() as folder:
            self._write_reports(folder, {'training_monitor.json': data})
            with mock.patch.object(cs, 'REPO', folder):
                m = cs._val_metrics()
        self.assertEqual(m, {'val.step': 300, 'val.ts': 45, 'val.note_f1_50ms': 0.7})  # 不是 old 的 0.5，也不是 new 的 step200

    def test_no_sources_yield_empty_without_fabrication(self):
        with tempfile.TemporaryDirectory() as folder:   # 连 reports 目录都没有
            with mock.patch.object(cs, 'REPO', folder):
                self.assertEqual(cs._val_metrics(), {})
        with tempfile.TemporaryDirectory() as folder:   # 有目录但既无 monitor json 也无验证记录
            self._write_reports(folder, {'events.jsonl': json.dumps({'event': 'train', 'step': 1}) + '\n'})
            with mock.patch.object(cs, 'REPO', folder):
                self.assertEqual(cs._val_metrics(), {})

    def test_jsonl_fallback_takes_last_val_line(self):
        events = '\n'.join([
            json.dumps({'event': 'train', 'step': 240, 'loss': 0.5}),
            json.dumps({'event': 'val', 'step': 200, 'ts': 10, 'full_note_f1': 0.4}),
            json.dumps({'event': 'train', 'step': 250, 'loss': 0.4}),
            json.dumps({'event': 'val', 'step': 250, 'ts': 20, 'full_note_f1': 0.6}),
        ]) + '\n'
        with tempfile.TemporaryDirectory() as folder:
            self._write_reports(folder, {'a.jsonl': events,
                                         'b.jsonl': json.dumps({'event': 'val', 'step': 100, 'ts': 99, 'full_note_f1': 0.1}) + '\n'})
            with mock.patch.object(cs, 'REPO', folder):
                m = cs._val_metrics()
        self.assertEqual(m['val.step'], 250)
        self.assertEqual(m['val.full_note_f1'], 0.6)    # 文件内取最后一条 val；跨文件比 (step, ts)

    def test_monitor_json_beats_jsonl_fallback(self):
        snap = {'run_id': 'r1', 'model_progress': {'latest_validation': {'event': 'val', 'step': 400, 'ts': 500, 'full_note_f1': 0.9}}}
        with tempfile.TemporaryDirectory() as folder:
            self._write_reports(folder, {'training_monitor.json': snap,
                                         'events.jsonl': json.dumps({'event': 'val', 'step': 250, 'ts': 20, 'full_note_f1': 0.6}) + '\n'})
            with mock.patch.object(cs, 'REPO', folder):
                m = cs._val_metrics()
        self.assertEqual(m['val.step'], 400)            # monitor json 里有就不走兜底


class TrainLogTailTest(unittest.TestCase):
    def test_env_log_source_crlf_and_tail_trim(self):
        with tempfile.TemporaryDirectory() as folder:
            log = os.path.join(folder, 'train.log')
            filler = '\n'.join(f'{{"step": {i}, "loss": 1.0}}' for i in range(cs.TRAIN_TAIL_LINES + 60))
            last = '{"step": 9999, "loss": 0.5, "lr": 1e-4}'
            with open(log, 'w', encoding='utf-8', newline='') as f:
                f.write(filler + '\r\n' + last + '\r\n')
            with mock.patch.dict(os.environ, {'BEATFCN_TRAIN_LOG': log}), \
                 mock.patch.object(cs, 'HERE', folder), \
                 mock.patch.object(cs, '_run_all', lambda *a, **k: ''):
                src, lines, age = cs._train_log_tail()
            self.assertEqual(src, '日志 ' + log)
            self.assertEqual(len(lines), cs.TRAIN_TAIL_LINES)
            self.assertEqual(lines[-1], last)                   # CRLF 已剥、最新行在末尾
            self.assertGreaterEqual(age, 0.0)

    def test_stages_json_log_path_is_a_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            log = os.path.join(folder, 'stage.log')
            with open(log, 'w', encoding='utf-8') as f:
                f.write('epoch=1 lr=0.1 loss=9.9\n')
            with open(os.path.join(folder, 'stages.json'), 'w', encoding='utf-8') as f:
                json.dump([dict(id='train', log=log)], f)
            with mock.patch.dict(os.environ, {}), \
                 mock.patch.object(cs, 'HERE', folder), \
                 mock.patch.object(cs, 'REPO', os.path.join(folder, 'no-repo')), \
                 mock.patch.object(cs, '_run_all', lambda *a, **k: ''):
                os.environ.pop('BEATFCN_TRAIN_LOG', None)
                src, lines, _ = cs._train_log_tail()
            self.assertEqual(src, '日志 ' + log)
            self.assertEqual(lines, ['epoch=1 lr=0.1 loss=9.9'])

    def test_no_source_returns_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.dict(os.environ, {}), \
                 mock.patch.object(cs, 'HERE', folder), \
                 mock.patch.object(cs, 'REPO', folder), \
                 mock.patch.object(cs, '_run_all', lambda *a, **k: ''):
                os.environ.pop('BEATFCN_TRAIN_LOG', None)
                self.assertEqual(cs._train_log_tail(), ('', [], None))


class TrainLogEndpointTest(unittest.TestCase):
    def _server(self):
        srv = ThreadingHTTPServer(('127.0.0.1', 0), cs.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv.server_address[1]

    def test_get_trainlog_returns_lines_metrics_and_truncates_long_lines(self):
        long_json = '{"step": 1, "note": "' + 'x'*500 + '"}'
        metric = '{"phase": "train", "updates": 120, "lr": 0.0009, "loss": 1.31, "fields": {"timing": 0.6}}'
        with tempfile.TemporaryDirectory() as folder:
            log = os.path.join(folder, 'train.log')
            with open(log, 'w', encoding='utf-8') as f:
                f.write(long_json + '\n' + metric + '\n')
            with mock.patch.dict(os.environ, {'BEATFCN_TRAIN_LOG': log}), \
                 mock.patch.object(cs, 'HERE', folder), \
                 mock.patch.object(cs, '_run_all', lambda *a, **k: ''):
                port = self._server()
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/trainlog', timeout=5) as r:
                    body = json.loads(r.read())
        self.assertTrue(body['ok'])
        self.assertEqual(body['source'], '日志 ' + log)
        self.assertEqual(len(body['lines']), 2)
        self.assertEqual(len(body['lines'][0]), 401)
        self.assertTrue(body['lines'][0].endswith('…'))
        self.assertEqual(body['metrics']['updates'], 120)
        self.assertEqual(body['metrics']['fields.timing'], 0.6)

    def test_get_trainlog_merges_val_metrics_into_train_metrics(self):
        metric = '{"phase": "train", "updates": 120, "lr": 0.0009, "loss": 1.31}'
        snap = {'run_id': 'r1', 'model_progress': {'latest_validation':
                {'event': 'val', 'step': 400, 'ts': 490, 'full_note_f1': 0.831}}}
        with tempfile.TemporaryDirectory() as folder:
            log = os.path.join(folder, 'train.log')
            with open(log, 'w', encoding='utf-8') as f:
                f.write(metric + '\n')
            reports = os.path.join(folder, 'reports')
            os.makedirs(reports)
            with open(os.path.join(reports, 'training_monitor.json'), 'w', encoding='utf-8') as f:
                json.dump(snap, f)
            with mock.patch.dict(os.environ, {'BEATFCN_TRAIN_LOG': log}), \
                 mock.patch.object(cs, 'HERE', folder), \
                 mock.patch.object(cs, 'REPO', folder), \
                 mock.patch.object(cs, '_run_all', lambda *a, **k: ''):
                port = self._server()
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/trainlog', timeout=5) as r:
                    body = json.loads(r.read())
        self.assertEqual(body['metrics']['updates'], 120)          # 训练行指标照旧
        self.assertEqual(body['metrics']['val.step'], 400)         # 验证侧以 val. 前缀并入
        self.assertEqual(body['metrics']['val.full_note_f1'], 0.831)

    def test_token_required_when_set(self):
        with mock.patch.object(cs, 'TOKEN', 'pwd'), \
             mock.patch.object(cs, '_run_all', lambda *a, **k: ''):
            port = self._server()
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/api/trainlog', timeout=5)
            self.assertEqual(raised.exception.code, 401)
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/trainlog?token=pwd', timeout=5) as r:
                self.assertTrue(json.loads(r.read())['ok'])
