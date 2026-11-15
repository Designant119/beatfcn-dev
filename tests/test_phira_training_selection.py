import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from beatfcn.train.data import load_manifest, selected_for_training


class PhiraTrainingSelection(unittest.TestCase):
    def row(self, **changes):
        return dict(dict(source='phira', difficulty_label='EZ Lv.3', rating=.75,
                    rating_count=50, status='parsed', kept=True, split='train'), **changes)

    def test_rating_and_vote_counts_no_longer_gate_entry(self):
        # 全量口径（2026-11-15 用户决定）：评分与票数只剩元数据意义，不再参与准入。
        # 旧的 075/50 门槛连同它的 fail-closed 校验一起退役；其余条件仍各自把关。
        weird_votes = [50, 51, 100, 10000, '50', None, 0, 49, 49.999, 50.1, True,
                       'invalid', float('nan'), float('inf'), -1]
        weird_ratings = [.75, .749999, .9, 1.0, '0.85', None, '', 'invalid',
                         float('nan'), float('inf'), -1, 1.01, True]
        for votes in weird_votes:
            for rating in weird_ratings:
                with self.subTest(votes=votes, rating=rating):
                    self.assertTrue(selected_for_training(
                        self.row(rating=rating, rating_count=votes)))
        row = self.row(); row.pop('rating'); row.pop('rating_count')
        self.assertTrue(selected_for_training(row))  # 两个字段整片缺失也不再拦

    def test_every_gameplay_tag_is_admitted_under_full_corpus(self):
        # 全量口径（2026-11-15）：SP/仿 mania/ADOFAI/缺标签的谱，采音与常规谱无差别，一律收。
        for label in ['SP Lv.3', '4K Lv.3', 'ADOFAI Lv.3', '', None]:
            row = self.row()
            row['difficulty_label'] = label
            row['rating'] = 1.0
            with self.subTest(label=label):
                self.assertTrue(selected_for_training(row))

    def test_loader_still_requires_technical_and_split_eligibility(self):
        rows = [dict(self.row(), chart_id='accepted'),
                dict(self.row(), chart_id='quarantined', kept=False,
                     quarantine_reasons=['invalid_time']),
                dict(self.row(), chart_id='heldout', split='val')]
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest/charts.jsonl'
            manifest.parent.mkdir()
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual([row['chart_id'] for row in load_manifest(folder, split='train')],
                             ['accepted'])

    def test_mania_approval_rules_are_unchanged(self):
        for approval, accepted in [(1, True), (2, True), (4, False), (0, False)]:
            self.assertEqual(selected_for_training(dict(source='osu', approved=approval)), accepted)

    def test_expansion_entrypoint_accepts_every_chart_with_a_file(self):
        # 全量口径（2026-11-15 用户决定）：不再按评分/票数/reviewed 过滤，只要有谱面文件就收。
        from data_expand.select_ids import main
        rows = [dict(id=1, name='First', level='EZ Lv.3', difficulty=3,
                     reviewed=True, tags=['regular'], file='https://example.org/1',
                     rating=.75, ratingCount=50),
                dict(id=2, name='Second', level='EZ Lv.3', difficulty=3,
                     reviewed=True, tags=['regular'], file='https://example.org/2',
                     rating=.749, ratingCount=10000, stable=True),
                dict(id=3, name='Third', level='EZ Lv.3', difficulty=3,
                     reviewed=True, tags=['regular'], file='https://example.org/3',
                     rating=.99, ratingCount=49, stable=True)]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            catalog, existing, output = (folder / name for name in ['catalog.jsonl', 'old.txt', 'out.txt'])
            catalog.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            existing.write_text('')
            with patch('sys.argv', ['select_ids.py', str(catalog), str(existing), '10', str(output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    main()
            self.assertEqual(sorted(output.read_text().split()), ['1', '2', '3'])


if __name__ == '__main__':
    unittest.main()
