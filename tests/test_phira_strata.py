import unittest

from beatfcn.ingest.phira_policy import (SCREEN_VERSION, metadata_identity,
                                       phira_stratum, screen_metadata)
from beatfcn.train.data import selected_for_training


class PhiraStrata(unittest.TestCase):
    def row(self, **changes):
        return dict(dict(source='phira', difficulty_label='IN Lv.13', difficulty=13,
                         rating=.9, rating_count=50, tags=['regular'], description=''), **changes)

    def test_tags_and_level_boundaries_remain_separate(self):
        self.assertEqual([phira_stratum(self.row(difficulty_label=f'IN Lv.{v}', difficulty=v))['key']
                          for v in [10.9, 11, 12.9, 13, 14.9, 15, 16]],
                         ['IN/8-11', 'IN/11-13', 'IN/11-13', 'IN/13-15',
                          'IN/13-15', 'IN/15-16', 'IN/16+'])
        self.assertEqual([phira_stratum(self.row(difficulty_label=f'{tag} Lv.13'))['key']
                          for tag in ['HD', 'IN', 'AT']], ['HD/13-15', 'IN/13-15', 'AT/13-15'])

    def test_conflicts_and_unknowns_fall_back_to_tag_strata(self):
        # Policy: every row can enter sampling; a missing/conflicting band falls back to tag/all.
        cases = [(dict(difficulty=1), 'IN/all'),
                 (dict(difficulty_label='IN', difficulty=0), 'IN/all'),
                 (dict(difficulty_label='SP Lv.13'), 'SP/13-15'),
                 (dict(difficulty_label='', difficulty=0), '?/all')]
        for changes, key in cases:
            self.assertEqual(phira_stratum(self.row(**changes))['key'], key)

    def test_full_corpus_admits_unreviewed_and_dirty_scores(self):
        # 全量口径（2026-11-15）：reviewed 标记、评分票数的脏值都不再充当准入条件。
        self.assertTrue(selected_for_training(self.row(reviewed=False)))
        self.assertTrue(selected_for_training(self.row(rating=True)))

    def test_explicit_performance_or_simulation_is_excluded_from_screen_but_admitted(self):
        # 筛查证据仍照记（excluded 会写进 manifest），但全量口径下不再拦训练准入。
        for tags in [['visual'], ['troll'], ['4K'], ['转谱'], ['表演谱']]:
            row = self.row(tags=tags)
            self.assertEqual(screen_metadata(row)['status'], 'excluded')
            self.assertTrue(selected_for_training(row))

    def test_ambiguous_source_still_flagged_but_not_gated(self):
        for description in ['这是4K模拟玩法', '本人转谱作品', '仅供观赏的演出谱']:
            row = self.row(description=description)
            self.assertEqual(screen_metadata(row)['status'], 'review_required')
            self.assertTrue(selected_for_training(row))

    def test_normal_animation_and_artistic_drags_are_not_auto_rejected(self):
        for description in ['包含演出效果；drag铺满这一段是音乐表现',
                            '不是转谱，非表演谱，原创谱面', '没有4K，正常双指谱']:
            self.assertTrue(selected_for_training(self.row(description=description)))

    def test_declared_primary_performance_and_converter_are_reviewed(self):
        for description in ['一张以note表演为主的谱，望大家喜欢',
                            '感谢起名钉子户的MalodyVersusPhigros转换器',
                            '本谱使用Malody制作，配置由我来写',
                            '依据上方按键点击下侧任意区域']:
            self.assertEqual(screen_metadata(self.row(description=description))['status'],
                             'review_required')

    def test_review_is_bound_to_metadata_and_cannot_erase_nonstandard_tags(self):
        row = self.row(description='曲绘来源：ADOFAI同名关卡')
        review = dict(screen_version=SCREEN_VERSION, metadata_identity=metadata_identity(row),
                      decision='eligible_metadata_only', reason='The reference names the illustration source.')
        row['phira_metadata_review'] = review
        self.assertEqual(screen_metadata(row)['status'], 'eligible_metadata_only')
        row['description'] = 'ADOFAI转谱'
        self.assertEqual(screen_metadata(row)['status'], 'review_required')  # 旧复核失效，不能靠它洗白
        row['tags'] = ['visual']
        review['metadata_identity'] = metadata_identity(row)
        self.assertEqual(screen_metadata(row)['status'], 'excluded')  # 非标准标签无法被 prose 复核清除

    def test_ingestion_preserves_screening_evidence_even_on_parse_failure(self):
        from tempfile import TemporaryDirectory
        from beatfcn.ingest.pipeline import phira_zip_row
        row = dict(self.row(description='以note表演为主'), id=17,
                   level='IN Lv.13', name='Example')
        with TemporaryDirectory() as folder:
            parsed = phira_zip_row(folder, folder + '/missing.zip', row, folder)
        self.assertEqual(parsed['tags'], row['tags'])
        self.assertEqual(parsed['description'], row['description'])
        self.assertEqual(metadata_identity(parsed), metadata_identity(row))
        self.assertEqual(screen_metadata(parsed)['status'], 'review_required')  # 证据照记，准入不再依赖它


if __name__ == '__main__':
    unittest.main()
