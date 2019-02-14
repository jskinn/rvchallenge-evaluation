from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import os.path
import evaluate_codalab


try:
    import cProfile as profile
except ImportError:
    import profile


class TestIntegration(unittest.TestCase):

    def test_integration(self):
        base_folder = os.path.dirname(__file__)
        input_dir = os.path.join(base_folder, 'integration_dir')
        output_dir = base_folder
        scores_file = os.path.join(base_folder, 'scores.txt')

        evaluate_codalab.main(input_dir, output_dir)

        self.assertTrue(os.path.isfile(scores_file))
        with open(scores_file, 'r') as fp:
            scores_text = fp.read()
        os.remove(scores_file)

        self.assert_regex(scores_text, 'score:[\\d.]+')
        self.assert_regex(scores_text, 'avg_spatial:[\\d.]+')
        self.assert_regex(scores_text, 'avg_label:[\\d.]+')
        self.assert_regex(scores_text, 'avg_pPDQ:[\\d.]+')
        self.assert_regex(scores_text, 'FPs:[\\d]+')
        self.assert_regex(scores_text, 'FNs:[\\d]+')
        self.assert_regex(scores_text, 'TPs:[\\d]+')

        scores_data = read_scores_file(scores_text)
        self.assertIn('score', scores_data)
        self.assertGreaterEqual(scores_data['score'], 0)
        self.assertLessEqual(scores_data['score'], 100)
        self.assertIn('avg_spatial', scores_data)
        self.assertGreaterEqual(scores_data['avg_spatial'], 0)
        self.assertLessEqual(scores_data['avg_spatial'], 1)
        self.assertIn('avg_label', scores_data)
        self.assertGreaterEqual(scores_data['avg_label'], 0)
        self.assertLessEqual(scores_data['avg_label'], 1)
        self.assertIn('avg_pPDQ', scores_data)
        self.assertGreaterEqual(scores_data['avg_pPDQ'], 0)
        self.assertLessEqual(scores_data['avg_pPDQ'], 1)
        self.assertIn('FPs', scores_data)
        self.assertGreaterEqual(scores_data['FPs'], 0)
        self.assertEqual(scores_data['FPs'], int(scores_data['FPs']))
        self.assertIn('FNs', scores_data)
        self.assertGreaterEqual(scores_data['FNs'], 0)
        self.assertEqual(scores_data['FNs'], int(scores_data['FNs']))
        self.assertIn('TPs', scores_data)
        self.assertGreaterEqual(scores_data['TPs'], 0)
        self.assertEqual(scores_data['TPs'], int(scores_data['TPs']))

    def test_profile(self):
        base_folder = os.path.dirname(__file__)
        input_dir = os.path.join(base_folder, 'integration_dir')
        output_dir = base_folder
        scores_file = os.path.join(output_dir, 'scores.txt')
        stats_file = os.path.join(base_folder, 'evaluate.pstat')

        if os.path.isfile(stats_file):
            os.remove(stats_file)

        profile.runctx('evaluate.main(input_dir, output_dir)', filename=stats_file, globals=globals(),
                       locals={'evaluate': evaluate_codalab, 'input_dir': input_dir, 'output_dir': output_dir})

        self.assertTrue(os.path.isfile(scores_file))
        os.remove(scores_file)

    def assert_regex(self, text, regex, msg=None):
        if hasattr(self, 'assertRegex'):
            # noinspection PyCompatibility
            self.assertRegex(text, regex, msg)
        elif hasattr(self, 'assertRegexpMatches'):
            self.assertRegexpMatches(text, regex, msg)


def read_scores_file(scores_text):
    scores_data = {}
    for line in scores_text.split('\n'):
        key, value = line.split(':')
        scores_data[key.strip()] = float(value)
    return scores_data
