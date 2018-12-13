from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import os.path
import evaluate


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

        evaluate.main(input_dir, output_dir)

        self.assertTrue(os.path.isfile(scores_file))
        with open(scores_file, 'r') as fp:
            scores_data = fp.read()
        self.assert_regex(scores_data, 'score:[\\d.]+')
        self.assert_regex(scores_data, 'avg_spatial:[\\d.]+')
        self.assert_regex(scores_data, 'avg_label:[\\d.]+')
        self.assert_regex(scores_data, 'avg_pPDQ:[\\d.]+')
        self.assert_regex(scores_data, 'FPs:[\\d]+')
        self.assert_regex(scores_data, 'FNs:[\\d]+')
        self.assert_regex(scores_data, 'TPs:[\\d]+')
        print("Got a scores file: \n\"{0}\"".format(scores_data))
        os.remove(scores_file)

    def test_profile(self):
        base_folder = os.path.dirname(__file__)
        input_dir = os.path.join(base_folder, 'integration_dir')
        output_dir = base_folder
        scores_file = os.path.join(output_dir, 'scores.txt')
        stats_file = os.path.join(base_folder, 'evaluate.pstat')

        if os.path.isfile(stats_file):
            os.remove(stats_file)

        profile.runctx('evaluate.main(input_dir, output_dir)', filename=stats_file, globals=globals(),
                       locals={'evaluate': evaluate, 'input_dir': input_dir, 'output_dir': output_dir})

        self.assertTrue(os.path.isfile(scores_file))
        os.remove(scores_file)

    def assert_regex(self, text, regex, msg=None):
        if hasattr(self, 'assertRegex'):
            # noinspection PyCompatibility
            self.assertRegex(text, regex, msg)
        elif hasattr(self, 'assertRegexpMatches'):
            self.assertRegexpMatches(text, regex, msg)
