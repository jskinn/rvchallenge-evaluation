from __future__ import absolute_import, division, print_function, unicode_literals

import types
import os.path
import shutil
import json
import numpy as np

import tests.test_helpers as th
import data_holders
import submission_loader
import class_list


class TestSubmissionLoaderGenImgPBoxes(th.ExtendedTestCase):
    def test_returns_empty_generator_for_no_detections(self):
        gen = submission_loader.gen_img_pboxes({}, [[0], [0]])
        self.assertIsInstance(gen, types.GeneratorType)
        dets = list(gen)
        self.assertEqual(dets, [])

    def test_builds_detection_instances(self):
        img_dets = [{
            'label_probs': [0.1, 0.1, 0.1, 0.3, 0.4],
            'bbox': [12, 14, 55, 46],
            'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        }, {
            'label_probs': [0.8, 0.05, 0.05, 0.05, 0.05],
            'bbox': [21, 81, 87, 112],
            'covars': [[[15, 5], [5, 21]], [[100, 0], [0, 5]]]
        }, {
            'label_probs': [0.2, 0.1, 0.3, 0.1, 0.3],
            'bbox': [3, 6, 13, 121]
        }]
        dets = list(submission_loader.gen_img_pboxes(img_dets, [list(range(5)), list(range(5))], num_classes=5))
        self.assertEqual(len(img_dets), len(dets))
        for idx in range(len(img_dets)):
            self.assertIsInstance(dets[idx], data_holders.DetectionInstance)
            expected_list = np.zeros(len(class_list.CLASSES), dtype=np.float32)
            expected_list[0:5] = img_dets[idx]['label_probs']
            self.assertNPEqual(expected_list, dets[idx].class_list)

    def test_reorders_label_probs(self):
        det = next(submission_loader.gen_img_pboxes([{
            'label_probs': [0.1, 0.2, 0.3, 0.4],
            'bbox': [12, 14, 55, 46],
            'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        }], [[2, 1, 3], [0, 2, 3]], num_classes=4))
        expected_list = np.zeros(len(class_list.CLASSES), dtype=np.float32)
        expected_list[2] = 0.1
        expected_list[1] = 0.3
        expected_list[3] = 0.4
        self.assertNPEqual(expected_list, det.class_list)

    def test_normalizes_class_list(self):
        det = next(submission_loader.gen_img_pboxes([{
            'label_probs': [0.1, 0.2, 0.3, 0.4, 0.5],
            'bbox': [12, 14, 55, 46],
            'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        }], [list(range(5)), list(range(5))], num_classes=5))
        expected_list = np.zeros(len(class_list.CLASSES), dtype=np.float32)
        expected_list[0:5] = [0.1, 0.2, 0.3, 0.4, 0.5]
        expected_list /= np.sum(expected_list)
        self.assertNPEqual(expected_list, det.class_list)

    def test_doesnt_normalize_underestimate(self):
        det = next(submission_loader.gen_img_pboxes([{
            'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
            'bbox': [12, 14, 55, 46],
            'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        }], [list(range(5)), list(range(5))], num_classes=5))
        expected_list = np.zeros(len(class_list.CLASSES), dtype=np.float32)
        expected_list[0:5] = [0.1, 0.1, 0.1, 0.1, 0.2]
        self.assertNPEqual(expected_list, det.class_list)

    def test_errors_for_missing_label_probs(self):
        with self.assertRaises(KeyError) as cm:
            next(submission_loader.gen_img_pboxes([{
                # 'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def test_errors_for_label_probs_wrong_length(self):
        with self.assertRaises(KeyError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46]
            }], [list(range(5)), list(range(5))], num_classes=6, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def test_errors_for_missing_bbox(self):
        with self.assertRaises(KeyError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2]
                # 'bbox': [12, 14, 55, 46]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def test_errors_for_bbox_wrong_size(self):
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46, 47]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def test_errors_for_bbox_xmax_less_than_xmin(self):
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [55, 14, 12, 46]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def test_errors_for_bbox_ymax_less_than_ymin(self):
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 46, 55, 14]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def test_errors_for_invalid_cov(self):
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[1, 0], [0, 1]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def test_errors_for_assymetric_cov(self):
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [3, 4]], [[1, 0], [0, 1]]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[5, 6], [7, 8]]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def test_errors_for_non_positive_definite_cov(self):
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [2, 1]], [[1, 0], [0, 1]]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)
        with self.assertRaises(ValueError) as cm:
            next(submission_loader.gen_img_pboxes([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 2], [2, 1]]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json'))
        self.assert_contains_indexes(cm, 'test.json', 13, 0)

    def assert_contains_indexes(self, cm, sequence_name, img_num, det_idx):
        msg = str(cm.exception)
        self.assertIn(str(sequence_name), msg)
        self.assertIn(str(img_num), msg)
        self.assertIn(str(det_idx), msg)


class TestSubmissionLoaderReadSequence(th.ExtendedTestCase):
    temp_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'temp')

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def make_sequence(self, detections):
        os.makedirs(self.temp_dir, exist_ok=True)
        json_file = os.path.join(self.temp_dir, 'test.json')
        patch_classes(detections)
        with open(json_file, 'w') as fp:
            json.dump({
                'classes': class_list.CLASSES,
                'detections': detections
            }, fp)
        return json_file

    def test_returns_generator_of_generators(self):
        detections = [
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            }],
            [],
            [],
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            }, {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            }],
        ]
        sequence_json = self.make_sequence(detections)
        gen = submission_loader.read_sequence(sequence_json)
        self.assertIsInstance(gen, types.GeneratorType)
        seq_dets = list(gen)
        self.assertEqual(len(detections), len(seq_dets))
        for img_idx in range(len(seq_dets)):
            self.assertIsInstance(seq_dets[img_idx], types.GeneratorType)
            img_dets = list(seq_dets[img_idx])
            self.assertEqual(len(detections[img_idx]), len(img_dets))

    def test_slice_start_index(self):
        det = {
            'label_probs': [0.1, 0.2, 0.3, 0.4],
            'bbox': [12, 14, 55, 46],
            'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        }
        detections = [
            [det for _ in range(1)],
            [det for _ in range(3)],
            [det for _ in range(5)],
            [det for _ in range(7)],
        ]
        sequence_json = self.make_sequence(detections)
        gen = submission_loader.read_sequence(sequence_json, start_index=2)
        self.assertIsInstance(gen, types.GeneratorType)
        seq_dets = list(gen)
        self.assertEqual(len(detections) - 2, len(seq_dets))
        for img_idx in range(2, len(seq_dets)):
            self.assertIsInstance(seq_dets[img_idx], types.GeneratorType)
            img_dets = list(seq_dets[img_idx])
            self.assertEqual(len(detections[img_idx]), len(img_dets))

    def test_slice_end_index(self):
        det = {
            'label_probs': [0.1, 0.2, 0.3, 0.4],
            'bbox': [12, 14, 55, 46],
            'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        }
        detections = [
            [det for _ in range(1)],
            [det for _ in range(3)],
            [det for _ in range(5)],
            [det for _ in range(7)],
        ]
        sequence_json = self.make_sequence(detections)
        gen = submission_loader.read_sequence(sequence_json, end_index=3)
        self.assertIsInstance(gen, types.GeneratorType)
        seq_dets = list(gen)
        self.assertEqual(3, len(seq_dets))
        for img_idx in range(3):
            self.assertIsInstance(seq_dets[img_idx], types.GeneratorType)
            img_dets = list(seq_dets[img_idx])
            self.assertEqual(len(detections[img_idx]), len(img_dets))

    def test_slice_both(self):
        det = {
            'label_probs': [0.1, 0.2, 0.3, 0.4],
            'bbox': [12, 14, 55, 46],
            'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        }
        detections = [
            [det for _ in range(1)],
            [det for _ in range(3)],
            [det for _ in range(5)],
            [det for _ in range(7)],
        ]
        sequence_json = self.make_sequence(detections)
        gen = submission_loader.read_sequence(sequence_json, start_index=1, end_index=3)
        self.assertIsInstance(gen, types.GeneratorType)
        seq_dets = list(gen)
        self.assertEqual(2, len(seq_dets))
        for img_idx in range(2):
            self.assertIsInstance(seq_dets[img_idx], types.GeneratorType)
            img_dets = list(seq_dets[img_idx])
            self.assertEqual(len(detections[img_idx + 1]), len(img_dets))

    def test_returns_empty_generator_for_no_images(self):
        sequence_json = self.make_sequence([])
        gen = submission_loader.read_sequence(sequence_json)
        self.assertIsInstance(gen, types.GeneratorType)
        dets = list(gen)
        self.assertEqual(dets, [])

    def test_inner_generator_returns_detection_instances(self):
        detections = [
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            }],
            [],
            [],
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            }, {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            }],
        ]
        sequence_json = self.make_sequence(detections)
        gen = submission_loader.read_sequence(sequence_json)
        seq_dets = list(gen)
        self.assertEqual(len(detections), len(seq_dets))
        for img_idx in range(len(seq_dets)):
            img_dets = list(seq_dets[img_idx])
            self.assertEqual(len(detections[img_idx]), len(img_dets))
            for det_idx in range(len(img_dets)):
                self.assertIsInstance(img_dets[det_idx], data_holders.DetectionInstance)
                expected_probs = detections[img_idx][det_idx]['label_probs']
                expected_list = np.zeros(len(class_list.CLASSES), dtype=np.float32)
                expected_list[0:len(expected_probs)] = expected_probs
                self.assertNPEqual(expected_list, img_dets[det_idx].class_list)

    def test_errors_contain_correct_image_numbers(self):
        detections = [
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }],
            [],
            [],
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }, {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [2, 1]], [[1, 0], [0, 1]]]
            }],
        ]
        sequence_json = self.make_sequence(detections)
        seq_gen = submission_loader.read_sequence(sequence_json)

        # the first 3 images are fine
        for _ in range(3):
            next(seq_gen)

        img_dets = next(seq_gen)
        next(img_dets)  # First detection is fine

        with self.assertRaises(ValueError) as cm:
            next(img_dets)  # Second detection, fourth image has a problem
        msg = str(cm.exception)
        self.assertIn(os.path.basename(sequence_json), msg)
        self.assertIn('3', msg)
        self.assertIn('1', msg)

    def test_errors_if_classes_missing(self):
        detections = [
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }],
            [],
            [],
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }, {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [2, 1]], [[1, 0], [0, 1]]]
            }],
        ]
        patch_classes(detections)

        os.makedirs(self.temp_dir, exist_ok=True)
        json_file = os.path.join(self.temp_dir, 'test.json')
        with open(json_file, 'w') as fp:
            json.dump({
                # 'classes': class_list.CLASSES,
                'detections': detections
            }, fp)

        with self.assertRaises(KeyError) as cm:
            next(submission_loader.read_sequence(json_file))
        msg = str(cm.exception)
        self.assertIn('test.json', msg)

    def test_errors_if_detections_missing(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        json_file = os.path.join(self.temp_dir, 'test.json')
        with open(json_file, 'w') as fp:
            json.dump({
                'classes': class_list.CLASSES,
                # 'detections': detections
            }, fp)

        with self.assertRaises(KeyError) as cm:
            next(submission_loader.read_sequence(json_file))
        msg = str(cm.exception)
        self.assertIn('test.json', msg)

    def test_errors_if_no_valid_classes(self):
        detections = [
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }],
            [],
            [],
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }, {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [2, 1]], [[1, 0], [0, 1]]]
            }],
        ]
        patch_classes(detections)

        os.makedirs(self.temp_dir, exist_ok=True)
        json_file = os.path.join(self.temp_dir, 'test.json')
        with open(json_file, 'w') as fp:
            json.dump({
                'classes': [str(idx) for idx in range(len(class_list.CLASSES))],
                'detections': detections
            }, fp)

        with self.assertRaises(ValueError) as cm:
            next(submission_loader.read_sequence(json_file))
        msg = str(cm.exception)
        self.assertIn('test.json', msg)


class TestSubmissionLoaderReadSubmission(th.ExtendedTestCase):
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def make_submission(self, detections_map, subfolder=None):
        root = self.temp_dir
        if subfolder is not None:
            root = os.path.join(self.temp_dir, subfolder)
        os.makedirs(root, exist_ok=True)
        for sequence_name, detections in detections_map.items():
            json_file = os.path.join(root, '{0}.json'.format(sequence_name))
            patch_classes(detections)
            with open(json_file, 'w') as fp:
                json.dump({
                    'classes': class_list.CLASSES,
                    'detections': detections
                }, fp)

    def test_returns_map_of_name_to_generator(self):
        self.make_submission({'000000': [], '000001': []})
        sequences = submission_loader.read_submission(self.temp_dir, {'000000', '000001'})
        self.assertIn('000000', sequences)
        self.assertIn('000001', sequences)
        self.assertIsInstance(sequences['000000'], types.GeneratorType)
        self.assertIsInstance(sequences['000001'], types.GeneratorType)

    def test_finds_only_requested_sequences(self):
        self.make_submission({'000000': [], '000001': []})
        sequences = submission_loader.read_submission(self.temp_dir, {'000000'})
        self.assertEqual({'000000'}, set(sequences.keys()))
        self.assertIsInstance(sequences['000000'], types.GeneratorType)

    def test_searches_for_sequences_in_subfolders(self):
        self.make_submission({'000000': [], '000001': []}, 'folder_a')
        self.make_submission({'000002': [], '000003': []}, 'folder_b')
        sequence_names = {'000000', '000001', '000002', '000003'}
        sequences = submission_loader.read_submission(self.temp_dir, sequence_names)
        self.assertEqual(sequence_names, set(sequences.keys()))
        self.assertIsInstance(sequences['000000'], types.GeneratorType)
        self.assertIsInstance(sequences['000001'], types.GeneratorType)
        self.assertIsInstance(sequences['000002'], types.GeneratorType)
        self.assertIsInstance(sequences['000003'], types.GeneratorType)

    def test_raises_error_if_duplicate_sequence(self):
        self.make_submission({'000000': [], '000001': []}, 'folder_a')
        self.make_submission({'000000': [], '000002': []}, 'folder_b')
        sequence_names = {'000000', '000001', '000002'}

        with self.assertRaises(ValueError) as cm:
            submission_loader.read_submission(self.temp_dir, sequence_names)
        msg = str(cm.exception)
        self.assertIn('folder_a/000000.json', msg)
        self.assertIn('folder_b/000000.json', msg)

    def test_generators_read_sequences(self):
        submission = {
            '000000': [
                [{
                    'label_probs': [0.1, 0.2, 0.3, 0.4],
                    'bbox': [12, 14, 55, 46],
                    'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
                }],
                [],
                [],
                [{
                    'label_probs': [0.1, 0.2, 0.3, 0.4],
                    'bbox': [12, 14, 55, 46],
                    'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
                }, {
                    'label_probs': [0.1, 0.2, 0.3, 0.4],
                    'bbox': [12, 14, 55, 46],
                    'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
                }],
            ],
            '000001': [
                [{
                    'label_probs': [0.1, 0.2, 0.3, 0.4],
                    'bbox': [12, 14, 55, 46],
                    'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
                }],
                [],
                [],
                [{
                    'label_probs': [0.1, 0.2, 0.3, 0.4],
                    'bbox': [12, 14, 55, 46],
                    'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
                }, {
                    'label_probs': [0.1, 0.2, 0.3, 0.4],
                    'bbox': [12, 14, 55, 46],
                    'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
                }],
            ]
        }
        self.make_submission(submission)
        sequences = submission_loader.read_submission(self.temp_dir, {'000000', '000001'})
        for sequence_name, detections in submission.items():
            self.assertIn(sequence_name, sequences)
            seq_dets = list(sequences[sequence_name])
            self.assertEqual(len(detections), len(seq_dets))
            for img_idx in range(len(seq_dets)):
                img_dets = list(seq_dets[img_idx])
                self.assertEqual(len(detections[img_idx]), len(img_dets))
                for det_idx in range(len(img_dets)):
                    self.assertIsInstance(img_dets[det_idx], data_holders.DetectionInstance)
                    expected_probs = detections[img_idx][det_idx]['label_probs']
                    expected_list = np.zeros(len(class_list.CLASSES), dtype=np.float32)
                    expected_list[0:len(expected_probs)] = expected_probs
                    self.assertNPEqual(expected_list, img_dets[det_idx].class_list)
                    self.assertNPEqual(img_dets[det_idx].box, detections[img_idx][det_idx]['bbox'])

    def test_generators_slice_sequences(self):
        def make_det(idx):
            return {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55 + idx, 46 + idx],
                'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            }
        submission = {
            '000000': [
                [make_det(idx) for idx in range(3)],
                [make_det(idx) for idx in range(5)],
                [make_det(idx) for idx in range(7)],
                [make_det(idx) for idx in range(9)]
            ],
            '000001': [
                [make_det(idx) for idx in range(2)],
                [make_det(idx) for idx in range(4)],
                [make_det(idx) for idx in range(6)]
            ]
        }
        self.make_submission(submission)
        sequences = submission_loader.read_submission(self.temp_dir, {'000000', '000001'}, start_index=1, end_index=5)
        for sequence_name in submission.keys():
            self.assertIn(sequence_name, sequences)
            seq_dets = list(sequences[sequence_name])
            detections = submission[sequence_name][1:5]
            self.assertEqual(len(detections), len(seq_dets))
            for img_idx in range(len(seq_dets)):
                img_dets = list(seq_dets[img_idx])
                self.assertEqual(len(detections[img_idx]), len(img_dets))
                for det_idx in range(len(img_dets)):
                    self.assertIsInstance(img_dets[det_idx], data_holders.DetectionInstance)
                    self.assertNPEqual(img_dets[det_idx].box, detections[img_idx][det_idx]['bbox'])


def patch_classes(detections):
    # Patch the label probabilities to be the right length
    for img_dets in detections:
        for det in img_dets:
            if len(det['label_probs']) < len(class_list.CLASSES):
                det['label_probs'] = make_probs(det['label_probs'])


def make_probs(probs):
    full_probs = [0.0] * len(class_list.CLASSES)
    full_probs[0:len(probs)] = probs
    return full_probs
