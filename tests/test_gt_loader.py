from __future__ import absolute_import, division, print_function, unicode_literals

import shutil
import unittest
import warnings
import types
import os.path
import json
import numpy as np
import cv2

import data_holders
import gt_loader
import class_list


class BaseTestGroundTruthLoader(unittest.TestCase):
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp'))

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def make_sequence(self, sequence_data, root_dir=None):
        if root_dir is None:
            root_dir = self.temp_dir
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # This is lifted out of sequence_processing.py, is how the ground truth is written
        labels = {}
        mask_idx = 0
        for image_id in sorted(sequence_data.keys()):
            image_name = '{0:06}'.format(image_id)
            mask_name = "{0}.masks.png".format(mask_idx)
            img_labels = {
                '_metadata': {
                    'mask_name': mask_name
                }
            }
            if len(sequence_data[image_id]) > 255:
                im_mask = np.zeros((100, 100), dtype=np.uint16)
            else:
                im_mask = np.zeros((100, 100), dtype=np.uint8)
            detection_mask_id = 1
            for instance_id in sorted(sequence_data[image_id].keys()):
                instance_name = '{0:06}'.format(instance_id)
                x1, y1, x2, y2 = sequence_data[image_id][instance_id]['bbox']
                img_labels[instance_name] = {
                    'class': class_list.CLASSES[sequence_data[image_id][instance_id]['class']],
                    'mask_id': detection_mask_id
                }
                add_mask(im_mask, x1, y1, x2, y2, detection_mask_id)
                detection_mask_id += 1

            # This mask image is full, save it and move on to the next one
            cv2.imwrite(os.path.join(root_dir, mask_name), im_mask)
            mask_idx += 1
            labels[image_name] = img_labels

        # Save the combined labels and masks to file
        with open(os.path.join(root_dir, 'labels.json'), 'w') as fp:
            json.dump(labels, fp)


class TestGroundTruthLoaderReadGTForImage(unittest.TestCase):

    def test_reads_single_instance(self):
        random = np.random.RandomState(13)
        image_id = 12172
        detection_id = '000455'
        class_id = 3 % len(class_list.CLASSES)
        mask_id = 4
        detection_data = {
            detection_id: {
                'class': class_list.CLASSES[class_id],
                'mask_id': mask_id
            }
        }

        masks = np.zeros((100, 100), dtype=np.uint8)
        for i in range(0, 4):
            add_random_mask(masks, i, random)
        for i in range(5, 16):
            add_random_mask(masks, i, random)
        add_mask(masks, 12, 32, 46, 74, mask_id)

        result = gt_loader.read_gt_for_image(
            image_id=image_id,
            image_data=detection_data,
            masks=masks
        )
        self.assertIsInstance(result, types.GeneratorType)
        result = list(result)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], data_holders.GroundTruthInstance)
        self.assertEqual(result[0].image_id, image_id)
        self.assertEqual(result[0].class_label, class_id)
        self.assertEqual(result[0].bounding_box, [12, 32, 46, 74])
        self.assertTrue(np.array_equal(result[0].segmentation_mask, make_mask(12, 32, 46, 74)))


class TestGroundTruthLoaderReadSequence(BaseTestGroundTruthLoader):

    def setUp(self):
        self.sequence_data = {
            17: {
                15: {'class': 1, 'bbox': (15, 22, 87, 98)},
                23: {'class': 4, 'bbox': (10, 12, 32, 35)},
            },
            32: {
                8: {'class': 3, 'bbox': (68, 91, 73, 99)},
                11: {'class': 2, 'bbox': (35, 32, 69, 71)},
                22: {'class': 5, 'bbox': (6, 12, 37, 34)}
            },
            453: {
                8: {'class': 4, 'bbox': (0, 3, 97, 95)}
            }
        }
        self.make_sequence(self.sequence_data)

    def test_reads_sequence_data(self):
        image_ids = sorted(self.sequence_data.keys())
        images_loaded = 0

        result = gt_loader.read_sequence(self.temp_dir)
        self.assertIsInstance(result, types.GeneratorType)
        for idx, image_gt in enumerate(result):
            image_id = image_ids[idx]
            images_loaded += 1
            det_ids = sorted(self.sequence_data[image_id])

            self.assertIsInstance(image_gt, types.GeneratorType)
            image_gt = list(image_gt)
            self.assertEqual(len(image_gt), len(self.sequence_data[image_id]))

            for det_idx, gt_obj in enumerate(image_gt):
                det_id = det_ids[det_idx]

                self.assertIsInstance(gt_obj, data_holders.GroundTruthInstance)
                self.assertEqual(gt_obj.image_id, image_id)
                self.assertEqual(gt_obj.class_label, self.sequence_data[image_id][det_id]['class'])
                self.assertEqual(gt_obj.bounding_box, list(self.sequence_data[image_id][det_id]['bbox']))
        self.assertEqual(images_loaded, len(self.sequence_data))

    def test_slice_start(self):
        image_ids = sorted(idx for idx in self.sequence_data.keys() if idx >= 32)
        images_loaded = 0

        result = gt_loader.read_sequence(self.temp_dir, start_index=32)
        self.assertIsInstance(result, types.GeneratorType)
        for idx, image_gt in enumerate(result):
            image_id = image_ids[idx]
            images_loaded += 1
            det_ids = sorted(self.sequence_data[image_id].keys())

            self.assertIsInstance(image_gt, types.GeneratorType)
            image_gt = list(image_gt)
            self.assertEqual(len(image_gt), len(self.sequence_data[image_id]))

            for det_idx, gt_obj in enumerate(image_gt):
                det_id = det_ids[det_idx]

                self.assertIsInstance(gt_obj, data_holders.GroundTruthInstance)
                self.assertEqual(gt_obj.image_id, image_id)
                self.assertEqual(gt_obj.class_label, self.sequence_data[image_id][det_id]['class'])
                self.assertEqual(gt_obj.bounding_box, list(self.sequence_data[image_id][det_id]['bbox']))
        self.assertEqual(images_loaded, len(self.sequence_data) - 1)

    def test_read_sequence_slice_end(self):
        image_ids = sorted(idx for idx in self.sequence_data.keys() if idx < 453)
        images_loaded = 0

        result = gt_loader.read_sequence(self.temp_dir, end_index=453)
        self.assertIsInstance(result, types.GeneratorType)
        for idx, image_gt in enumerate(result):
            image_id = image_ids[idx]
            images_loaded += 1
            det_ids = sorted(self.sequence_data[image_id])

            self.assertIsInstance(image_gt, types.GeneratorType)
            image_gt = list(image_gt)
            self.assertEqual(len(image_gt), len(self.sequence_data[image_id]))

            for det_idx, gt_obj in enumerate(image_gt):
                det_id = det_ids[det_idx]

                self.assertIsInstance(gt_obj, data_holders.GroundTruthInstance)
                self.assertEqual(gt_obj.image_id, image_id)
                self.assertEqual(gt_obj.class_label, self.sequence_data[image_id][det_id]['class'])
                self.assertEqual(gt_obj.bounding_box, list(self.sequence_data[image_id][det_id]['bbox']))
        self.assertEqual(images_loaded, len(self.sequence_data) - 1)


class TestGroundTruthLoaderReadSequenceLarge(BaseTestGroundTruthLoader):

    def test_read_lots_of_objects(self):
        sequence_data = {
            22: {
                3 * (x + 20 * y) + 7: {
                    'class': 1 + (13 * (x + 20 * y)) % (len(class_list.CLASSES) - 1),
                    'bbox': (
                        5 * x,
                        5 * y,
                        5 * (x + 1) - 1,
                        5 * (y + 1) - 1
                    )
                }
                for x in range(20)
                for y in range(20)
            }
        }
        self.make_sequence(sequence_data)

        image_ids = sorted(sequence_data.keys())
        images_loaded = 0

        result = gt_loader.read_sequence(self.temp_dir)
        self.assertIsInstance(result, types.GeneratorType)
        for idx, image_gt in enumerate(result):
            image_id = image_ids[idx]
            images_loaded += 1
            det_ids = sorted(sequence_data[image_id].keys())

            self.assertIsInstance(image_gt, types.GeneratorType)
            image_gt = list(image_gt)
            self.assertEqual(len(image_gt), len(sequence_data[image_id]))

            for det_idx, gt_obj in enumerate(image_gt):
                det_id = det_ids[det_idx]

                self.assertIsInstance(gt_obj, data_holders.GroundTruthInstance)
                self.assertEqual(gt_obj.image_id, image_id)
                self.assertEqual(gt_obj.class_label, sequence_data[image_id][det_id]['class'])
                self.assertEqual(gt_obj.bounding_box, list(sequence_data[image_id][det_id]['bbox']))
        self.assertEqual(images_loaded, len(sequence_data))


class TestGroundTruthLoaderReadGroundTruth(BaseTestGroundTruthLoader):

    def setUp(self):
        self.all_data = {
            '000000': {
                17: {
                    15: {'class': 1, 'bbox': (15, 22, 87, 98)},
                    23: {'class': 4, 'bbox': (10, 12, 32, 35)},
                },
                32: {
                    8: {'class': 3, 'bbox': (68, 91, 73, 99)},
                    11: {'class': 2, 'bbox': (35, 32, 69, 71)},
                    22: {'class': 5, 'bbox': (6, 12, 37, 34)}
                },
                453: {
                    8: {'class': 4, 'bbox': (0, 3, 97, 95)}
                }
            },
            '000006': {
                3: {
                    15: {'class': 1, 'bbox': (15, 22, 69, 71)},
                    11: {'class': 2, 'bbox': (35, 32, 87, 98)},
                    8: {'class': 4, 'bbox': (0, 3, 97, 95)}
                },
                5: {
                    8: {'class': 3, 'bbox': (68, 91, 73, 99)},
                    22: {'class': 5, 'bbox': (6, 12, 37, 34)}
                },
                48: {
                    23: {'class': 4, 'bbox': (10, 12, 32, 35)}
                }
            },
            '000032': {
                41: {
                    8: {'class': 4, 'bbox': (0, 3, 97, 95)}
                },
                46: {
                    8: {'class': 3, 'bbox': (68, 91, 73, 99)},
                    11: {'class': 2, 'bbox': (35, 32, 69, 71)},
                    22: {'class': 5, 'bbox': (6, 12, 37, 34)},
                    23: {'class': 4, 'bbox': (10, 13, 32, 35)}
                },
                358: {
                    15: {'class': 1, 'bbox': (15, 2, 33, 98)}
                }
            },
        }
        for sequence_name, sequence_data in self.all_data.items():
            self.make_sequence(sequence_data, os.path.join(self.temp_dir, sequence_name))

    def test_read_ground_truth(self):
        ground_truth = gt_loader.read_ground_truth(self.temp_dir)
        self.assertEqual(set(self.all_data.keys()), set(ground_truth.keys()))

        for sequence_name, sequence_data in self.all_data.items():
            image_ids = sorted(sequence_data.keys())
            images_loaded = 0

            result = ground_truth[sequence_name]
            self.assertIsInstance(result, types.GeneratorType)
            for idx, image_gt in enumerate(result):
                image_id = image_ids[idx]
                images_loaded += 1
                det_ids = sorted(sequence_data[image_id])

                self.assertIsInstance(image_gt, types.GeneratorType)
                image_gt = list(image_gt)
                self.assertEqual(len(image_gt), len(sequence_data[image_id]))

                for det_idx, gt_obj in enumerate(image_gt):
                    det_id = det_ids[det_idx]

                    self.assertIsInstance(gt_obj, data_holders.GroundTruthInstance)
                    self.assertEqual(gt_obj.image_id, image_id)
                    self.assertEqual(gt_obj.class_label, sequence_data[image_id][det_id]['class'])
                    self.assertEqual(gt_obj.bounding_box, list(sequence_data[image_id][det_id]['bbox']))
            self.assertEqual(images_loaded, len(sequence_data))

    def test_omits_sequences(self):
        included_sequences = {sequence_name for sequence_name in self.all_data.keys() if int(sequence_name) > 5}
        ground_truth = gt_loader.read_ground_truth(self.temp_dir,
                                                   valid_sequences={int(idx) for idx in included_sequences})
        self.assertEqual(included_sequences, set(ground_truth.keys()))

        for sequence_name in included_sequences:
            sequence_data = self.all_data[sequence_name]
            image_ids = sorted(sequence_data.keys())
            images_loaded = 0

            result = ground_truth[sequence_name]
            self.assertIsInstance(result, types.GeneratorType)
            for idx, image_gt in enumerate(result):
                image_id = image_ids[idx]
                images_loaded += 1
                det_ids = sorted(sequence_data[image_id])

                self.assertIsInstance(image_gt, types.GeneratorType)
                image_gt = list(image_gt)
                self.assertEqual(len(image_gt), len(sequence_data[image_id]))

                for det_idx, gt_obj in enumerate(image_gt):
                    det_id = det_ids[det_idx]

                    self.assertIsInstance(gt_obj, data_holders.GroundTruthInstance)
                    self.assertEqual(gt_obj.image_id, image_id)
                    self.assertEqual(gt_obj.class_label, sequence_data[image_id][det_id]['class'])
                    self.assertEqual(gt_obj.bounding_box, list(sequence_data[image_id][det_id]['bbox']))
            self.assertEqual(images_loaded, len(sequence_data))

    def test_slices_sequences(self):
        ground_truth = gt_loader.read_ground_truth(self.temp_dir, start_index=10, end_index=20)
        self.assertEqual(set(self.all_data.keys()), set(ground_truth.keys()))

        for sequence_name, sequence_data in self.all_data.items():
            image_ids = sorted(idx for idx in sequence_data.keys() if 10 < idx < 20)
            images_loaded = 0

            result = ground_truth[sequence_name]
            self.assertIsInstance(result, types.GeneratorType)
            for idx, image_gt in enumerate(result):
                image_id = image_ids[idx]
                images_loaded += 1
                det_ids = sorted(sequence_data[image_id])

                self.assertIsInstance(image_gt, types.GeneratorType)
                image_gt = list(image_gt)
                self.assertEqual(len(image_gt), len(sequence_data[image_id]))

                for det_idx, gt_obj in enumerate(image_gt):
                    det_id = det_ids[det_idx]

                    self.assertIsInstance(gt_obj, data_holders.GroundTruthInstance)
                    self.assertEqual(gt_obj.image_id, image_id)
                    self.assertEqual(gt_obj.class_label, sequence_data[image_id][det_id]['class'])
                    self.assertEqual(gt_obj.bounding_box, list(sequence_data[image_id][det_id]['bbox']))
            self.assertEqual(images_loaded, len(image_ids))


def make_mask(x1, y1, x2, y2):
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    mask = np.zeros((100, 100), dtype=np.bool)
    mask[max(y1, 0):min(y2+1, 100), max(x1, 0):min(x2+1, 100)] = True
    return mask


def add_mask(masks, x1, y1, x2, y2, mask_id):
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    masks[max(y1, 0):min(y2 + 1, 100), max(x1, 0):min(x2 + 1, 100)] = mask_id


def add_random_mask(masks, mask_id, random):
    x1, x2, y1, y2 = random.randint(0, 100, 4)
    add_mask(masks, x1, y1, x2, y2, mask_id)


class TestMatchSequence(unittest.TestCase):

    def test_returns_iterable_over_image_detections_from_sequences(self):
        ground_truth = {
            '000000': make_generator([(1,)]),
            '000001': make_generator([(2,), (3,)]),
            '000002': make_generator([(4, 5)]),
            '000003': make_generator([(6,)]),
            '000004': make_generator([(7,)])
        }
        submission = {
            '000000': make_generator([['1']]),
            '000001': make_generator([['2'], ['3']]),
            '000002': make_generator([['4', '5']]),
            '000003': make_generator([['6']]),
            '000004': make_generator([['7']])
        }
        matches = gt_loader.match_sequences(ground_truth, submission)
        for gt, detect in matches:
            self.assertIsInstance(gt, list)
            self.assertIsInstance(detect, list)
            self.assertEqual(len(gt), len(detect))
            for idx in range(len(gt)):
                self.assertEqual(gt[idx], int(detect[idx]))

    def test_errors_if_missing_sequence(self):
        ground_truth = {
            '000000': make_generator([(1,)]),
            '000001': make_generator([(2,), (3,)]),
            '000002': make_generator([(4, 5)]),
            '000003': make_generator([(6,)]),
            '000004': make_generator([(7,)])
        }
        submission = {
            '000000': make_generator([['1']]),
            # '000001': make_generator([['2'], ['3']]),
            '000002': make_generator([['4', '5']]),
            # '000003': make_generator([['6']]),
            '000004': make_generator([['7']])
        }
        with self.assertRaises(ValueError) as cm:
            next(gt_loader.match_sequences(ground_truth, submission))
        msg = str(cm.exception)
        self.assertIn('000001', msg)
        self.assertIn('000003', msg)

    def test_warns_if_extra_sequence(self):
        ground_truth = {
            '000000': make_generator([(1,)]),
            '000001': make_generator([(2,), (3,)]),
            '000002': make_generator([(4, 5)]),
            '000003': make_generator([(6,)]),
            '000004': make_generator([(7,)])
        }
        submission = {
            '000000': make_generator([['1']]),
            '000001': make_generator([['2'], ['3']]),
            '000002': make_generator([['4', '5']]),
            '000003': make_generator([['6']]),
            '000004': make_generator([['7']]),
            '000005': make_generator([['8']])
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            next(gt_loader.match_sequences(ground_truth, submission))
            self.assertEqual(1, len(w))
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertIn('000005', str(w[-1].message))

    def test_errors_if_sequences_arent_same_length(self):
        ground_truth = {
            '000000': make_generator([(1,)]),
            '000001': make_generator([(2,), (3,)]),
            '000002': make_generator([(4, 5)]),
            '000003': make_generator([(6,)]),
            '000004': make_generator([(7,), (99,)])
        }
        submission = {
            '000000': make_generator([['1']]),
            '000001': make_generator([['2'], ['3']]),
            '000002': make_generator([['4', '5']]),
            '000003': make_generator([['6']]),
            '000004': make_generator([['7']])
        }
        gen = gt_loader.match_sequences(ground_truth, submission)

        # first 5 images are ok
        for _ in range(5):
            next(gen)
        with self.assertRaises(ValueError) as cm:
            next(gen)
        msg = str(cm.exception)
        self.assertIn('000004', msg)


def make_generator(iterable):
    for thing in iterable:
        yield thing
