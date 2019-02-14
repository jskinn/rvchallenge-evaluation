import unittest
import numpy as np
from data_holders import BBoxDetInst, GroundTruthInstance
from pdq import PDQ
_SMALL_VAL = 1e-14
_MAX_LOSS = np.log(_SMALL_VAL)


class TestPDQ(unittest.TestCase):

    def setUp(self):
        self.img_size = (2000, 2000)
        self.default_covar = [[1000, 0], [0, 1000]]
        self.default_label_list = [1, 0]
        self.square_mask = np.zeros(self.img_size, dtype=np.bool)
        self.square_mask[750:1250, 750:1250] = True
        self.square_gt = [GroundTruthInstance(self.square_mask, 0, 0, 0)]
        self.gt_box = self.square_gt[0].bounding_box

    def test_perfect_bbox(self):
        detections = [BBoxDetInst(self.default_label_list, self.gt_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertAlmostEqual(score, 1, 4)

    def test_no_detection(self):
        detections = []
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertEqual(score, 0)

    def test_detection_no_gt(self):
        detections = [BBoxDetInst(self.default_label_list, self.gt_box)]
        gts = []
        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertEqual(score, 0)

    def test_half_label_confidence(self):
        detections = [BBoxDetInst([0.5, 0.5], self.gt_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertAlmostEqual(np.sqrt(0.5), score, 4)

    def test_half_position_confidence(self):
        detections = [BBoxDetInst(self.default_label_list, self.gt_box, 0.5)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertAlmostEqual(np.sqrt(0.5), score, 4)

    def test_half_position_and_label_confidences(self):
        detections = [BBoxDetInst([0.5, 0.5], self.gt_box, 0.5)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertAlmostEqual(0.5, score, 4)

    def test_miss_500_pixels(self):
        det_box = [val for val in self.gt_box]
        det_box[2] -= 1
        detections = [BBoxDetInst(self.default_label_list, det_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        expected_spatial_quality = np.exp((_MAX_LOSS*500)/(500*500))

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(expected_gmean, score, 4)

    def test_detect_500_extra_pixels(self):
        det_box = [val for val in self.gt_box]
        det_box[2] += 1
        detections = [BBoxDetInst(self.default_label_list, det_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        expected_spatial_quality = np.exp((_MAX_LOSS*500)/(500*500))

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(expected_gmean, score, 4)

    def test_shift_right_one_pixel(self):
        det_box = [val for val in self.gt_box]
        det_box[2] += 1
        det_box[0] += 1
        detections = [BBoxDetInst(self.default_label_list, det_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        expected_spatial_quality = np.exp((_MAX_LOSS*1000)/(500*500))

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(expected_gmean, score, 4)

    def test_run_score_multiple_times_on_single_pdq_instance(self):
        two_detections = [BBoxDetInst(self.default_label_list, self.gt_box),
                          BBoxDetInst(self.default_label_list, self.gt_box)]
        one_detection = [BBoxDetInst(self.default_label_list, self.gt_box)]

        evaluator = PDQ()
        score_two = evaluator.score([(self.square_gt, two_detections)])
        score_one = evaluator.score([(self.square_gt, one_detection)])

        self.assertAlmostEqual(score_two, 0.5)
        self.assertAlmostEqual(score_one, 1.0)

    def test_no_overlap_box(self):
        det_box = [val for val in self.gt_box]
        box_width = (self.gt_box[2]+1) - self.gt_box[0]
        det_box[0] += box_width
        det_box[2] += box_width

        detections = [BBoxDetInst(self.default_label_list, det_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        expected_spatial_quality = 0

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(expected_gmean, score, 4)

    def test_multiple_detections(self):
        ten_detections = [BBoxDetInst(self.default_label_list, self.gt_box) for _ in range(10)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, ten_detections)])

        self.assertAlmostEqual(score, 0.1)

    def test_multiple_missed_gts(self):

        gts = [val for val in self.square_gt]
        for i in range(9):
            # Create small 11x11 boxes which are missed around an edge of the image (buffer of 2 pixels)
            new_gt_mask = np.zeros(gts[0].segmentation_mask.shape, gts[0].segmentation_mask.dtype)
            new_gt_mask[2:14, 2 + i * 14:14 + i * 14] = np.amax(gts[0].segmentation_mask)
            gts.append(GroundTruthInstance(new_gt_mask, 0, 0, i+1))
        detections = [BBoxDetInst(self.default_label_list, self.gt_box)]
        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 0.1)

    def test_multiple_missed_gts_too_small(self):

        gts = [val for val in self.square_gt]
        for i in range(9):
            # Create small 2x2 boxes which are missed around an edge of the image (buffer of 2 pixels)
            new_gt_mask = np.zeros(gts[0].segmentation_mask.shape, gts[0].segmentation_mask.dtype)
            new_gt_mask[2:4, 2 + i * 4:4 + i * 4] = np.amax(gts[0].segmentation_mask)
            gts.append(GroundTruthInstance(new_gt_mask, 0, 0, i+1))
        detections = [BBoxDetInst(self.default_label_list, self.gt_box)]
        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 1.0)

    def test_missed_gts_and_unmatched_detections(self):
        gts = [val for val in self.square_gt]
        for i in range(10):
            # Create small 11x11 boxes which are missed around an edge of the image (buffer of 2 pixels)
            new_gt_mask = np.zeros(gts[0].segmentation_mask.shape, gts[0].segmentation_mask.dtype)
            new_gt_mask[2:14, 2 + i * 14:14 + i * 14] = np.amax(gts[0].segmentation_mask)
            gts.append(GroundTruthInstance(new_gt_mask, 0, 0, i+1))

        detections = [BBoxDetInst(self.default_label_list, self.gt_box) for _ in range(10)]

        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 1/20.)

    def test_correct_second_detection(self):
        gts = [val for val in self.square_gt]
        detections = [BBoxDetInst(self.default_label_list, [0, 0, 10, 10]),
                      BBoxDetInst(self.default_label_list, self.gt_box)]

        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 0.5)

    def test_no_detections_for_image(self):
        gts1 = [val for val in self.square_gt]
        gts2 = [GroundTruthInstance(self.square_mask, 0, 1, 1)]
        dets1 = [BBoxDetInst(self.default_label_list, self.gt_box)]
        dets2 = []
        evaluator = PDQ()
        score = evaluator.score([(gts1, dets1), (gts2, dets2)])

        self.assertAlmostEqual(score, 0.5)

    def test_no_detections_for_image_with_too_small_gt(self):
        gts1 = [val for val in self.square_gt]
        small_mask = np.zeros(self.img_size, dtype=np.bool)
        small_mask[500:504, 500:501] = True
        gts2 = [GroundTruthInstance(small_mask, 0, 1, 1)]
        dets1 = [BBoxDetInst(self.default_label_list, self.gt_box)]
        dets2 = []
        evaluator = PDQ()
        score = evaluator.score([(gts1, dets1), (gts2, dets2)])

        self.assertAlmostEqual(score, 1.0)

    def test_no_detections_for_image_with_small_and_big_gt(self):
        gts1 = [val for val in self.square_gt]
        small_mask = np.zeros(self.img_size, dtype=np.bool)
        small_mask[500:504, 500:501] = True
        gts2 = [GroundTruthInstance(self.square_mask, 0, 1, 1),
                GroundTruthInstance(small_mask, 0, 1, 2)]
        dets1 = [BBoxDetInst(self.default_label_list, self.gt_box)]
        dets2 = []
        evaluator = PDQ()
        score = evaluator.score([(gts1, dets1), (gts2, dets2)])

        self.assertAlmostEqual(score, 0.5)
