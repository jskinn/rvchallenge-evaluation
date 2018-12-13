import numpy as np
import tests.test_helpers as th
import data_holders
from scipy.stats import multivariate_normal
import time

class TestBBoxDetInst(th.ExtendedTestCase):

    def test_calc_heatmap_int_args(self):
        det = data_holders.BBoxDetInst([0.1], [15, 20, 45, 60], pos_prob=1)
        expected_map = np.zeros((100, 100))
        expected_map[20:61, 15:46] = 1
        heatmap = det.calc_heatmap((100, 100))
        self.assertNPEqual(expected_map, heatmap)

    def test_calc_heatmap_float_args(self):
        det = data_holders.BBoxDetInst([0.1], [15.5, 20.8, 45.3, 60.6], pos_prob=1)
        expected_map = np.zeros((100, 100))
        expected_map[21:61, 16:46] = 1
        expected_map[21:61, 15] = 0.5
        expected_map[21:61, 46] = 0.3
        expected_map[20, 16:46] = 0.2
        expected_map[61, 16:46] = 0.6
        expected_map[20, 15] = 0.5 * 0.2
        expected_map[61, 15] = 0.5 * 0.6
        expected_map[20, 46] = 0.3 * 0.2
        expected_map[61, 46] = 0.3 * 0.6
        heatmap = det.calc_heatmap((100, 100))
        self.assertNPClose(expected_map, heatmap)

    def test_calc_heatmap_mixed_args(self):
        det = data_holders.BBoxDetInst([0.1], [15, 20.8, 45, 60.6], pos_prob=1)
        expected_map = np.zeros((100, 100))
        expected_map[21:61, 15:46] = 1
        expected_map[20, 15:46] = 0.2
        expected_map[61, 15:46] = 0.6
        heatmap = det.calc_heatmap((100, 100))
        self.assertNPClose(expected_map, heatmap)

    def test_calc_heatmap_float_args_pos_prob(self):
        prob = 0.7
        det = data_holders.BBoxDetInst([0.1], [15.5, 20.8, 45.3, 60.6], pos_prob=prob)
        expected_map = np.zeros((100, 100))
        expected_map[21:61, 16:46] = prob
        expected_map[21:61, 15] = 0.5 * prob
        expected_map[21:61, 46] = 0.3 * prob
        expected_map[20, 16:46] = 0.2 * prob
        expected_map[61, 16:46] = 0.6 * prob
        expected_map[20, 15] = 0.5 * 0.2 * prob
        expected_map[61, 15] = 0.5 * 0.6 * prob
        expected_map[20, 46] = 0.3 * 0.2 * prob
        expected_map[61, 46] = 0.3 * 0.6 * prob
        heatmap = det.calc_heatmap((100, 100))
        self.assertNPClose(expected_map, heatmap)

    def test_calc_heatmap_outsize_bounds(self):
        det = data_holders.BBoxDetInst([0.1], [-15.5, 80.8, 45.3, 160.6], pos_prob=1)
        expected_map = np.zeros((100, 100))
        expected_map[81:100, 0:46] = 1
        expected_map[81:100, 46] = 0.3
        expected_map[80, 0:46] = 0.2
        expected_map[80, 46] = 0.3 * 0.2
        heatmap = det.calc_heatmap((100, 100))
        self.assertNPClose(expected_map, heatmap)


class TestPBoxGeneration(th.ExtendedTestCase):

    def setUp(self):
        self.img_size = (1000, 1010)
        self.default_covar = [[1000, 0], [0, 1000]]
        self.default_label_list = [1, 0]
        self.places = 2

    def test_central_pbox(self):
        correct_heatmap = create_correct_pbox_heatmap(self.img_size, [250, 250, 749, 749],
                                                      [self.default_covar, self.default_covar])
        detector_heatmap = data_holders.PBoxDetInst(
            self.default_label_list, [250, 250, 749, 749],
            [self.default_covar, self.default_covar]
        ).calc_heatmap(self.img_size)

        self.assertGreaterEqual(np.min(detector_heatmap), 0.0)
        self.assertLessEqual(np.max(detector_heatmap), 1.0)
        self.assertAlmostEqual(np.max(np.abs(correct_heatmap - detector_heatmap)), 0.0, places=self.places)

    def test_pbox_far_left(self):
        correct_heatmap = create_correct_pbox_heatmap(self.img_size, [0, 250, 499, 749],
                                                      [self.default_covar, self.default_covar])
        detector_heatmap = data_holders.PBoxDetInst(
            self.default_label_list, [0, 250, 499, 749],
            [self.default_covar, self.default_covar]
        ).calc_heatmap(self.img_size)

        self.assertGreaterEqual(np.min(detector_heatmap), 0.0)
        self.assertLessEqual(np.max(detector_heatmap), 1.0)
        self.assertAlmostEqual(np.max(np.abs(correct_heatmap - detector_heatmap)), 0.0, places=self.places)

    def test_pbox_far_right(self):
        correct_heatmap = create_correct_pbox_heatmap(self.img_size, [500, 250, 999, 749],
                                                      [self.default_covar, self.default_covar])
        detector_heatmap = data_holders.PBoxDetInst(self.default_label_list, [500, 250, 999, 749],
                                                    [self.default_covar, self.default_covar]
                                                    ).calc_heatmap(self.img_size)

        self.assertGreaterEqual(np.min(detector_heatmap), 0.0)
        self.assertLessEqual(np.max(detector_heatmap), 1.0)
        self.assertAlmostEqual(np.max(np.abs(correct_heatmap - detector_heatmap)), 0.0, places=self.places)

    def test_pbox_top(self):
        correct_heatmap = create_correct_pbox_heatmap(self.img_size, [250, 0, 749, 499],
                                                      [self.default_covar, self.default_covar])
        detector_heatmap = data_holders.PBoxDetInst(self.default_label_list, [250, 0, 749, 499],
                                                    [self.default_covar, self.default_covar]
                                                    ).calc_heatmap(self.img_size)

        self.assertGreaterEqual(np.min(detector_heatmap), 0.0)
        self.assertLessEqual(np.max(detector_heatmap), 1.0)
        self.assertAlmostEqual(np.max(np.abs(correct_heatmap - detector_heatmap)), 0.0, places=self.places)

    def test_pbox_bottom(self):
        correct_heatmap = create_correct_pbox_heatmap(self.img_size, [250, 500, 749, 999],
                                                      [self.default_covar, self.default_covar])
        detector_heatmap = data_holders.PBoxDetInst(self.default_label_list, [250, 500, 749, 999],
                                                    [self.default_covar, self.default_covar]
                                                    ).calc_heatmap(self.img_size)

        self.assertGreaterEqual(np.min(detector_heatmap), 0.0)
        self.assertLessEqual(np.max(detector_heatmap), 1.0)
        self.assertAlmostEqual(np.max(np.abs(correct_heatmap - detector_heatmap)), 0.0, places=self.places)

    def test_is_faster_than_correct(self):
        start_correct_time = time.time()
        create_correct_pbox_heatmap(self.img_size, [250, 250, 749, 749],
                                    [self.default_covar, self.default_covar])
        end_correct_time = time.time()
        data_holders.PBoxDetInst(
            self.default_label_list, [250, 250, 749, 749],
            [self.default_covar, self.default_covar]
        ).calc_heatmap(self.img_size)
        end_detector_time = time.time()
        correct_time = end_correct_time - start_correct_time
        detector_time = end_detector_time - end_correct_time

        self.assertLess(detector_time, correct_time)

    # def test_correct_is_close_to_approximate(self):
    #     correct_heatmap = create_correct_pbox_heatmap(self.img_size, [250, 250, 749, 749],
    #                                                   [self.default_covar, self.default_covar])
    #     approx_heatmap = create_approx_pbox_heatmap(self.img_size, [250, 250, 749, 749],
    #                                                 [self.default_covar, self.default_covar])
    #
    #     self.assertAlmostEqual(np.max(np.abs(correct_heatmap - approx_heatmap)), 0.0, places=self.places)
    #
    # def test_detector_is_close_to_approximate(self):
    #     detector_heatmap = data_holders.PBoxDetInst(self.default_label_list, [250, 250, 749, 749],
    #                                                 [self.default_covar, self.default_covar]
    #                                                 ).calc_heatmap(self.img_size)
    #     approx_heatmap = create_approx_pbox_heatmap(self.img_size, [250, 250, 749, 749],
    #                                                 [self.default_covar, self.default_covar])
    #
    #     self.assertAlmostEqual(np.max(np.abs(detector_heatmap - approx_heatmap)), 0.0, places=self.places)


def create_correct_pbox_heatmap(img_shape, box, covs):
    _HEATMAP_THRESH = 0.0027
    # Note cuurently assume covs is a list of two lists examining first x then y variances
    # Note currently assume box is given in format [x1,y1,x2,y2]
    # get all covs in format (y,x) to match matrix ordering
    covs2 = [np.flipud(np.fliplr(cov)) for cov in covs]

    # Create multivariate Gaussians for both box corners
    g1 = multivariate_normal(mean=[box[1], box[0]], cov=covs2[0])
    # note we reverse bottom right to calculate probs and we
    # reverse correlations to be correct as well
    g2 = multivariate_normal(mean=[img_shape[0] - (box[3]+1), img_shape[1] - (box[2]+1)],
                             cov=np.array(covs2[1]).T)

    # Define the points within the image
    positions = np.dstack(np.mgrid[1:img_shape[0]+1, 1:img_shape[1]+1])

    # Define points around the outside of the image.
    pos_outside_y = np.dstack(np.mgrid[0:1, 0:img_shape[1]+1])  # points above the image
    pos_outside_x = np.dstack(np.mgrid[0:img_shape[0]+1, 0:1])  # points left of the image

    # Calculate the overall cdf for them main image points and outside border points
    prob1 = g1.cdf(positions)
    prob1_outside_y = g1.cdf(pos_outside_y)
    prob1_outside_x = g1.cdf(pos_outside_x)

    # Final probability is your overall cdf minus the probability in-line with that point along
    # the border for both dimensions plus the cdf at (-1, -1) which has points counted twice otherwise
    prob1 -= prob1_outside_y[1:].reshape((1, len(prob1_outside_y)-1))
    prob1 -= prob1_outside_x[1:].reshape((len(prob1_outside_x)-1, 1))
    prob1 += prob1_outside_y[0]

    # Repeat for other corner
    prob2 = g2.cdf(positions)
    prob2_outside_y = g2.cdf(pos_outside_y)
    prob2_outside_x = g2.cdf(pos_outside_x)

    prob2 -= prob2_outside_y[1:].reshape((1, len(prob2_outside_y)-1))
    prob2 -= prob2_outside_x[1:].reshape((len(prob2_outside_x)-1, 1))
    prob2 += prob2_outside_y[0]

    # flip left-right and up-down to provide probability in from bottom-right corner
    prob2 = np.fliplr(np.flipud(prob2))

    # generate final heatmap
    heatmap = prob1 * prob2

    # # Hack to enforce that there are no pixels with probs greater than 1 due to floating point errors
    # heatmap[heatmap > 1] = 1

    if np.min(heatmap) < 0:
        print("Whoops, correct heatmap is negative")
    if np.max(heatmap) > 1:
        print("Whoops, correct heatmap is > 1")

    heatmap[heatmap < _HEATMAP_THRESH] = 0

    return heatmap


def create_approx_pbox_heatmap(img_shape, box, covs):
    # Note cuurently assume covs is a list of two lists examining first x then y variances
    # Note currently assume box is given in format [x1,y1,x2,y2]
    # get all covs in format (y,x) to match matrix ordering
    covs2 = [np.flipud(np.fliplr(cov)) for cov in covs]
    n_samples = 100000
    stream = np.random.RandomState(42)
    corners = stream.multivariate_normal([box[1], box[0]], covs2[0], n_samples)
    corners = np.hstack((corners, stream.multivariate_normal([box[3], box[2]], covs2[1], n_samples)))

    heatmap = inside(corners, img_shape)

    return heatmap


def inside(corners, img_shape):
    p = np.zeros(img_shape)
    count = 0
    for c in corners:
        c1y, c1x, c2y, c2x = c.astype(np.int)
        count += 1

        # Check if the corners create a valid box or not
        if c1y >= 0 and c1x >= 0 and c2y < img_shape[0] and c2x < img_shape[1] and c1y <= c2y and c1x <= c2x:
            # if valid we add one to all pixels within the box (inclusive)
            p[c1y:c2y+1, c1x:c2x+1] += 1

    # divide by the number of corners to get the probability that a point will be within the bounding box in quesiton
    return p / corners.shape[0]
