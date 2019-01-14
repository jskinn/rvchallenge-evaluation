#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import os.path
import gt_loader
import submission_loader
from pdq import PDQ


def do_evaluation(submission_dir, ground_truth_dir):
    """
    Evaluate a particular image sequence
    :param submission_dir:
    :param ground_truth_dir:
    :return:
    """
    ground_truth = gt_loader.read_ground_truth(ground_truth_dir)
    detections = submission_loader.read_submission(submission_dir, expected_sequence_names=set(ground_truth.keys()))
    matches = gt_loader.match_sequences(ground_truth, detections)
    evaluator = PDQ()
    score = evaluator.score(matches)
    TP, FP, FN = evaluator.get_assignment_counts()
    avg_spatial_quality = evaluator.get_avg_spatial_score()
    avg_label_quality = evaluator.get_avg_label_score()
    avg_overall_quality = evaluator.get_avg_overall_quality_score()
    return {
        'score': score*100,
        'avg_spatial': avg_spatial_quality,
        'avg_label': avg_label_quality,
        'avg_pPDQ': avg_overall_quality,
        'TPs': TP,
        'FPs': FP,
        'FNs': FN
    }


def main(input_dir, output_dir):
    # Get the submission and ground truth directories from the input directory
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    # Validate the directories, make sure they exist
    if not os.path.isdir(submit_dir):
        print("Submit dir \"{0}\" doesn't exist, cannot evaluate".format(submit_dir))
    elif not os.path.isdir(truth_dir):
        print("Ground truth dir \"{0}\" doesn't exist, cannot evaluate".format(truth_dir))
    else:
        result = do_evaluation(submit_dir, truth_dir)

        # Write the result to file, to be read by the leaderboard
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
            output_file.write("\n".join("{0}:{1}".format(k, v) for k, v in result.items()))


if __name__ == '__main__':
    main(input_dir=sys.argv[1], output_dir=sys.argv[2])
