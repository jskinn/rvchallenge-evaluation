#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path
import gt_loader
import submission_loader
from pdq import PDQ
import argparse


def do_evaluation(submission_dir, ground_truth_dir, sequences=None, num_frames=-1, start_frame=0):
    """
    Evaluate a particular image sequence
    :param submission_dir: location of the detections .json files (one for each sequence)
    :param ground_truth_dir: location of the ground-truth folders (one for each sequence).
    Each ground-truth folder must contain mask images (.png format) and a matching labels.json file.
    :param sequences: A whitelist of sequence ids to include, as integers
    :param num_frames: The number of frames to read from each sequence, default is all available.
    :param start_frame: The index of the first frame to read
    :return: Dictionary containing summary of all metrics used in competition leaderboard
    (score, average spatial quality, average label quality, average overall quality (avg_pPDQ),
    true positives, false positives, and false negatives)
    """
    ground_truth = gt_loader.read_ground_truth(ground_truth_dir, sequences,
                                               start_index=start_frame, end_index=start_frame + num_frames)
    detections = submission_loader.read_submission(submission_dir, expected_sequence_names=set(ground_truth.keys()),
                                                   start_index=start_frame, end_index=start_frame + num_frames)
    matches = gt_loader.match_sequences(ground_truth, detections)
    evaluator = PDQ()
    score = evaluator.score(matches)
    TP, FP, FN = evaluator.get_assignment_counts()
    avg_spatial_quality = evaluator.get_avg_spatial_score()
    avg_label_quality = evaluator.get_avg_label_score()
    avg_overall_quality = evaluator.get_avg_overall_quality_score()
    avg_fp_quality = evaluator.get_avg_fp_score()
    return {
        'score': score*100,
        'avg_spatial': avg_spatial_quality,
        'avg_label': avg_label_quality,
        'avg_pPDQ': avg_overall_quality,
        'avg_fp_quality': avg_fp_quality,
        'TPs': TP,
        'FPs': FP,
        'FNs': FN
    }


def main(submit_dir, truth_dir, save_folder, sequences=None, num_frames=-1, start_frame=0):
    # Validate the directories, make sure they exist
    if not os.path.isdir(submit_dir):
        print("Submit dir \"{0}\" doesn't exist, cannot evaluate".format(submit_dir))
    elif not os.path.isdir(truth_dir):
        print("Ground truth dir \"{0}\" doesn't exist, cannot evaluate".format(truth_dir))
    else:
        result = do_evaluation(submit_dir, truth_dir,sequences, num_frames, start_frame)

        # Write the result to file
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(os.path.join(save_folder, 'scores.txt'), 'w') as output_file:
            output_file.write("\n".join("{0}:{1}".format(k, v) for k, v in result.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', '-g', help='location of folder with sub-folders of gt annotations'
                                                  '(one sub-folder per sequence)')
    parser.add_argument('--det_folder', '-d', help='location of folder containing detection .json files'
                                                   '(must have one .json file for each sequence)')
    parser.add_argument('--save_folder', '-s', help='location of folder where scores.txt will be saved')
    parser.add_argument('--num_frames', '-n', type=int, help='Number of frames to include from each sequence')
    parser.add_argument('--start_frame', '-b', type=int,
                        help='The index of the first frame to test on from each sequence')
    parser.add_argument('--sequences', '-i', metavar='SEQ', nargs='+', default=set(range(18)),
                        help='Specific sequences to include, by index.')
    args = parser.parse_args()
    main(
        truth_dir=args.gt_folder,
        submit_dir=args.det_folder,
        save_folder=args.save_folder,
        sequences={int(idx) for idx in args.sequences},
        num_frames=args.num_frames,
        start_frame=args.start_frame
    )
