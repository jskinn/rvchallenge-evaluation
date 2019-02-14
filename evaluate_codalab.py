#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import os.path
from evaluate import do_evaluation


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
