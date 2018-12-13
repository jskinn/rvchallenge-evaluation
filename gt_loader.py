from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path
import warnings
import json
import cv2
import data_holders
import class_list


def read_ground_truth(directory):
    """
    Read all the ground truth from all the sequences in the given folder.
    A sequence folder is only one

    Folder structure is is:
    sequence_0/
        labels.json
        000000.masks.npz
        ...
    sequence_1/
        labels.json
        000000.masks.npz
        ...
    ...

    :param directory:
    :return:
    """
    sequences = {}
    for sequence_dir in os.listdir(directory):
        sequence_path = os.path.join(directory, sequence_dir)
        if os.path.isdir(sequence_path) and os.path.isfile(os.path.join(sequence_path, 'labels.json')):
            sequences[sequence_dir.lower()] = read_sequence(sequence_path)
    return sequences


def read_sequence(sequence_directory):
    """
    Create a generator to read all the ground truth information for a particular
    Each iteration of the generator returns another generator over the ground truth instances
    for that image.
    Given that the image ids are integers, it is guaranteed to return the ground truth in
    numerical frame order.

    Ground truth format is a labels.json file, and some number of <image_id>.masks.npz files containing
    the segmentation masks.
    labels.json should be:
    {
      <image_id>: {
        <instance_id>: <class name>
        ...
      }
      ...
    }

    :param sequence_directory: The ground truth directory, containing labels.json
    :return:
    """
    with open(os.path.join(sequence_directory, 'labels.json'), 'r') as fp:
        labels = json.load(fp)
    mask_im = None
    current_mask_name = None
    for image_id, image_name in sorted((int(l), l) for l in labels.keys()):
        if '_metadata' in labels[image_name]:
            im_mask_name = labels[image_name]['_metadata']['mask_name']
            im_mask_channel = int(labels[image_name]['_metadata']['mask_channel'])
            if current_mask_name != im_mask_name:
                mask_im = cv2.imread(os.path.join(sequence_directory, im_mask_name))
                current_mask_name = im_mask_name

            yield read_gt_for_image(
                image_id=image_id,
                image_data=labels[image_name],
                masks=mask_im[:, :, im_mask_channel]
            )
        else:
            yield []


def read_gt_for_image(image_id, image_data, masks):
    """
    Read ground truth for a particular image
    :param image_id: The id of the image from the labels
    :param image_data: The image data from the labels json
    :param masks: An image containing all the masks in the image
    :return:
    """
    if len(image_data) > 0:
        for instance_name in sorted(image_data.keys()):
            if not instance_name.startswith('_'):  # Ignore metadata attributes
                detection_data = image_data[instance_name]
                class_id = class_list.get_class_id(detection_data['class'])
                if class_id is not None:
                    mask_id = int(detection_data['mask_id'])
                    yield data_holders.GroundTruthInstance(
                        image_id=image_id,
                        true_class_label=class_id,
                        segmentation_mask=(masks == mask_id),
                        instance_id=int(instance_name),
                        bounding_box=[int(v) for v in detection_data.get('bounding_box', [])],
                        num_pixels=int(detection_data.get('num_pixels', -1))
                    )


def match_sequences(ground_truth_sequences, detection_sequences):
    """
    Match ground truth sequences with detection
    :param ground_truth_sequences:
    :param detection_sequences:
    :return: A generator over pairs of frame ground truth and frame detections, which are both lists
    """
    extra = set(detection_sequences.keys()) - set(ground_truth_sequences.keys())
    if len(extra) > 0:
        warnings.warn("The submission has data for the following extra sequences, which is being ignored: {0}".format(
            sorted(extra)
        ))
    missing = set(ground_truth_sequences.keys()) - set(detection_sequences.keys())
    if len(missing) > 0:
        raise ValueError("The following sequences do not have any detections submitted: {0}".format(sorted(missing)))
    for sequence_id in sorted(ground_truth_sequences.keys()):
        if sequence_id in detection_sequences:
            ground_truth_list = list(ground_truth_sequences[sequence_id])
            detections_list = list(detection_sequences[sequence_id])
            if len(ground_truth_list) != len(detections_list):
                raise ValueError("{0}.json : The number of dectections does not match the number of images "
                                 "({1} vs {2}). Remember that you need to output detections for every image, "
                                 "even if it is an empty list.".format(sequence_id, len(ground_truth_list),
                                                                       len(detections_list)))

            for frame_gt, frame_detections in zip(ground_truth_list, detections_list):
                ground_truth = list(frame_gt)
                detections = list(frame_detections)

                yield ground_truth, detections
