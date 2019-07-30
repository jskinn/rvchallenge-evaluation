from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path
import warnings
import json
import cv2
import data_holders
import class_list


def read_ground_truth(directory, valid_sequences=None, start_index=0, end_index=-1):
    """
    Read all the ground truth from all the sequences in the given folder.
    Each sequence is a folder containing a json file and some number of mask images

    Folder structure is is:
    000000/
        labels.json
        0.png
        1.png
        ...
    000001/
        labels.json
        0.png
        1.png
        ...
    ...

    :param directory: location of root directory where folders containing sequence's gt data are located
    :param valid_sequences: A whitelist of sequences to read, as integers
    :param start_index: The first index to slice the returned ground truth to. Default 0
    :param end_index: The end index to stop slicing the returned ground truth. Defaults to all remaining
    :return: sequences: dictionary of sequence gt generators
    """
    sequences = {}
    for sequence_dir in os.listdir(directory):
        if valid_sequences is None or int(sequence_dir) in valid_sequences:
            sequence_path = os.path.join(directory, sequence_dir)
            if os.path.isdir(sequence_path) and os.path.isfile(os.path.join(sequence_path, 'labels.json')):
                sequences[sequence_dir.lower()] = read_sequence(sequence_path, start_index, end_index)
    return sequences


def read_sequence(sequence_directory, start_index=0, end_index=-1):
    """
    Create a generator to read all the ground truth information for a particular sequence.
    Each iteration of the generator returns another generator over the ground truth instances
    for that image.
    Given that the image ids are integers, it is guaranteed to return the ground truth in
    numerical frame order.

    Ground truth format is a labels.json file, and some number of images containing
    the segmentation masks.
    labels.json should be:
    {
      <image_id>: {
        "_metadata": {"mask_name":<mask_image_filename>, "mask_channel": <mask_image_channel>},
        <instance_id>: {
            "class": <class_name>, "mask_id": <mask_id>, "bounding_box": <bounding_box>
            }
        ...
      }
      ...
    }

    <image_id> : The id of the particular image, a 6-digit id in ascending frame order.
    The first frame is "000000", the second "000001", etc.

    <mask_image_filename> : Path of the mask image containing the masks for this image,
    which should exist in the same folder as labels.json (e.g. "0.png")
    Note that the masks are contained in one channel of this RGB image.

    <mask_image_channel> : channel from the mask image containing the masks for this image.
    As OpenCV is in BGR order, the 0th channel will be the Blue channel of the mask image.

    <instance_id> : id given to the object itself (not just the current visualised instance thereof).
    These ids are consistent between frames, and can be used for instance tracking.

    <class_name> : string name of the given object's class

    <mask_id> : the value of the mask image pixels where this object appears.
    That is, if <mask_id> is 10, <mask_image_filename> is "0.png", and <mask_image_channel> is 0,
    then the pixels for this object are all the places the blue channel of "0.png" is 10.

    <bounding_box> : bounding box encapsulating instance mask in format [left, top, right, bottom]


    :param sequence_directory: The ground truth directory, containing labels.json
    :param start_index: The first index to slice the returned ground truth to. Default 0
    :param end_index: The end index to stop slicing the returned ground truth. Defaults to all remaining
    :return: sequence_generator: generator for the gt of a given sequence over all images in that sequence.
    Note that the sequence_generator produces an image gt generator over all gt instances in that image.
    """
    with open(os.path.join(sequence_directory, 'labels.json'), 'r') as fp:
        labels = json.load(fp)
    if end_index < start_index:
        end_index = max(int(k) for k in labels.keys()) + 1
    for image_id, image_name in sorted((int(l), l) for l in labels.keys() if start_index <= int(l) < end_index):
        if '_metadata' in labels[image_name]:
            im_mask_name = labels[image_name]['_metadata']['mask_name']
            mask_im = cv2.imread(os.path.join(sequence_directory, im_mask_name), cv2.IMREAD_ANYDEPTH)

            yield read_gt_for_image(
                image_id=image_id,
                image_data=labels[image_name],
                masks=mask_im
            )
        else:
            yield []


def read_gt_for_image(image_id, image_data, masks):
    """
    Read ground truth for a particular image
    :param image_id: The id of the image from the labels
    :param image_data: The image data from the labels json
    :param masks: A greyscale image containing all the masks in the image
    :return: image_generator: generator of GroundTruthInstances objects for each gt instance present in
    the given image.
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
