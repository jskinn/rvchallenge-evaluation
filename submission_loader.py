from __future__ import absolute_import, division, print_function, unicode_literals

import os.path
import json
import numpy as np
import data_holders
import class_list


def read_submission(directory, expected_sequence_names, start_index=0, end_index=-1):
    """
    Read all the submissions for all the sequences outlined in the given folder.
    Each sequence's detections are provided in a file ending with 'detections.json'.
    Each detections.json file contains a dictionary which has a key 'detections' containing a list of list of
    detections for each image.
    Individual detections are given as dictionaries which have the keys:
        'img_size': (height x width)
        'img_num': int identifying which image the detection is a part of
        'label_probs': full list of probabilities that the detection is describing each of the classes
        'bbox': coordinates of the bounding box corners [left, top, right, bottom]
        'covars': covariances for the top-left and bottom-right corners respectively.
            Each with format [[xx, xy], [yx, yy]]. Covariances must be positive semi-definite
            or all zeros (regular BBox).
    Order of list of lists should correspond with ground truth image order.
    If an image does not have any detections, entry should be an empty list.

    :param directory: location of each sequence's submission json file.
    :param expected_sequence_names: The list of sequence names we're looking for submissions for.
    :param start_index: The first index to slice the detections for each sequence
    :param end_index: The index to stop slicing the detections for each sequence
    :return: generator of generator of DetectionInstances for each image
    """
    sequences = {}
    for root, _, files in os.walk(directory):
        for sequence_name in expected_sequence_names:
            json_file = sequence_name + '.json'
            if json_file in files:
                if sequence_name in sequences:
                    raise ValueError("{0} : more than one json file found for sequence, {1} and {2}".format(
                        sequence_name,
                        os.path.relpath(sequences[sequence_name], directory),
                        os.path.relpath(os.path.join(root, json_file), directory)
                    ))
                else:
                    sequences[sequence_name] = os.path.join(root, json_file)
    return {
        sequence_name: read_sequence(sequence_path, start_index=start_index, end_index=end_index)
        for sequence_name, sequence_path in sequences.items()
    }


def read_sequence(sequence_json, start_index=0, end_index=-1):
    """
    Read a sequence's detection json file.
    json file contains a dictionary which has a key 'detections' containing a list of list of
    detections for each image.
    Individual detections are given as dictionaries which have the keys:
        'img_size': (height x width)
        'img_num': int identifying which image the detection is a part of
        'label_probs': full list of probabilities that the detection is describing each of the classes
        'bbox': coordinates of the bounding box corners [left, top, right, bottom]
        'covars': covariances for the top-left and bottom-right corners respectively.
            Each with format [[xx, xy], [yx, yy]]. Covariances must be positive semi-definite
            or all zeros (regular BBox).
    Order of list of lists should correspond with ground truth image order.
    If an image does not have any detections, entry should be an empty list.
    :param sequence_json:
    :param start_index: The first index to slice the read detections.
    :param end_index: The index to stop slicing the detections
    :return: generator of generator of DetectionInstances for each image
    """
    with open(sequence_json, 'r') as f:
        data_dict = json.load(f)
    sequence_name = os.path.basename(sequence_json)

    # Validate
    if 'classes' not in data_dict:
        raise KeyError("{0} : Missing key \'classes\'".format(sequence_name))
    if 'detections' not in data_dict:
        raise KeyError("{0} : Missing key \'detections\'".format(sequence_name))
    if len(set(data_dict['classes']) & (set(class_list.CLASS_IDS) | set(class_list.SYNONYMS.keys()))) <= 0:
        raise ValueError("{0} : classes does not contain any recognized classes".format(sequence_name))

    # Work out which of the submission classes correspond to which of our classes
    our_class_ids = []
    sub_class_ids = []
    for sub_class_id, class_name in enumerate(data_dict['classes']):
        our_class_id = class_list.get_class_id(class_name)
        if our_class_id is not None:
            our_class_ids.append(our_class_id),
            sub_class_ids.append(sub_class_id)

    # create a detection instance for each detection described by dictionaries in dict_dets
    dict_dets = data_dict['detections']
    if end_index < start_index or end_index > len(dict_dets):
        end_index = len(dict_dets)
    for img_idx in range(start_index, end_index):
        yield gen_img_pboxes(dict_dets[img_idx], (our_class_ids, sub_class_ids),
                             num_classes=len(data_dict['classes']), img_idx=img_idx, sequence_name=sequence_name)


def gen_img_pboxes(img_dets, class_mapping, num_classes=len(class_list.CLASSES), img_idx=-1, sequence_name='unknown'):
    """
    Generate DetectionInstances for a given image.
    DetectionInstances will be of sub-class BBoxDetInst or PBoxDetInst.
    BBoxDetInst is DetectionInstance for standard bounding box detection.
    PBoxDetInst is DetectionInstance for probabilistic bounding box detection.
    :param img_dets: list of detections given as dictionaries.
    Individual detection dictionaries have the keys:
        'img_size': (height x width)
        'img_num': int identifying which image the detection is a part of
        'label_probs': full list of probabilities that the detection is describing each of the classes
        'bbox': coordinates of the bounding box corners [left, top, right, bottom]
        'covars': covariances for the top-left and bottom-right corners respectively.
            Each with format [[xx, xy], [yx, yy]]. Covariances must be positive semi-definite
            or all zeros (regular BBox).
    :param class_mapping: A pair of lists of indexes, the first to our class list, and the second to theirs
    :param num_classes: The number of classes to expect
    :param img_idx: The current image index, for error reporting
    :param sequence_name: The current image name, for error reporting
    :return: generator of DetectionInstances
    """
    # Handle no detections for the image
    for det_idx, det in enumerate(img_dets):
        if 'label_probs' not in det:
            raise KeyError(make_error_msg("missing key \'label_probs\'", sequence_name, img_idx, det_idx))
        if 'bbox' not in det:
            raise KeyError(make_error_msg("missing key \'bbox\'", sequence_name, img_idx, det_idx))
        if len(det['label_probs']) != num_classes:
            raise KeyError(make_error_msg("The number of class probabilities doesn't match the number of classes",
                                          sequence_name, img_idx, det_idx))
        if len(det['bbox']) != 4:
            raise ValueError(make_error_msg("The bounding box must contain exactly 4 entries",
                                            sequence_name, img_idx, det_idx))
        if det['bbox'][2] < det['bbox'][0]:
            raise ValueError(make_error_msg("The x1 coordinate must be less than the x2 coordinate",
                                            sequence_name, img_idx, det_idx))
        if det['bbox'][3] < det['bbox'][1]:
            raise ValueError(make_error_msg("The y1 coordinate must be less than the y2 coordinate",
                                            sequence_name, img_idx, det_idx))

        # Use numpy list indexing to move specific indexes from the submission
        label_probs = np.zeros(len(class_list.CLASSES), dtype=np.float32)
        label_probs[class_mapping[0]] = np.array(det['label_probs'])[class_mapping[1]]
        total_prob = np.sum(label_probs)

        if total_prob > 0.5:    # Arbitrary theshold for classes we care about.
            # Normalize the label probability
            if total_prob > 1:
                label_probs /= total_prob
            if 'covars' not in det or det['covars'] == [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]:
                yield data_holders.BBoxDetInst(
                    class_list=label_probs,
                    box=det['bbox']
                )
            else:
                covars = np.array(det['covars'])
                if covars.shape != (2, 2, 2):
                    raise ValueError(make_error_msg("Key 'covars' must contain 2 2x2 matrices",
                                                    sequence_name, img_idx, det_idx))
                if not np.allclose(covars.transpose((0, 2, 1)), covars):
                    raise ValueError(make_error_msg("Given covariances are not symmetric",
                                                    sequence_name, img_idx, det_idx))
                if not is_positive_semi_definite(covars[0]):
                    raise ValueError(make_error_msg("The upper-left covariance is not positive semi-definite",
                                                    sequence_name, img_idx, det_idx))
                if not is_positive_semi_definite(covars[1]):
                    raise ValueError(make_error_msg("The lower-right covariance is not positive semi-definite",
                                                    sequence_name, img_idx, det_idx))
                yield data_holders.PBoxDetInst(
                    class_list=label_probs,
                    box=det['bbox'],
                    covs=det['covars']
                )


def is_positive_semi_definite(mat):
    """
    Check if a matrix is positive semi-definite, that is, all it's eigenvalues are positive.
    All covariance matrices must be positive semi-definite.
    Only works on symmetric matricies (due to eigh), so check that first
    :param mat:
    :return:
    """
    eigvals, _ = np.linalg.eigh(mat)
    return np.all(eigvals >= -1e-14)


def make_error_msg(msg, sequence_name, img_idx, det_idx):
    return "{0}, image index {1}, detection index {2} : {3}".format(sequence_name, img_idx, det_idx, msg)
