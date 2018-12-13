from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from functools import reduce  # forward compatibility for Python 3
import operator

def generate_bounding_box_from_mask(mask):
    flat_x = np.any(mask, axis=0)
    flat_y = np.any(mask, axis=1)
    if not np.any(flat_x) and not np.any(flat_y):
        raise ValueError("No positive pixels found, cannot compute bounding box")
    xmin = np.argmax(flat_x)
    ymin = np.argmax(flat_y)
    xmax = len(flat_x) - 1 - np.argmax(flat_x[::-1])
    ymax = len(flat_y) - 1 - np.argmax(flat_y[::-1])
    return [xmin, ymin, xmax, ymax]


def bb_iou(boxA, boxB):
    # Code from online source
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    # determine the (x, y)-coordinates of the intersection rectangle
    # Assumes boxes input in format [[xmin,ymin],[xmax,ymax]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, (xB - xA) + 1) * max(0, (yB - yA) + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = ((boxA[2] - boxA[0]) + 1) * ((boxA[3] - boxA[1]) + 1)
    boxBArea = ((boxB[2] - boxB[0]) + 1) * ((boxB[3] - boxB[1]) + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def put_image_in_centre_of_other_image(img1, img2, add=False):
    """
    Code to add a mask to the centre of a given image (both 2D at the moment)
    :param img1: image to be putting other image in the centre of
    :param img2: image to put in the centre of another image
    :param add: flag as to whether to add the image to the centre of the other one or replace all pixels
    :return: Final image with img2 now in the centre of img1
    """

    borderx_size = (img1.shape[1] - img2.shape[1])/2
    bordery_size = (img1.shape[0] - img2.shape[0])/ 2

    x1 = int(borderx_size)
    x2 = int(borderx_size + img2.shape[1])
    y1 = int(bordery_size)
    y2 = int(bordery_size + img2.shape[0])

    new_img = img1.copy()
    if add:
        new_img[y1:y2, x1:x2] += img2
    else:
        new_img[y1:y2, x1:x2] = img2

    return new_img

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value