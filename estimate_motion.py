import cv2
import numpy as np
import logging

from visualize import visualize_edgeflow


def penality_to_probability(penality):
    """
    Transform penality(float) to probability()
    """
    raise NotImplementedError


def NCC(lhs_patch, rhs_patch):
    """
    Calculate normalized cross correlation
    """
    raise NotImplementedError


def solve_mrf(img_before, img_after, edge_before, edge_after):
    cv2.imshow("", edge_before)
    cv2.waitKey(0)
    logging.info("creating {} edge nodes as factor graph node".format(
        np.count_nonzero(edge_before)))
    # start building loss functions
    raise NotImplementedError


def get_pyramid(image):
    pyramid = cv2.buildOpticalFlowPyramid(
        image, winSize=(21, 21), maxLevel=3)
    return pyramid


def edgeflow(img_before, img_after, edge_before, edge_after):
    """
    Calculate edge flow from image.
    Returns:
        edgeflow[array(K,4)]: edge flow for each edge point with format (y, x, delta_y, delta_x).
    """
    edge_before_points = np.expand_dims(np.array(
        np.where(edge_before != 0), dtype=np.float32).transpose(), axis=1)  # (N, 1, 2) for each point.
    '''
    cv2.imshow("", edge_before)
    cv2.waitKey(0)

    cv2.imshow("", edge_after)
    cv2.waitKey(0)
    '''
    edge_after_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prevImg=cv2.blur(img_before, (3, 3)),
        nextImg=cv2.blur(img_after, (3, 3)),
        prevPts=edge_before_points[:, :, ::-1],
        nextPts=None,
    )

    edge_after_points = edge_after_points[:, :, ::-1]
    motion_points = edge_after_points - edge_before_points  # (N, 1, 2)
    edgeflow = np.concatenate(
        [edge_before_points[status == 1], motion_points[status == 1]], axis=1)  # (N*, 4)

    return edgeflow
