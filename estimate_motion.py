import cv2
import numpy as np
import logging

from visualize import visualize_edgeflow
from mrf_min_sum_test import produce_motion_fields


def edgeflow(img_before, img_after, edge_before, edge_after):
    """
    Calculate edge flow from image.
    Returns:
        edgeflow[array(x, y, deltax, deltay)]: edge flow for each edge point
    """
    cv2.imshow("", edge_before)
    cv2.waitKey(0)
    cv2.imshow("", edge_after)
    cv2.waitKey(0)
    
    logging.info("creating {} edge nodes as factor graph node".format(
        np.count_nonzero(edge_before)))
    patch_size = 7
    max_motion_x = 50
    max_motion_y = 50
    motion_fields, edge_points_before = produce_motion_fields(img_before, img_after, edge_before, patch_size=patch_size, max_motion_x=max_motion_x, max_motion_y=max_motion_y, message_passing_rounds=10)
    height, width = img_before.shape
    edgeflow = []
    for point_id, motion_field in enumerate(motion_fields):
        point_pos = edge_points_before[point_id]
        max_shift_x = max_motion_x - 1 + patch_size // 2
        max_shift_y = max_motion_y - 1 + patch_size // 2
        if point_pos[0] - max_shift_x <= 0 or point_pos[1] - max_shift_y <= 0 \
                or point_pos[0] + max_shift_x >= height or point_pos[1] + max_shift_y >= width:
                    continue
        edgeflow.append([point_pos[0], point_pos[1], motion_field[point_id, 0], motion_field[point_id, 1]])

    return np.array(edgeflow)
