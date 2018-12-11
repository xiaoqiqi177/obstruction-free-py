import cv2
import numpy as np
import logging

from estimate_motion import edgeflow
from ransac import RANSACModel, ransac
from visualize import visualize_edgeflow, visualize_separated_motion

# Deprecated.


class PerspectiveModel(RANSACModel):

    def __init__(self, data_points):
        """
        data_points: list of (4,) array for motion.
        """
        data_points_npy = np.array(data_points)
        source_points = data_points_npy[:, :2]
        target_points = source_points + data_points_npy[:, 2:]
        self.data_points = data_points
        self.transform_mat, _ = cv2.findHomography(
            srcPoints=source_points,
            dstPoints=target_points
        )

    def get_error(self, data_points):
        """
        data_points: list of (4,) array for motion.
        """

        data_points_npy = np.array(data_points)
        source_points = data_points_npy[:, :2]
        target_points = source_points + data_points_npy[:, 2:]

        source_points_homo = np.concatenate(
            [source_points, np.ones([source_points.shape[0], 1])], axis=1)
        predicted_points_homo = source_points_homo.dot(
            self.transform_mat.transpose())

        predicted_points = predicted_points_homo[:, :2] \
            / predicted_points_homo[:, 2:3]

        reconstruct_error = np.linalg.norm(target_points - predicted_points)
        return reconstruct_error


def fit_perspective(motion_points):
    """
    Returns:
        used motion_points indices (set(np.array)), remain motion_points indices.
    """

    data_points_npy = np.array(motion_points)
    source_points = data_points_npy[:, :2]
    target_points = (source_points + data_points_npy[:, 2:])
    transform_mat, mask = cv2.findHomography(
        srcPoints=source_points,
        dstPoints=target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5,
    )
    used_motion_points = [motion_points[idx] for idx in range(
        len(motion_points)) if mask[idx] == 1]
    remain_motion_points = [motion_points[idx] for idx in range(
        len(motion_points)) if mask[idx] == 0]
    return np.array(used_motion_points), np.array(remain_motion_points)


def extract_edgemap(image):
    """
    Extract edge map from image using canny edge detector.
    """
    edge_map = cv2.Canny(image, threshold1=30, threshold2=90)
    return edgemap

def calculate_motion(images, edge_maps):
    """
    Calculate sparse motion map with images and edge maps
    by solving discrete Markov Random field.
    """
    logging.info("calculating motion.")
    edge_motions = []
    for idx in range(len(images)):
        if idx != len(images)-1:
            edge_motion = edgeflow(img_before=images[idx],
                                   img_after=images[idx+1],
                                   edge_before=edge_maps[idx],
                                   edge_after=edge_maps[idx+1])
        else:
            edge_motion = edgeflow(img_before=images[idx],
                                   img_after=images[idx-1],
                                   edge_before=edge_maps[idx],
                                   edge_after=edge_maps[idx-1])
            # reverse direction for final image.
            edge_motion[:, 2:] = -edge_motion[:, 2:]
        # visualize_edgeflow(edge_motion, images[idx].shape)
        edge_motions.append(edge_motion)
    return edge_motions


def separate_motion_fields(sparse_motions):
    """
    Separate motion fields into obstruction and background
    by fitting perspective transform and do RANSAC.
    Args:
        sparse_motion[list(array(K, 4))]
    Returns:
        obstruction_motions[list(array(K, 4))
                                 ], background_motions[list(array(K, 4))]
    """
    logging.info("separating motion fields.")
    obstruction_motions = []
    background_motions = []
    for motion_points in sparse_motions:
        motion_points = [motion_point for motion_point in motion_points]
        logging.info("Find {} motion points.".format(len(motion_points)))
        background_motion, remain_motion_points = fit_perspective(
            motion_points)
        logging.info("Classify {} motion points as background.".format(
            len(background_motion)))

        obstruction_motion, _ = fit_perspective(remain_motion_points)
        logging.info("Classify {} motion points as obstruction.".format(
            len(obstruction_motion)))
        background_motions.append(background_motion)
        obstruction_motions.append(obstruction_motion)

    return obstruction_motions, background_motions


def initial_motion_estimation(images, cached):
    if cached:
        try:
            obstruction_motions, background_motions = np.load(
                "./initial_motion.npy")
            obstruction_motions = [obstruction_motions[i]
                                   for i in range(len(images))]
            background_motions = [background_motions[i]
                                  for i in range(len(images))]
            return obstruction_motions, background_motions
        except Exception as error:
            logging.info(error)
            logging.info("no cache found.")

    edge_maps = [extract_edgemap(image) for image in images]
    motions = calculate_motion(images, edge_maps)
    obstruction_motions, background_motions = separate_motion_fields(motions)

    for om, bm, img in zip(obstruction_motions, background_motions, images):
        visualize_separated_motion(om, bm, img.shape)

    np.save("./initial_motion.npy",
            (obstruction_motions, background_motions))
    logging.info("saved.")

    return obstruction_motions, background_motions


def initial_decomposition(obstruction_motion_maps, dense_motion_maps):
    raise NotImplementedError
    # return It, I_O_init, I_B_init, A_init, Vt_O_init, Vt_B_init


def initialize_motion_based_decomposition(images, cached):
    return initial_decomposition(*initial_motion_estimation(images, cached), cached)
