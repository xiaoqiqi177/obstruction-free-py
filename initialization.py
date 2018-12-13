import cv2
import numpy as np
import logging

from scipy.interpolate import griddata
from estimate_motion import edgeflow
from ransac import RANSACModel
from warp import tf_warp
from visualize import visualize_edgeflow, visualize_separated_motion, visualize_dense_motion

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
    image = cv2.blur(image, (3, 3))
    return cv2.Canny(image, threshold1=30, threshold2=90)


def calculate_motion(images, edge_maps, cached):
    """
    Calculate sparse motion map with images and edge maps
    by solving discrete Markov Random field.
    """
    logging.info("calculating motion.")
    edge_motions = []
    for idx in range(len(images)):
        if idx != 2:
            edge_motion = edgeflow(img_before=images[idx],
                                   img_after=images[2],
                                   edge_before=edge_maps[idx],
                                   edge_after=edge_maps[2])
            #visualize_edgeflow(edge_motion, images[idx].shape)
            edge_motions.append(edge_motion)
    return edge_motions


def interpolate_dense_motion_from_sparse_motion(sparse_motion, image_shape):
    grid_x, grid_y = np.mgrid[0:image_shape[0], 0:image_shape[1]]
    points = sparse_motion[:, :2]
    delta_y = sparse_motion[:, 2]
    delta_x = sparse_motion[:, 3]
    delta_y_grid = griddata(points, delta_y,
                            (grid_x, grid_y), method='nearest', fill_value=0)
    delta_x_grid = griddata(points, delta_x,
                            (grid_x, grid_y), method='nearest', fill_value=0)

    dense_motion = np.stack([delta_y_grid, delta_x_grid], axis=-1)
    visualize_dense_motion(dense_motion)
    return dense_motion


def separate_and_densify_motion_fields(sparse_motions, image_shape):
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

        assert len(remain_motion_points) != 0
        obstruction_motion, _ = fit_perspective(remain_motion_points)
        logging.info("Classify {} motion points as obstruction.".format(
            len(obstruction_motion)))
        
        visualize_separated_motion(obstruction_motion, background_motion, image_shape)
        background_motions.append(
            interpolate_dense_motion_from_sparse_motion(background_motion, image_shape))
        obstruction_motions.append(
            interpolate_dense_motion_from_sparse_motion(obstruction_motion, image_shape))

    return obstruction_motions, background_motions


def initial_motion_estimation(images, cached):

    edge_maps = [extract_edgemap(image) for image in images]
    motions = calculate_motion(images, edge_maps, cached)
    obstruction_motions, background_motions = separate_and_densify_motion_fields(
        motions, images[0].shape)

    return obstruction_motions, background_motions

def warpImg(img, motions):
    height, width = img.shape[0:2]
    x, y = np.meshgrid(range(width), range(height))
    warpx = x + motions[y, x, 1]
    warpy = y + motions[y, x, 0]
    valid_y_index, valid_x_index = np.where((warpx >= 0) & (warpx < width) & (warpy >= 0) & (warpy < height))
    warpy = warpy[valid_y_index, valid_x_index].astype(int)
    warpx = warpx[valid_y_index, valid_x_index].astype(int)
    return warpy, warpx, img[y[valid_y_index, valid_x_index], x[valid_y_index, valid_x_index]]

def align_background(It, background_motions, otype):
    assert otype == 'r' or otype == 'o'
    logging.info("Use frame {} as reference.".format(
        len(background_motions)//2))
    height, width = It.shape[1:3]
    if otype == 'r':
        I_B = np.ones((height, width, 1))
        for frame_id in range(len(background_motions)):
            warpy, warpx, warp_img = warpImg(It[frame_id, :, :, :], background_motions[frame_id])
            I_B[warpy, warpx, :] = np.minimum(I_B[warpy, warpx, :], warp_img)
    else:
        I_B = np.zeros((height, width, 1))
        I_B_cnt = np.zeros((height, width, 1))
        for frame_id in range(len(background_motions)):
            warpy, warpx, warp_img = warpImg(It[frame_id, :, :, :], background_motions[frame_id])
            I_B[warpy, warpx, :] = I_B[warpy, warpx, :] + warp_img
            I_B_cnt[warpy, warpx, :] += 1
        I_B = I_B / (I_B_cnt + 1e-8)
    return I_B

def initialize_motion_based_decomposition(images, otype, cached):
    assert otype == 'r' or otype == 'o'
    # size: 4 * H * W * 2
    obstruction_motions, background_motions = initial_motion_estimation(images, cached)
    obstruction_motions = np.array(obstruction_motions[:2] + [np.zeros_like(obstruction_motions[0])] + obstruction_motions[2:])
    background_motions = np.array(background_motions[:2] + [np.zeros_like(background_motions[0])] + background_motions[2:])
    It = np.array([img / 255. for img in images])
    I_B_init = align_background(It, background_motions, otype)
    if otype == 'o':
        # compute alpha map
        difference = abs(It[2] - I_B_init)
        _, A = cv2.threshold(difference, 0.1, 1, cv2.THRESH_BINARY_INV)
        A_init = A[..., np.newaxis]
        # compute Initial I_O
        I_O_init = It[2] - I_B_init * A_init
    else:
        A_init = None
        I_O_init = It[2] - I_B_init
    Vt_O_init = obstruction_motions
    Vt_B_init = background_motions
    return It, I_O_init, I_B_init, A_init, Vt_O_init, Vt_B_init

