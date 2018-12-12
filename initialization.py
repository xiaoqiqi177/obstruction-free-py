import cv2
import numpy as np
import logging

from scipy.interpolate import griddata
from estimate_motion import edgeflow
from ransac import RANSACModel, ransac
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
        visualize_edgeflow(edge_motion, images[idx].shape)
        edge_motions.append(edge_motion)
    return edge_motions


def interpolate_dense_motion_from_sparse_motion(sparse_motion, image_shape):
    grid_x, grid_y = np.mgrid[0:image_shape[0], 0:image_shape[1]]
    points = sparse_motion[:, :2]
    delta_y = sparse_motion[:, 2]
    delta_x = sparse_motion[:, 3]
    delta_y_grid = griddata(points, delta_y,
                            (grid_x, grid_y), method='cubic', fill_value=0)
    delta_x_grid = griddata(points, delta_x,
                            (grid_x, grid_y), method='cubic', fill_value=0)

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

    for om, bm, img in zip(obstruction_motions, background_motions, images):
        visualize_separated_motion(om, bm, img.shape)

    return obstruction_motions, background_motions


def align_background(It, background_motions, otype):
    assert otype == 'r' or otype == 'o'
    reference_background = background_motions[len(background_motions)//2]
    logging.info("Use frame {} as reference.".format(
        len(background_motions)//2))
    height, width = It.shape[1:3]
    if otype == 'r':
        I_B = np.ones((height, width, 1))
        for frame_id in len(background_motions):
            for y in range(height):
                for x in range(width):
                    warpx = x + background_motions[frame_id, y, x, 1]
                    warpy = y + background_motions[frame_id, y, x, 0]
                    if warpx < 0 or warpy < 0 or warpx >= width or warpy >= height:
                        continue
                    else:
                        I_B[y, x, :] = min(I_B[y, x, :], It[frame_id, warpy, warpx])
    else:
        I_B = np.zeros((height, width, 1))
        for frame_id in len(background_motions):
            for y in range(height):
                for x in range(width):
                    warpx = x + background_motions[frame_id, y, x, 1]
                    warpy = y + background_motions[frame_id, y, x, 0]
                    if warpx < 0 or warpy < 0 or warpx >= width or warpy >= height:
                        continue
                    else:
                        I_B[y, x, :] += It[frame_id, warpy, warpx]
    return I_B

def initialize_motion_based_decomposition(images, otype, cached):
    assert otype == 'r' or otype == 'o'
    obstruction_motions, background_motions = initial_motion_estimation(images, cached)
    It = np.array([img / 255. for img in images])
    I_B_init = align_background(
        It, background_motions, otype)
    if otype == 'o':
        difference = abs(It[2] - I_B)
        _, A = cv2.threshold(difference, 0.1, 1, cv2.THRESH_BINARY_INV)
        A_init = A[..., np.newaxis]
       
        #I_O is actually the I_O * (1 - A)
        I_O = It[2] - I_B * A
    elif otype == 'r':
        A_init = None
        I_O = It[2] - I_B
    Vt_O_init = obstruction_motions
    Vt_B_init = background_motions
    import IPython
    IPython.embed()
    return It, I_O_init, I_B_init, A_init, Vt_O_init, Vt_B_init

