import cv2
import numpy as np


def generate_fake_initialization(images, gt_background_images, gt_obstruction_images):
    It = np.array([img / 255. for img in images])
    I_B = (gt_background_images[2] / 255.)

    # difference = abs(It[2] -
    #                 I_B)
    #
    # _, A = cv2.threshold(
    #    difference, 0.1, 1, cv2.THRESH_BINARY_INV)

    # This A seems perfect.
    _, A = cv2.threshold(
        gt_obstruction_images[2] / 255., 0.1, 1, cv2.THRESH_BINARY_INV)

    A = A[..., np.newaxis]

    I_O = (gt_obstruction_images[2] / 255. * (1-A))

    Vt_O = np.zeros((5, I_O.shape[0], I_O.shape[1], 2))
    Vt_B = np.zeros((5, I_O.shape[0], I_O.shape[1], 2))

    return It, I_O, I_B, A, Vt_O, Vt_B

    # should return It, I_O_init, I_B_init, A_init, Vt_O_init, Vt_B_init
