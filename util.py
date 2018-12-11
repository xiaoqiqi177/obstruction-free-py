import cv2
import numpy as np


def scale_images(images, from_shape, to_shape):
    """
    sample images from from_shape to to_shape.
    Args:
        images[Image]: NHW(C) image.
        from_shape[h, w], to_shape[h, w]
    """
    scaled_images = []
    for i in range(images.shape[0]):
        W = images.shape[2]
        H = images.shape[1]
        assert H == from_shape[0] and W == from_shape[1]
        C = images.shape[3]
        new_W = to_shape[1]
        new_H = to_shape[0]
        if C == 1:
            scaled_images.append(cv2.resize(
                images[i], (new_W, new_H))[..., np.newaxis])
        else:
            scaled_images.append(cv2.resize(images[i], (new_W, new_H)))

    return np.array(scaled_images)


def scale_image(image, from_shape, to_shape):
    """
    sample images from from_shape to to_shape.
    Args:
        images[Image]: HW(C) image.
        from_shape[h, w], to_shape[h, w]
    """
    W = image.shape[1]
    H = image.shape[0]
    assert H == from_shape[0] and W == from_shape[1]
    C = image.shape[2]
    new_W = to_shape[1]
    new_H = to_shape[0]
    if C == 1:
        return cv2.resize(image, (new_W, new_H))[..., np.newaxis]
    else:
        return cv2.resize(image, (new_W, new_H))
