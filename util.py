import cv2
import numpy as np


def scale_images(images, from_scale, to_scale):
    """
    sample images from from_scale to to_scale.
    Args:
        images[Image]: NHW(C) image.
        scale[float], to_scale[float]
    """
    scaled_images = []
    for i in range(images.shape[0]):
        W = images.shape[2]
        H = images.shape[1]
        C = images.shape[3]
        new_W = int(W / from_scale * to_scale)
        new_H = int(H / from_scale * to_scale)
        if C == 1:
            scaled_images.append(cv2.resize(
                images[i], (new_W, new_H))[..., np.newaxis])
        else:
            scaled_images.append(cv2.resize(images[i], (new_W, new_H)))

    return np.array(scaled_images)


def scale_image(image, from_scale, to_scale):
    """
    sample images from from_scale to to_scale.
    Args:
        images[Image]: HW(C) image.
        scale[float], to_scale[float]
    """
    W = image.shape[1]
    H = image.shape[0]
    C = image.shape[2]
    new_W = int(W / from_scale * to_scale)
    new_H = int(H / from_scale * to_scale)
    if C == 1:
        return cv2.resize(image, (new_W, new_H))[..., np.newaxis]
    else:
        return cv2.resize(image, (new_W, new_H))
