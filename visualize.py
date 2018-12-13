import numpy as np
import cv2


def visualize_edgeflow(edgeflow, image_shape):
    """
    Args:
        edgeflow[array]: shape (K, 4)
        image_shape[list]: of (h, w)
    """

    # Draw edgeflow as a dense image.
    dense_flow = np.zeros((image_shape[0], image_shape[1], 2))

    indices = edgeflow[:, :2].astype(np.int).transpose()

    # rewrap as tuple in order to do indexing. (Numpy does not work as expected here for ND array indices.)
    indices = (indices[0], indices[1])
    values = edgeflow[:, 2:]
    dense_flow[indices] = values
    # some visualization code port from stackoverflow.
    hsv = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    magnitude, angle = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("colored flow", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_image(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_dense_motion(dense_flow):
    """
    Args:
        dense_flow[array]: shape (h,w,2)
        image_shape[list]: of (h, w)
    """
    # some visualization code port from stackoverflow.
    hsv = np.zeros(
        (dense_flow.shape[0], dense_flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    magnitude, angle = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("colored flow", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_on_image(image, motion, color):
    indices = motion[:, :2].astype(np.int).transpose()
    indices = (indices[0], indices[1])
    image[indices] = color


def visualize_separated_motion(obstruction_motion, background_motion, image_shape):
    """
    Visualize the two motions.
    """
    image = np.zeros((image_shape[0], image_shape[1], 3))
    draw_on_image(image, obstruction_motion, [255, 0, 0])
    draw_on_image(image, background_motion, [0, 255, 0])
    cv2.imshow("separated motion", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
