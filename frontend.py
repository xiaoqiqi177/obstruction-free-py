import argparse
import cv2
import glob
import logging
import os

from initialization import initialize_motion_based_decomposition
from optimization import optimize_motion_based_decomposition, OptimizationParams


def read_images(image_dir):
    """
    read images in grayscale.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
              for image_path in image_paths]
    return images


def motion_based_decomposition(image_dir, cached):
    """
    Do motion based decomposition.
    Args:
        image_dir[list(str)]: path to images.
    """

    logging.info("reading images.")
    images = read_images(image_dir)

    logging.info("initializing values.")
    It, I_O_init, I_B_init, A_init, Vt_O_init, Vt_B_init\
        = initialize_motion_based_decomposition(images, cached)

    optimization_params = OptimizationParams(
        scales=[1./8, 1./4, 1./2, 1],
        num_iterations_by_scale=[4, 1, 1, 1])

    logging.info("optimizing values.")
    optimize_motion_based_decomposition(
        It=It,
        I_O_init=I_O_init,
        I_B_init=I_B_init,
        A_init=A_init,
        Vt_O_init=Vt_O_init,
        Vt_B_init=Vt_B_init,
        params=optimization_params)

    # TODO: add visualization and save output.

    logging.info("DONE")


def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir",
                        help="path to the image directory")
    parser.add_argument("--cached",
                        action="store_true",
                        help="use cached value when possible.")
    args = parser.parse_args()
    motion_based_decomposition(args.image_dir, args.cached)


if __name__ == "__main__":
    main()
