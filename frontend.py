import argparse
import cv2
import numpy as np
import glob
import logging
import os

from initialization import initialize_motion_based_decomposition
from optimization import optimize_motion_based_decomposition, OptimizationParams

#from generate_fake_initialization import generate_fake_initialization

def read_images(image_dir):
    """
    read images in grayscale.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
              for image_path in image_paths]
    #for test 1/4 images
    images = [cv2.resize(image, (288, 162)) for image in images]
    return images


def motion_based_decomposition(image_dir, otype='r', cached=False):
    """
    Do motion based decomposition.
    Args:
        image_dir[list(str)]: path to images.
        otype: 'r', reflection; 'o', obstruction
    """

    logging.info("reading images.")
    images = read_images(image_dir)

    logging.info("initializing values.")
    It, I_O_init, I_B_init, A_init, Vt_O_init, Vt_B_init\
        = initialize_motion_based_decomposition(images, otype, cached)

    optimization_params = OptimizationParams(
        #scales=[1./8, 1./4, 1./2, 1],
        #num_iterations_by_scale=[4, 1, 1, 1])
        scales=[1],
        num_iterations_by_scale=[8])

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


def motion_based_decomposition_from_gt(image_dir):
    """
    Do motion based decomposition.
    Args:
        image_dir[list(str)]: path to images.
    """

    logging.info("reading images.")
    bg_images = read_images(os.path.join(image_dir, "bg"))
    input_images = read_images(os.path.join(image_dir, "input"))
    ob_images = read_images(os.path.join(image_dir, "rf"))

    logging.info("fake initializing values.")
    It, I_O_init, I_B_init, A_init, Vt_O_init, Vt_B_init\
        = generate_fake_initialization(input_images, bg_images, ob_images)

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
    motion_based_decomposition(args.image_dir, otype='r', cached=args.cached)


if __name__ == "__main__":
    main()
