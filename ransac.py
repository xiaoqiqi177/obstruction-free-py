import numpy as np
from tqdm import tqdm
import logging

# The whole file is deprecated.


class RANSACModel(object):

    def __init__(self, data):
        """
        fit data to generate a model.
        """
        raise NotImplementedError

    def get_error(self, data_points):
        """
        calculate error w.r.t. data_points.
        """
        raise NotImplementedError


def ransac(data, Model, n, k, t, d):

    best_model = None
    best_error = float("inf")
    best_inlier_idxs = None
    for _ in tqdm(range(k)):
        maybe_inlier_idxs = set(np.random.choice(
            range(len(data)), size=n, replace=False))
        maybe_inliers = [data[idx] for idx in maybe_inlier_idxs]
        maybe_model = Model(maybe_inliers)
        also_inlier_idxs = set()
        for idx, data_point in enumerate(data):
            if idx in maybe_inlier_idxs:
                continue
            if maybe_model.get_error([data_point]) < t:
                also_inlier_idxs.add(idx)
        if len(also_inlier_idxs) > d:
            # A good model, test how good it is.
            inlier_idxs = maybe_inlier_idxs.union(also_inlier_idxs)
            inlier_datapoints = [data_point for idx, data_point in enumerate(data)
                                 if idx in inlier_idxs]
            better_model = Model(inlier_datapoints)
            error = better_model.get_error(inlier_datapoints)
            logging.info("Error={}".format(error))
            if error < best_error:
                best_model = better_model
                best_error = error
                best_inlier_idxs = inlier_idxs
    return best_model, best_error, best_inlier_idxs
