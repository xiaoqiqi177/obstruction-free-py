from util import scale_image, scale_images

import numpy as np
import tensorflow as tf

from warp import tf_warp
from visualize import visualize_dense_motion, visualize_image

LAMBDA_1 = 1.
LAMBDA_2 = 0.1
LAMBDA_3 = 3000.
LAMBDA_4 = 0.5
LAMBDA_p = 1e5


class OptimizationParams(object):
    def __init__(self,
                 scales,
                 num_iterations_by_scale):
        self.scales = scales  # list(float): scale used for upsampling.
        # list(int): iteration for optimization for each scale
        self.num_iterations_by_scale = num_iterations_by_scale


def l1_norm(tensor):
    return tf.norm(tensor, ord=1)


def constraint_penalty(tensor):
    """
    constrain tensor to lie within [0,1]
    """
    return tf.reduce_mean(LAMBDA_p * tf.minimum(np.float32(0), tensor)**2 +
                          LAMBDA_p * (tf.maximum(np.float32(1), tensor) - 1)**2)


def spatial_gradient(tensor):
    """
    tensor of shape NHWC.
    """
    return tf.concat(tf.image.image_gradients(tensor), axis=-1)


def decompose(It, Vt_O, Vt_B, I_O_init, I_B_init, A_init):
    tf.reset_default_graph()
    It = tf.constant(It, tf.float32)
    Vt_O = tf.constant(Vt_O, tf.float32)
    Vt_B = tf.constant(Vt_B, tf.float32)

    I_O = tf.Variable(I_O_init, name='I_O', dtype=tf.float32)
    I_B = tf.Variable(I_B_init, name='I_B', dtype=tf.float32)
    A = tf.Variable(A_init, name='A', dtype=tf.float32)

    warp_I_O = tf_warp(It, Vt_O)
    warp_A = tf_warp(tf.tile(tf.expand_dims(A, 0), [5, 1, 1, 1]), Vt_O)
    warp_I_B = tf_warp(tf.tile(tf.expand_dims(I_B, 0), [5, 1, 1, 1]), Vt_B)

    g_O = spatial_gradient(tf.expand_dims(I_O, 0))
    g_B = spatial_gradient(tf.expand_dims(I_B, 0))

    residual = l1_norm(It - warp_I_O - tf.multiply(warp_A, warp_I_B))
    loss = l1_norm(residual)
    loss += LAMBDA_1 * \
        tf.norm(spatial_gradient(tf.expand_dims(A, 0)), ord=2)**2
    loss += LAMBDA_2 * (l1_norm(g_O) +
                        l1_norm(g_B))
    loss += LAMBDA_3 * tf.norm(g_O*g_O*g_B*g_B, ord=2)**2

    loss += constraint_penalty(I_O) + \
        constraint_penalty(I_B) + constraint_penalty(A)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-6)
    train = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for step in range(1000):
            _, loss_val = session.run([train, loss])
            print("step {}:loss = {}".format(step, loss_val))
        I_O, I_B, A = session.run([I_O, I_B, A])
        """
        visualize_image(I_O)
        visualize_image(I_B)
        visualize_image(A)
        """
    return I_O, I_B, A


def estimate_motion(It, I_O, I_B, A, Vt_O_init, Vt_B_init):
    tf.reset_default_graph()
    It = tf.constant(It, tf.float32)
    I_O = tf.constant(I_O, tf.float32)
    I_B = tf.constant(I_B, tf.float32)
    A = tf.constant(A, tf.float32)

    Vt_O = tf.Variable(Vt_O_init, name='Vt_O', dtype=tf.float32)
    Vt_B = tf.Variable(Vt_B_init, name='Vt_B', dtype=tf.float32)
    warp_I_O = tf_warp(It, Vt_O)
    warp_A = tf_warp(tf.tile(tf.expand_dims(A, 0), [5, 1, 1, 1]), Vt_O)
    warp_I_B = tf_warp(tf.tile(tf.expand_dims(I_B, 0), [5, 1, 1, 1]), Vt_B)

    residual = It - warp_I_O - tf.multiply(warp_A, warp_I_B)

    loss = l1_norm(residual)
    loss += LAMBDA_4 * (l1_norm(spatial_gradient(Vt_O)) +
                        l1_norm(spatial_gradient(Vt_B)))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for step in range(500):
            _, loss_val = session.run([train, loss])
            print("step {}:loss = {}".format(step, loss_val))
        Vt_O, Vt_B = session.run([Vt_O, Vt_B])
        """
        for i in range(5):
            visualize_dense_motion(Vt_O[i])
            visualize_dense_motion(Vt_B[i])
        """
    return Vt_O, Vt_B


def optimize_motion_based_decomposition(It, I_O_init, I_B_init, A_init, Vt_O_init, Vt_B_init, params):
    """
    Optimize motion based decomposition problem.

    Args:
        It[list(Image)]: input image
        I_O_init[Image]: init of image of obstruction
        I_B_init[Image]: init of image of background
        A_init[Image]: init of image of occlusion mask
        Vt_O_init[Image]: init of dense motion field of obstruction
        Vt_B_init[Image]: init of dense motion field of background
        params[OptimizationParams]: params for the optimization
    """

    original_shape = It.shape[1:3]
    previous_shape = original_shape

    # initialize all values
    I_O = I_O_init
    I_B = I_B_init
    A = A_init
    Vt_O = Vt_O_init
    Vt_B = Vt_B_init

    for current_scale, num_iterations in zip(params.scales, params.num_iterations_by_scale):

        current_shape = (int(original_shape[0] * current_scale), \
                int(original_shape[1] * current_scale))
        # Scale values to proper scale.
        It_scaled = scale_images(
            It, from_shape=original_shape, to_shape=current_shape)

        I_O = scale_image(I_O, from_shape=previous_shape,
                          to_shape=current_shape)
        I_B = scale_image(I_B, from_shape=previous_shape,
                          to_shape=current_shape)
        A = scale_image(A, from_shape=previous_shape, to_shape=current_shape)
        Vt_O = scale_images(Vt_O, from_shape=previous_shape,
                            to_shape=current_shape)
        Vt_B = scale_images(Vt_B, from_shape=previous_shape,
                            to_shape=current_shape)
        """
        visualize_image(I_O)
        visualize_image(I_B)
        visualize_image(A)
        """
        for _ in range(num_iterations):
            Vt_O, Vt_B = estimate_motion(It_scaled, I_O, I_B, A, Vt_O, Vt_B)
            I_O, I_B, A = decompose(
                It_scaled, Vt_O, Vt_B, I_O, I_B, A)

            previous_shape = current_shape

    # TODO: check return value
    return
