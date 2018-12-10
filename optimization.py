from util import scale_image


class OptimizationParams(object):
    def __init__(self,
                 scales,
                 num_iterations_by_scale):
        self.scales = scales  # list(float): scale used for upsampling.
        # list(int): iteration for optimization for each scale
        self.num_iterations_by_scale = num_iterations_by_scale


def decompose(It, Vt_O, Vt_B):
    raise NotImplementedError


def estimate_motion(It, I_O, I_B, A):
    raise NotImplementedError


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

    original_scale = 1.
    previous_scale = original_scale

    # initialize all values
    I_O = I_O_init
    I_B = I_B_init
    A = A_init
    Vt_O = Vt_O_init
    Vt_B = Vt_B_init

    for current_scale, num_iterations in zip(params.scales, params.num_iterations_by_scale):

        # Scale values to proper scale.
        It_scaled = [scale_image(
            It, from_scale=original_scale, to_scale=current_scale)]
        I_O = scale_image(I_O, from_scale=previous_scale,
                          to_scale=current_scale)
        I_B = scale_image(I_B, from_scale=previous_scale,
                          to_scale=current_scale)
        A = scale_image(A, from_scale=previous_scale, to_scale=current_scale)
        Vt_O = [scale_image(Vt_O, from_scale=previous_scale,
                            to_scale=current_scale)]
        Vt_B = [scale_image(Vt_B, from_scale=previous_scale,
                            to_scale=current_scale)]

        for _ in range(num_iterations):
            I_O, I_B, A = decompose(It_scaled, Vt_O, Vt_B)
            Vt_O, Vt_B = estimate_motion(It_scaled, I_O, I_B, A)

        previous_scale = current_scale

    # TODO: check return value
    return
