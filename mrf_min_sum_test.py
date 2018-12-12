import numpy as np
import cv2
from scipy import signal

def get_cost_by_motion(I1, I2, point1, patch_size, max_motion_x, max_motion_y):
    """
    Args:
        I1, I2: gray input images
        point1: (x, y), point position in I1
        patch_size: to calculate the datacost
        max_motion_x, max_motion_y: motion range within [-max_motion_x+1, max_motion_x)
    Outputs:
        motion_cost: [max_motion_x*2-1, max_motion_y*2-1], motion cost for one point
    """
    
    height, width = I1.shape
    motion_cost = np.zeros((max_motion_x*2-1, max_motion_y*2-1))
    max_shift_x = max_motion_x - 1 + patch_size // 2
    max_shift_y = max_motion_y - 1 + patch_size // 2
    # ignore points near margins
    if point1[0] - max_shift_x <= 0 or point1[1] - max_shift_y <= 0 \
            or point1[0] + max_shift_x >= height or point1[1] + max_shift_y >= width:
                return motion_cost

    # using correlate2d to calculate ncc parallelly
    patch1 = I1[point1[0] - patch_size//2:point1[0] + patch_size//2+1, point1[1]-patch_size//2:point1[1] + patch_size//2+1]
    patch2 = I2[point1[0] - max_shift_x:point1[0]+max_shift_x+1, point1[1]-max_shift_y:point1[1]+max_shift_y+1]
    corrs = signal.correlate2d(patch2, patch1)
    sqrt1 = np.sqrt((patch1 * patch1).sum())
    sqrt2 = np.sqrt(signal.correlate2d(patch2 * patch2, np.ones((patch_size, patch_size))))
    motion_cost_with_outliners = 1. - corrs / (sqrt1 * sqrt2)
    motion_cost = motion_cost_with_outliners[patch_size-1:1-patch_size, patch_size-1:1-patch_size]
    return motion_cost

def get_belief(motion_fields, edge_points1, edge_points1_map, max_motion_x, max_motion_y):
    """
    Args:
        motion_fields: [point_number, max_motion_x*2-1, max_motion_y*2-1], motion cost for all points
        edge_points1: edge pixels positions
        edge_points1_map: (key: edge pixels positions, value: point id)
        max_motion_x, max_motion_y: motion range within [-max_motion_x+1, max_motion_x)
        message_passing_rounds: iteration number to pass message
    Outputs:
        final_motion_fields: [point_number, 2], final motion decisions by taking the minimum
    """
    
    point_number = len(edge_points1)
    final_motion_fields = np.zeros((point_number, 2))
    for point_id in range(point_number):
        point1 = edge_points1[point_id]
        best_motion = (0, 0)
        best_motion_belief = float("inf")
        for motion_x in range(-max_motion_x+1, max_motion_x):
            for motion_y in range(-max_motion_y+1,max_motion_y):
                data_cost = motion_fields[point_id][motion_x + max_motion_x-1][motion_y + max_motion_y - 1]
                belief = data_cost
                if belief < best_motion_belief:
                    best_motion_belief = belief
                    best_motion = (motion_x, motion_y)
        final_motion_fields[point_id] = best_motion
    return final_motion_fields

def produce_motion_fields(I1, I2, edge_image1, patch_size, max_motion_x, max_motion_y, message_passing_rounds=10):
    """
    Args: 
        I1, I2: input images
        edge_image1: edge image of I1 using canny
        patch_size: to calculate the datacost
        max_motion_x, max_motion_y: motion range within [-max_motion_x+1, max_motion_x)
        message_passing_rounds: iteration number to pass message
    Outputs:
        final_motion_fields: [pixel_number, 2], final motion estimations
        edge_points1: intermediate results, edge pixels positions
    """
    edge_points1 = np.argwhere(edge_image1)
    point_number = len(edge_points1)
    motion_fields = np.zeros((point_number, max_motion_x*2-1, max_motion_y*2-1))

    edge_points1_map = {}
    for point_id in range(point_number):
        motion_fields[point_id, :, :] = get_cost_by_motion(I1, I2, edge_points1[point_id], patch_size, max_motion_x, max_motion_y)
        edge_points1_map[tuple(edge_points1[point_id])] = point_id
    
    final_motion_fields = get_belief(motion_fields, edge_points1, edge_points1_map, max_motion_x, max_motion_y)
    return final_motion_fields, edge_points1

def test():
    edgeI1 = cv2.imread('./test_image/edge_dorm_0.png',0)
    I1 = cv2.imread('./test_image/dorm1_0.png', 0) / 255.
    I2 = cv2.imread('./test_image/dorm1_1.png', 0) / 255.

    height, width = edgeI1.shape
    height, width = height//2, width//2
    edgeI1 = cv2.resize(edgeI1, (width, height))
    I2 = cv2.resize(I2, (width, height))
    I1 = cv2.resize(I1, (width, height))
    patch_size = 7
    max_motion_x = 30
    max_motion_y = 30
    message_passing_rounds = 10

    final_motion_fields, edge_points1 = produce_motion_fields(I1, I2, edgeI1, patch_size, max_motion_x, max_motion_y, message_passing_rounds)

    np.save('motion_fields_test.npy', np.array(final_motion_fields))
    flow = np.zeros((height, width, 2))
    for point_id, motion_field in enumerate(final_motion_fields):
        point_pos = edge_points1[point_id]
        max_shift_x = max_motion_x - 1 + patch_size // 2
        max_shift_y = max_motion_y - 1 + patch_size // 2
        if point_pos[0] - max_shift_x <= 0 or point_pos[1] - max_shift_y <= 0 \
            or point_pos[0] + max_shift_x >= height or point_pos[1] + max_shift_y >= width:
                continue
        flow[point_pos[0], point_pos[1], :] = final_motion_fields[point_id, :]

    # Visualize Motion Fields.
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite('motion_fields_test.png', bgr)
