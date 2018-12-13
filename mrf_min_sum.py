from multiprocessing import Pool
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

def get_penalty_matrix(max_motion_x, max_motion_y):
    """
    Args:
        max_motion_x, max_motion_y: motion range within [-max_motion_x+1, max_motion_x)
    Outputs:
        penalty_template_matrix: [max_motion_x*2-1, max_motion_y*2-1, max_motion_x*2-1, max_motion_], 
            to care with cases with small distances
    """
    
    penalty_template_matrix = np.ones((max_motion_x*2-1, max_motion_y*2-1, max_motion_x*2-1, max_motion_y*2-1))*float("inf")
    for motion_x1 in range(-max_motion_x+1, max_motion_x):
        for motion_y1 in range(-max_motion_y+1, max_motion_y):
            for motion_x2 in range(-max_motion_x+1, max_motion_x):
                for motion_y2 in range(-max_motion_y+1, max_motion_y):
                    if abs(motion_x1 - motion_x2) + abs(motion_y1 - motion_y2) == 1:
                        penalty_template_matrix[motion_x1+max_motion_x-1, motion_y1+max_motion_y-1, motion_x2+max_motion_x-1, motion_y2+max_motion_y-1] = 1.
                    if abs(motion_x1 - motion_x2) + abs(motion_y1 - motion_y2) == 0:
                        penalty_template_matrix[motion_x1+max_motion_x-1, motion_y1+max_motion_y-1, motion_x2+max_motion_x-1, motion_y2+max_motion_y-1] = 0.
    return penalty_template_matrix

def min_sum(self_motion_fields, neighbor_messages, w12, penalty_template_matrix, max_motion_x, max_motion_y):
    """
    Args:
        self_motion_fields: data cost
        neighbor_messages: messages from neighbors
        w12: penalty weight
        penalty_template_matrix: [max_motion_x*2-1, max_motion_y*2-1, max_motion_x*2-1, max_motion_], 
            to care with cases with small distances
        max_motion_x, max_motion_y: motion range within [-max_motion_x+1, max_motion_x)
    Outputs:
        message_matrix: [max_motion_x*2-1, max_motion_y*2-1], new message to pass
    """
    message_matrix = np.zeros((max_motion_x*2-1, max_motion_y*2-1))
    neighbor_contribution = np.zeros((max_motion_x*2-1, max_motion_y*2-1))
    for neighbor_message in neighbor_messages:
        neighbor_contribution += neighbor_message

    small_weight = 0.005
    real_penalty_matrix = small_weight * penalty_template_matrix
    infindex = np.isinf(real_penalty_matrix)
    real_penalty_matrix[infindex] = w12 * np.ones_like(penalty_template_matrix)[infindex]

    possible_vals_matrix = self_motion_fields[None, None, :, :] + real_penalty_matrix + neighbor_contribution[None, None, :, :]
    message_matrix = np.min(possible_vals_matrix, (2, 3))
    
    # normalization step
    message_matrix = message_matrix - np.log(np.exp(message_matrix).sum())
    return message_matrix

def calculate_w12(I1, point1, point_new):
    """
    Args:
        I1: input image
        point1: current point
        point_new: neighboring point

    Outputs:
        w12: penalty weight
    """
    
    epsilon = 0.075
    g1 = np.array([I1[point1[0]][point1[1]] - I1[max(point1[0]-1, 0)][point1[1]], I1[point1[0]][point1[1]] - I1[point1[0]][max(point1[1]-1, 0)]])
    g2 = np.array([I1[point_new[0]][point_new[1]] - I1[max(point_new[0]-1, 0)][point_new[1]], I1[point_new[0]][point_new[1]] - I1[point_new[0]][max(point_new[1]-1, 0)]])
    g1_norm = np.sqrt((g1*g1).sum())
    g2_norm = np.sqrt((g2*g2).sum())
    if g1_norm < epsilon and g2_norm < epsilon:
        w12 = 0.2
    elif g1_norm > epsilon and g2_norm > epsilon:
        w12 = (g1*g2).sum() / (g1_norm * g2_norm)
    else:
        w12 = 0.005
    return w12

directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
def pass_message(I1, motion_fields, penalty_template_matrix, edge_points1, edge_points1_map, message_passing_rounds, max_motion_x, max_motion_y):
    """
    Args:
        I1: input image
        motion_fields: [point_number, max_motion_x*2-1, max_motion_y*2-1], motion cost for all points
        penalty_template_matrix: [max_motion_x*2-1, max_motion_y*2-1, max_motion_x*2-1, max_motion_], 
            to care with cases with small distances
        edge_points1: edge pixels positions
        edge_points1_map: (key: edge pixels positions, value: point id)
        max_motion_x, max_motion_y: motion range within [-max_motion_x+1, max_motion_x)
        message_passing_rounds: iteration number to pass message
    Outputs:
        message_map: [point_number, 4, max_motion_x*2-1, max_motion_y*2-1], messages from 4 directions
    """
    
    point_number = len(edge_points1)
    message_map = np.zeros((point_number, 4, max_motion_x*2-1, max_motion_y*2-1))
    next_message_map = np.zeros((point_number, 4, max_motion_x*2-1, max_motion_y*2-1))
    
    # precalculate w12 map
    w12_map = {}
    for point_id in range(point_number):
        point1 = edge_points1[point_id]
        for d in range(4):
            direction = directions[d]
            point_new = (point1[0]+direction[0], point1[1]+direction[1])
            if point_new in edge_points1_map:
                w12 = calculate_w12(I1, point1, point_new)
                w12_map[(point_id, edge_points1_map[point_new])] = w12

    for m_round in range(message_passing_rounds):
        print('m_round: ', m_round)
        for d in range(4):
            direction = directions[d]
            for point_id in range(point_number):
                print(point_id)
                point1 = edge_points1[point_id]
                # exclude the message from the neighbor we are passing to
                point_new = (point1[0]+direction[0], point1[1]+direction[1])
                if point_new in edge_points1_map:
                    neighbor_messages = []
                    for i in range(4):
                        if i != d:
                            point_new_tmp = (point1[0]+directions[i][0], point1[1]+directions[i][1])
                            if point_new_tmp in edge_points1_map:
                                neighbor_messages.append(message_map[point_id, i])
                    # populate the new message map with the next round's messages
                    w12 = w12_map[(point_id, edge_points1_map[point_new])]
                    next_message_map[edge_points1_map[point_new], 3-d] = min_sum(motion_fields[point_id], neighbor_messages, w12, penalty_template_matrix, max_motion_x, max_motion_y)
        # update message map
        message_map = next_message_map
    return message_map

def get_belief(motion_fields, message_map, edge_points1, edge_points1_map, max_motion_x, max_motion_y):
    """
    Args:
        motion_fields: [point_number, max_motion_x*2-1, max_motion_y*2-1], motion cost for all points
        message_map: [point_number, 4, max_motion_x*2-1, max_motion_y*2-1], messages from 4 directions
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
                neighbor_belief = 0.
                neighbors = message_map[point_id]
                for i in range(4):
                    point_new_tmp = (point1[0]+directions[i][0], point1[1]+directions[i][1])
                    if point_new_tmp in edge_points1_map:
                        neighbor_belief += neighbors[i][motion_x+max_motion_x-1][motion_y+max_motion_y-1]
                belief = data_cost + neighbor_belief
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
    
    penalty_template_matrix = get_penalty_matrix(max_motion_x, max_motion_y)

    message_map = pass_message(I1, motion_fields, penalty_template_matrix, edge_points1, edge_points1_map, message_passing_rounds, max_motion_x,max_motion_y)
    final_motion_fields = get_belief(motion_fields, message_map, edge_points1, edge_points1_map, max_motion_x, max_motion_y)
    return final_motion_fields, edge_points1

def test():
    cur_frame = 0
    edgeI1 = cv2.imread('./test_image/edge_dorm_' + str(cur_frame) + '.png',0)
    I1 = cv2.imread('./test_image/dorm1_' + str(cur_frame) + '.png', 0) / 255.
    I2 = cv2.imread('./test_image/dorm1_2.png', 0) / 255.

    height, width = edgeI1.shape
    height, width = height, width
    edgeI1 = cv2.resize(edgeI1, (width, height))
    I2 = cv2.resize(I2, (width, height))
    I1 = cv2.resize(I1, (width, height))
    patch_size = 5
    max_motion_x = 30
    max_motion_y = 30
    message_passing_rounds = 15

    motion_fields, edge_points_before = produce_motion_fields(I1, I2, edgeI1, patch_size, max_motion_x, max_motion_y, message_passing_rounds)
    edgeflow = []
    for point_id, motion_field in enumerate(motion_fields):
        point_pos = edge_points_before[point_id]
        max_shift_x = max_motion_x - 1 + patch_size // 2
        max_shift_y = max_motion_y - 1 + patch_size // 2
        if point_pos[0] - max_shift_x <= 0 or point_pos[1] - max_shift_y <= 0 \
                or point_pos[0] + max_shift_x >= height or point_pos[1] + max_shift_y >= width:
            continue
        edgeflow.append([point_pos[0], point_pos[1], motion_field[0], motion_field[1]])
    edgeflow = np.array(edgeflow)

    np.save('./edgeflow_dorm/motion_fields_' + str(cur_frame) + '.npy', edgeflow)

    # Visualize Motion Fields.
    mag, ang = cv2.cartToPolar(edgeflow[...,2], edgeflow[...,3])
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite('./edgeflow_dorm/motion_fields_' + str(cur_frame) + '.png', bgr)

test()
