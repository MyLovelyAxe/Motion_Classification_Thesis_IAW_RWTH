import numpy as np

########################################################
###### Feature: distances between random 2 joints ######
########################################################

def get_joint_index():
    """
    the index of each joint from original data
    """
    joint_index_dict = {'LWrist':0, 'LElbow':1, 'LShoulder':2,
                       'RWrist':3, 'RElbow':4, 'RShoulder':5,
                       'LToe':6, 'LAnkle':7, 'LKnee':8, 'LHip':9,
                       'RToe':10, 'RAnkle':11, 'RKnee':12, 'RHip':13,
                       'LClavicle':14, 'LHandEnd':15, 'LToesEnd':16,
                       'RClavicle':17, 'RHandEnd':18, 'RToesEnd':19,
                       'spine1':20, 'spine2':21, 'spine3':22, 'spine4':23, 'spine5':24, 'head':25
                      }
    return joint_index_dict

def calc_all_distances(coords):
    """
    calculate all distance between random 2 joints for further slicing
    """
    # coords: [frames,xyz,num_joints] = [18000,3,26]
    num_frame, num_joints = coords[:,0,:].shape
    all_distances = np.zeros((num_frame,num_joints,num_joints))
    new_coords = np.expand_dims(coords, axis=-2) # shape [18000,3,26] -> [18000,3,1,26]
    # calculate dis_feature:
    #     diff = coords - transpose(coords,(0,1,3,2)):
    #       meaning: x-x.T, y-y.T, z-z.T
    #       shape: [18000,3,1,26] - [18000,3,26,1] = [18000,3,26,26]
    #     sum(diff**2,axis=1):
    #       meaning: (x-x.T)**2 + (y-y.T)**2 + (z-z.T)**2
    #       shape: [18000,3,26,26] -> [18000,26,26]
    all_distances = np.sqrt(np.sum((new_coords - np.transpose(new_coords,(0,1,3,2)))**2,axis=1))
    all_distances = np.round(all_distances.astype(np.float64),decimals=5)
    # e,g, dist_feature[3,1,4] means distance between LElbow and RElbow in 3rd frame
    return all_distances

def examine_distance(coords,frame,joint_1,joint_2):
    """
    examine whether the calculated distance from np.array operation is equal to elementwise calculation
    """
    # coords: [frames,xyz,num_joints] = [18000,3,26]
    all_distances = calc_all_distances(coords)
    joint_index_dict = get_joint_index()
    joint_1_idx = joint_index_dict[joint_1]
    joint_2_idx = joint_index_dict[joint_2]
    # due to unclear reason, np.round doesn't work on type np.float32, but only on np.float64
    # and values inside all_distances are defaultly calculated as np.float32
    dist_from_array = all_distances[frame,joint_1_idx,joint_2_idx]
    # dist_from_array = np.round(all_distances[frame,joint_1_idx,joint_2_idx].astype(np.float64),decimals=5)
    dist_calculated = np.round(np.sqrt((coords[frame,0,joint_1_idx] - coords[frame,0,joint_2_idx])**2
                                       +(coords[frame,1,joint_1_idx] - coords[frame,1,joint_2_idx])**2
                                       +(coords[frame,2,joint_1_idx] - coords[frame,2,joint_2_idx])**2
                                       ),decimals=5)
    print(f'distance indexed from distances array: {dist_from_array}')
    print(f'distance calculated from coordinates: {dist_calculated}')
    print(f'wehther the same: {dist_from_array == dist_calculated}')

def get_dist_feature(coords,desired_dists):
    """
    extract desired distance features from all distances
    """
    all_distances = calc_all_distances(coords)
    joint_index_dict = get_joint_index()
    dist_feature = np.zeros((all_distances.shape[0],len(desired_dists)))
    for idx,desired_dist in enumerate(desired_dists):
        joint1,joint2 = desired_dist.split('_')
        dist_feature[:,idx] = all_distances[:,joint_index_dict[joint1],joint_index_dict[joint2]]
    del all_distances
    return dist_feature

####################################################
###### Feature: angles between adjacent links ######
####################################################

def get_angle_pairs(desized_angles):
    """
    get index of corresponding index of joints based on desired angles
    """
    angle_pairs_dict = {}
    all_angle_pairs_dict = {'LHandEnd_LWrist_LElbow':[15,0,1],
                        'LWrist_LElbow_LShoulder':[0,1,2],
                        'LElbow_LShoulder_LClavicle':[1,2,14],
                        'LShoulder_LClavicle_spine5':[2,14,24],
                        'LClavicle_spine5_spine4':[14,24,23],
                        'LClavicle_spine5_head':[14,24,25],
                        'RHandEnd_RWrist_RElbow':[18,3,4],
                        'RWrist_RElbow_RShoulder':[3,4,5],
                        'RElbow_RShoulder_RClavicle':[4,5,17],
                        'RShoulder_RClavicle_spine5':[5,17,24],
                        'RClavicle_spine5_spine4':[17,24,23],
                        'RClavicle_spine5_head':[17,24,25],
                        'LClavicle_spine5_RClavicle':[14,24,17],
                        'LToesEnd_LToe_LAnkle':[16,6,7],
                        'LToe_LAnkle_LKnee':[6,7,8],
                        'LAnkle_LKnee_LHip':[7,8,9],
                        'LKnee_LHip_spine1':[8,9,20],
                        'RToesEnd_RToe_RAnkle':[19,10,11],
                        'RToe_RAnkle_RKnee':[10,11,12],
                        'RAnkle_RKnee_RHip':[11,12,13],
                        'RKnee_RHip_spine1':[12,13,20],
                        'LHip_spine1_RHip':[9,20,13],
                        'LHip_spine1_spine2':[9,20,21],
                        'RHip_spine1_spine2':[13,20,21],
                        'spine1_spine2_spine3':[20,21,22],
                        'spine2_spine3_spine4':[21,22,23],
                        'spine3_spine4_spine5':[22,23,24],
                        'spine4_spine5_head':[23,24,25]
                       }
    for desired_angle in desized_angles:
        if not all_angle_pairs_dict.get(desired_angle) is None:
            angle_pairs_dict[desired_angle] = all_angle_pairs_dict[desired_angle]
    return angle_pairs_dict

def calc_angles(joints_lst,distances):
    """
    calculate angles:
        b: the edge in triangle ABC which is opposite to the angle to be calculated
        a,c: side edges
        refer to formular: 
        https://www.mathsisfun.com/algebra/trig-cosine-law.html
    """
    a = distances[joints_lst[0],joints_lst[1]]
    b = distances[joints_lst[0],joints_lst[2]]
    c = distances[joints_lst[1],joints_lst[2]]
    cos_b = (c**2 + a**2 - b**2) / (2*c*a)
    # usually only when B slightly exceeds 180°, cos_b would be less than -1
    if cos_b < -1:
        B = np.pi
    else:
        B = np.arccos(cos_b) # radius = np.arccos(cos_value)
    return np.round(B,decimals=5)

def get_angle_feature(coords,desized_angles):
    """
    get desired angle features
    """
    all_distances = calc_all_distances(coords)
    angle_pairs_dict = get_angle_pairs(desized_angles)
    num_frames = all_distances.shape[0]
    num_angles = len(angle_pairs_dict)
    angle_feature = np.zeros((num_frames,num_angles))
    for frame, distances in enumerate(all_distances):
        for angle_idx, (_,joints_lst) in enumerate(angle_pairs_dict.items()):
            angle_feature[frame,angle_idx] = calc_angles(joints_lst,distances)
    return angle_feature

def get_all_features(coords,desired_dists,desized_angles):
    """
    concatenate all features
    """
    dist_feature = get_dist_feature(coords,desired_dists)
    angle_feature = get_angle_feature(coords,desized_angles)
    all_features = np.concatenate((dist_feature, angle_feature), axis=1)
    return all_features
