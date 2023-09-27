import re
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import yaml

########################################################
###### Feature: distances between random 2 joints ######
########################################################

def get_joint_index():
    """
    the index of each joint from original data
    """
    joint_index_dict = {
        'LWrist':0, 'LElbow':1, 'LShoulder':2,
        'RWrist':3, 'RElbow':4, 'RShoulder':5,
        'LToe':6, 'LAnkle':7, 'LKnee':8, 'LHip':9,
        'RToe':10, 'RAnkle':11, 'RKnee':12, 'RHip':13,
        'LClavicle':14, 'LHandEnd':15, 'LToesEnd':16,
        'RClavicle':17, 'RHandEnd':18, 'RToesEnd':19,
        'spine1':20, 'spine2':21, 'spine3':22, 'spine4':23, 'spine5':24, 'head':25
        }
    return joint_index_dict

def get_scales_dict():
    """
    scales_dict:
        contains distance features for standarizing the whole distance-related features, i.e. distance, velocity
    """
    scales_dict = {
        'len_spine': ['spine5_spine4','spine4_spine3','spine3_spine2','spine2_spine1'],
        'neck_height': ['spine5_spine4','spine4_spine3','spine3_spine2','spine2_spine1','RHip_RKnee','RKnee_RAnkle'],
        'len_arm': ['LHandEnd_LWrist','LWrist_LElbow','LElbow_LShoulder','LShoulder_LClavicle','LClavicle_spine5','spine5_RClavicle','RClavicle_RShoulder','RShoulder_RElbow','RElbow_RWrist','RWrist_RHandEnd'],
        'len_shoulder': ['LShoulder_LClavicle','LClavicle_spine5','spine5_RClavicle','RClavicle_RShoulder'],
        }
    return scales_dict

def calc_all_distances(coords):
    """
    calculate all distance between random 2 joints for further slicing
    """
    # coords: [frames,xyz,num_joints] = [18000,3,26]
    new_coords = np.expand_dims(coords, axis=-2) # shape [18000,3,26] -> [18000,3,1,26]
    # calculate dis_feature:
    #     diff = coords - transpose(coords,(0,1,3,2)):
    #       meaning: x-x.T, y-y.T, z-z.T
    #       shape: [18000,3,1,26] - [18000,3,26,1] = [18000,3,26,26]
    #     sum(diff**2,axis=1):
    #       meaning: (x-x.T)**2 + (y-y.T)**2 + (z-z.T)**2
    #       shape: [18000,3,26,26] -> [18000,26,26]
    all_distances = np.sqrt(np.sum((new_coords - np.transpose(new_coords,(0,1,3,2)))**2,axis=1))
    all_distances = np.round(all_distances.astype(np.float32),decimals=5)
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

def get_dist_feature(all_distances,desired_dists):
    """
    extract desired distance features from all distances
    """
    joint_index_dict = get_joint_index()
    dist_feature = np.zeros((all_distances.shape[0],len(desired_dists)))
    for idx,desired_dist in enumerate(desired_dists):
        joint1,joint2 = desired_dist.split('_')
        dist_feature[:,idx] = all_distances[:,joint_index_dict[joint1],joint_index_dict[joint2]]
    return dist_feature

####################################################
###### Feature: angles between adjacent links ######
####################################################

def get_angle_pairs(desired_angles):
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
    for desired_angle in desired_angles:
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
        B = np.float32(np.pi)
    else:
        B = np.arccos(cos_b) # radius = np.arccos(cos_value)
    B = np.round(B.astype(np.float32),decimals=5)
    return B

def get_angle_feature(all_distances,desired_angles):
    """
    get desired angle features
    """
    angle_pairs_dict = get_angle_pairs(desired_angles)
    num_frames = all_distances.shape[0]
    num_angles = len(angle_pairs_dict)
    angle_feature = np.zeros((num_frames,num_angles))
    for frame, distances in enumerate(all_distances):
        for angle_idx, (_,joints_lst) in enumerate(angle_pairs_dict.items()):
            angle_feature[frame,angle_idx] = calc_angles(joints_lst,distances)
    return angle_feature

def calc_ChangeRate(features):

    diff = features[1:,:] - features[:-1,:] # diff: [#frame-1,#feature]
    first_diff = np.expand_dims(diff[0],axis=0) # first_diff: [1,#feature]
    diff = np.concatenate((first_diff,diff),axis=0) # diff: [#frame,#feature]
    frame_rate = 1/60
    change_rate = diff / frame_rate # change_rate: [#frame,#feature]

    return change_rate

def calc_height_rate(frame):
    """
    frame: skeleton data to calculate scaling values for standarization of distance-related features, i.e. distance, velocity
            which is from only one frame, i.e. with shape [3,26]
    """

    ### only one frame to calculate, because these features never change during the process
    frame = np.expand_dims(frame,axis=0) # frame: [3,26] -> [1,3,26]
    distances = calc_all_distances(frame) # all_distances: [1.26,26]

    scale_elements = {}
    scales_dict = get_scales_dict()
    for scale_name, scales_idx in scales_dict.items():
        scale_length = np.sum(get_dist_feature(distances,scales_idx))
        scale_elements[scale_name] = scale_length
    scale_elements['no_scale'] = 1.0

    del distances
    return scale_elements

def get_all_features(coords,desired_dists,desired_angles,standard):
    """
    concatenate all features
    """
    all_distances = calc_all_distances(coords)
    dist_feature = get_dist_feature(all_distances,desired_dists) # dist_feature: [#frames,#desired_dist]
    dist_rate = calc_ChangeRate(dist_feature) # dist_rate with same shape of dist_feature
    angle_feature = get_angle_feature(all_distances,desired_angles) # angle_feature: [#frames,#desired_angle]
    angle_rate = calc_ChangeRate(angle_feature) # angle_rate with same shape of angle_feature

    # standarize the distance-related features
    scale_elements = calc_height_rate(coords[0])
    scale_factor = scale_elements[standard]
    print(f'------------------------------------------------------------------')
    print(f'selected scaling factor is: {standard} = {scale_factor}')
    print(f'------------------------------------------------------------------')
    dist_feature = dist_feature / scale_factor
    dist_rate = dist_rate / scale_factor

    all_features = np.concatenate((dist_feature,
                                   dist_rate,
                                   angle_feature,
                                   angle_rate), axis=1)
    
    del all_distances
    return all_features

#################################################################
###### Dynamic features: statistic features inside windows ######
#################################################################

def calc_Mean(data) -> np.array:
    """
    shape:
        inpu: [#win,window_size,#features]
        output: [#win,#features]
    type:
        mean of data along dimension of window
    """
    Mean = np.mean(data,axis=1)
    if np.count_nonzero(np.isnan(Mean)):
        print(f'There is Nan in Mean')
    return Mean

def calc_Std(data) -> np.array:
    """
    shape:
        inpu: [#win,window_size,#features]
        output: [#win,#features]
    type:
        standard deviation of data along dimension of window
    """
    Std = np.std(data,axis=1)
    if np.count_nonzero(np.isnan(Std)):
        print(f'There is Nan in Std')
    return Std

def calc_TopMean_Range(data,num=5) -> np.array:
    """
    shape:
        inpu: [#win,window_size,#features]
        output: [#win,#features]
    type:
        return mean of the num-highest maximum and mean of num-lowest minimum
        and range between TopMaxMean and TopMinMean
    """
    assert num>1, 'calc_TopMaxMean(): num should be larger than 1'
    # sort along dimension of window
    sort_index = np.argsort(data, axis=1)
    sorted_data = np.take_along_axis(data,sort_index,axis=1)
    top_max = sorted_data[:,:num,:]
    top_min = sorted_data[:,-num:,:]
    # take mean of num-highest maximum and num-lowest minimum
    top_max_mean = np.mean(top_max,axis=1)
    top_min_mean = np.mean(top_min,axis=1)
    # take range
    max_min_range = top_max_mean - top_min_mean

    # check Nan
    if np.count_nonzero(np.isnan(top_max_mean)):
        print(f'There is Nan in top_max_mean')
    if np.count_nonzero(np.isnan(top_min_mean)):
        print(f'There is Nan in top_min_mean')
    if np.count_nonzero(np.isnan(max_min_range)):
        print(f'There is Nan in max_min_range')

    # return concatenated array
    return np.concatenate((top_max_mean,
                           top_min_mean,
                           max_min_range),axis=1)

def calc_Kurtosis(data) -> np.array:
    """
    shape:
        inpu: [#win,window_size,#features]
        output: [#win,#features]
    type:
        the concept of kurtosis:
        https://www.scribbr.com/statistics/kurtosis/#:~:text=Kurtosis%20is%20a%20measure%20of,(medium%20tails)%20are%20mesokurtic.
    """
    Kurtosis = kurtosis(data,axis=1)
    # in the case of classification of static movements, window_size is relatively small, which leads to that some of features
    # are almost or exactly the same within one window, which causes Kurtosis a NaN. Just set Kurtosis=0 in such cases.
    if np.count_nonzero(np.isnan(Kurtosis)):
        # print(f'These windows is Nan in Kurtosis: ')
        # print(f'{np.where(np.isnan(Kurtosis))}')
        Kurtosis = np.where(np.isnan(Kurtosis),0,Kurtosis)

    return Kurtosis

def calc_Skewness(data) -> np.array:
    """
    shape:
        inpu: [#win,window_size,#features]
        output: [#win,#features]
    type:
        the concept of skewness:
        https://www.scribbr.com/statistics/skewness/
    """
    Skewness = skew(data,axis=1)
    # in the case of classification of static movements, window_size is relatively small, which leads to that some of features
    # are almost or exactly the same within one window, which causes Skewness a NaN. Just set Skewness=0 in such cases.
    if np.count_nonzero(np.isnan(Skewness)):
        # print(f'These windows is Nan in Skewness: ')
        # print(f'{np.where(np.isnan(Skewness))}')
        Skewness = np.where(np.isnan(Skewness),0,Skewness)

    return Skewness

def dynamic_features(x_data):
    """
    shape:
        input x_data should have shape: [#win,window_size,#features]
        output x_data should have shape: [#win,#stat_features] = [#win,#features*#metrics]
    type:
        concatenation of different statistic features of features inside each window
    """
    dynamic_statistics = np.concatenate((calc_Mean(x_data),
                                         calc_Std(x_data),
                                         calc_TopMean_Range(x_data,num=30),
                                         calc_Kurtosis(x_data),
                                         calc_Skewness(x_data),
                                         ),axis=1)
    return dynamic_statistics

def get_act_index(split_method_paths,which_activity):
    act_idx_lst = []
    act_index_dict = get_act_index_dict(split_method_paths)
    print(f'act_index_dict: {act_index_dict}')
    for f in which_activity:
        act_idx_lst.append(act_index_dict[f])
    return act_idx_lst

def get_ActName(act_name):
    """
    Extract only letters from a string containing both letters and digits
    """
    return re.findall(r'[a-zA-Z]+', act_name)[0]

def get_act_index_dict(split_method,NL=True):
    """
    NL = True:
        act_index_dict = {actName1:actLabel1, actName2:actLabel2, ...}
    NL = False:
        act_index_dict = {actLabel1:actName1, actLabel2:actName2, ...}
    """
    act_index_dict = {}
    for actName,actConfig in split_method.items():
        _,_,label = list(i for _,i in actConfig.items())
        # extract label name without digits
        actName = get_ActName(actName)
        if not actName in act_index_dict:
            if NL:
                act_index_dict[actName] = label
            else:
                act_index_dict[label] = actName
    return act_index_dict

def get_feature_index(which_feature):
    feature_idx_lst = []
    feature_index_dict = get_feature_index_dict()
    for f in which_feature:
        feature_idx_lst.append(feature_index_dict[f])
    return feature_idx_lst

def get_feature_index_dict(split=False):
    """
    If split is True, then return dict containg 4 dicts, each containing corresponding features
    IF split is False, then concatenate all features together in 1 dict
    """
    feature_yaml_path = 'config/desired_features.yaml'
    with open(feature_yaml_path, "r") as file:
        features = yaml.safe_load(file)
    dists = features['desired_dists']
    angles = features['desired_angles']

    if not split:
        feature_index_dict = {}
        idx = 0
        for dist_name in dists:
            feature_index_dict[f'dist_{dist_name}']=idx
            idx += 1
        for dist_name in dists:
            feature_index_dict[f'dist_rate_{dist_name}']=idx
            idx += 1
        for angle_name in angles:
            feature_index_dict[f'angle_{angle_name}']=idx
            idx += 1
        for angle_name in angles:
            feature_index_dict[f'angle_rate_{angle_name}']=idx
            idx += 1

    else:
        feature_index_dict = {'dist':{},'dist_rate':{},'angle':{},'angle_rate':{}}
        idx = 0
        for dist_name in dists:
            feature_index_dict['dist'][f'dist_{dist_name}']=idx
            idx += 1
        for dist_name in dists:
            feature_index_dict['dist_rate'][f'dist_rate_{dist_name}']=idx
            idx += 1
        for angle_name in angles:
            feature_index_dict['angle'][f'angle_{angle_name}']=idx
            idx += 1
        for angle_name in angles:
            feature_index_dict['angle_rate'][f'angle_rate_{angle_name}']=idx
            idx += 1

    return feature_index_dict

def get_metric_index(which_metric):
    metric_idx_lst = []
    metric_index_dict = get_metric_index_dict()
    for metric in which_metric:
        metric_idx_lst.append(metric_index_dict[metric])
    return metric_idx_lst

def get_metric_index_dict():
    metric_index_dict = {'mean':0,
                         'std':1,
                         'top_max_mean':2,
                         'top_min_mean':3,
                         'max_min_range':4,
                         'kurtosis':5,
                         'skewness':6}
    return metric_index_dict
