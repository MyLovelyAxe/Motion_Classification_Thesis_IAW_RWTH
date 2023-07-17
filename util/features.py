import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

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
        B = np.float32(np.pi)
    else:
        B = np.arccos(cos_b) # radius = np.arccos(cos_value)
    B = np.round(B.astype(np.float32),decimals=5)
    return B

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
    return np.mean(data,axis=1)

def calc_Std(data) -> np.array:
    """
    shape:
        inpu: [#win,window_size,#features]
        output: [#win,#features]
    type:
        standard deviation of data along dimension of window
    """
    return np.std(data,axis=1)

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
    return kurtosis(data,axis=1)

def calc_Skewness(data) -> np.array:
    """
    shape:
        inpu: [#win,window_size,#features]
        output: [#win,#features]
    type:
        the concept of skewness:
        https://www.scribbr.com/statistics/skewness/
    """
    return skew(data,axis=1)

def calc_FFT(data):
    pass

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
                                         calc_TopMean_Range(x_data),
                                         calc_Kurtosis(x_data),
                                         calc_Skewness(x_data),
                                         ),axis=1)
    return dynamic_statistics

if __name__ == '__main__':

    from scipy.fft import fft,fftfreq
    import matplotlib.pyplot as plt

    data = np.random.randint(0,24,40).reshape(2,5,4)
    FFT = fft(data,axis=1)
    freq = fftfreq(data.shape[1])
    freq_index = np.argsort(freq)
    print(f'freq index shape: {freq_index.shape}')
    # freq_index = np.expand_dims(np.expand_dims(freq_index,axis=0),axis=-1)
    # sorted_real_FFT = np.take_along_axis(FFT.real,freq_index,axis=1)
    # sorted_imag_FFT = np.take_along_axis(FFT.imag,freq_index,axis=1)
    # print(f'real sorted fft: {sorted_real_FFT}')
    # print(f'imag sorted fft: {sorted_imag_FFT}')
    freq.sort()
    fig = plt.figure(figsize=(8,6))
    ax1 = plt.subplot(1,1,1)
    # ax2 = plt.subplot(1,3,2)
    # ax3 = plt.subplot(1,3,3)
    half_index = freq_index[len(freq)//2:]
    half_freq = freq[len(freq)//2:]
    for frame in range(data.shape[0]):
        for feature in range(data.shape[-1]):
            
            ax1.plot(half_freq, FFT[frame,:,feature][half_index],label=f'frame{frame+1} feature{feature+1}')
            # ax2.plot(freq, FFT.real[frame,:,feature][freq_index],label=f'real frame{frame+1} feature{feature+1}')
            # ax3.plot(freq, FFT.imag[frame,:,feature][freq_index],label=f'imag frame{frame+1} feature{feature+1}')
    ax1.set_title(f'fft')
    ax1.set_xticks(half_freq,half_freq)
    ax1.legend()

    # ax2.set_xticks(freq,freq)
    # ax2.set_title(f'real fft')
    # ax2.legend()
    # ax3.set_xticks(freq,freq)
    # ax3.set_title(f'imag fft')
    # ax3.legend()

    plt.show()
