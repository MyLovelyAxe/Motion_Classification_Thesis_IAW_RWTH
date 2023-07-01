import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime as dt

#################################
###### calculate distances ######
#################################
        
def calc_dist_plot(coords,links_idx):
    dist_sum = np.zeros(coords.shape[0])
    for i,j in zip(links_idx[:-1],links_idx[1:]):
        # coords[:,0,i]: all frames, x, i-th joint
        dist_sum += np.sqrt((coords[:,0,i] - coords[:,0,j])**2 + (coords[:,1,i] - coords[:,1,j])**2 + (coords[:,2,i] - coords[:,2,j])**2)
    return dist_sum
    
def calc_distances_timeline(coords,links_dict):
    # coords: [#frames,3,26]
    dist_time = np.zeros((len(links_dict),coords.shape[0])) # dist_time: [5,#frames]
    for idx,(_,links_idx) in enumerate(links_dict.items()):
        dist_time[idx,:] = calc_dist_plot(coords,links_idx)
    return dist_time

##############################
###### plot links in 3D ######
##############################

def connect_with_lines(frame,ax,joints_dict):
    # frame: [3,26]
    for links,joints in joints_dict.items():
        ax.plot(frame[0,joints], frame[1,joints], frame[2,joints], color='red')

#######################################
###### plot distance-lines in 2D ######
#######################################

def plot_distances(dist_time,frame_id,ax):
    for joints in dist_time:
        ax.plot(np.arange(frame_id),dist_time[joints][:frame_id+1])

#########################################################
###### get coordinates, distances, and joints list ######
#########################################################

def get_ori_data(path):
    # load original data from .csv
    csv_df = pd.read_csv(path ,header=None, delimiter=';',low_memory=False)
    csv_np = csv_df.to_numpy()
    csv_np = csv_np.astype(np.float32)
    csv_np = csv_np / 1000.0 # mm -> m

    # index in coords: x_pos, y_pos, z_pos:
    # original coordinates are different in y and z, so reserve them, both cog and other coordinates
    cog = csv_np[:,3:6]
    cog[:,[2,1]] = cog[:,[1,2]]
    x_pos_ori = csv_np[:,[6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,168]] # x_pos_ori: [#frames,26]
    z_pos_ori = csv_np[:,[7,13,19,25,31,37,43,49,55,61,67,73,79,85,91,97,103,109,115,121,127,133,139,145,151,169]]
    y_pos_ori = csv_np[:,[8,14,20,26,32,38,44,50,56,62,68,74,80,86,92,98,104,110,116,122,128,134,140,146,152,170]]
    
    # create an array
    coords = np.concatenate((np.expand_dims(x_pos_ori, axis=1),
                             np.expand_dims(y_pos_ori, axis=1),
                             np.expand_dims(z_pos_ori, axis=1)), axis=1) # coords: [#frames,3,26]
    cog = np.expand_dims(cog,axis=1)
    # print(f'coords shape: {coords.shape}')
    # print(f'cog shape: {cog.shape}')
    
    return cog,coords

def get_links_dict():
    links_dict = {'left arm': [15,0,1,2,14,24],
                  'right arm': [18,3,4,5,17,24],
                  'left leg': [16,6,7,8,9,20],
                  'right leg': [19,10,11,12,13,20],
                  'spine': [20,21,22,23,24,25]
                 }
    return links_dict

def get_prepared(path,frame_range=None):

    cog,coords = get_ori_data(path=path) 
    # if frame_range is designated, then return designated ones
    if not frame_range == None:
        coords = coords[frame_range[0]:frame_range[1]]
    # joints list
    links_dict = get_links_dict()
    # change of distances
    dist_time = calc_distances_timeline(coords,links_dict)
    
    return coords,dist_time,links_dict

##################################
###### plot with each frame ######
##################################

def calc_axis_limit(coords):
    high = np.array(list(int(np.ceil(coords[:,i,:].max()*1000/200.0))*200 for i in range(coords.shape[1])))
    low = np.array(list(int(np.ceil(coords[:,i,:].min()*1000/200.0))*200 for i in range(coords.shape[1])))
    return high,low

def prepare_ax(ax,high,low):
    # axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # axis limit
    ax.axes.set_xlim3d(left=low[0]/1000.0, right=high[0]/1000.0)
    ax.axes.set_ylim3d(bottom=low[1]/1000.0, top=high[1]/1000.0)
    ax.axes.set_zlim3d(bottom=low[2]/1000.0, top=high[2]/1000.0)
    # axis scale
    ax.set_xticks(list(i/1000.0 for i in range(low[0],high[0],200)))
    ax.set_yticks(list(i/1000.0 for i in range(low[1],high[1],200)))
    ax.set_zticks(list(i/1000.0 for i in range(low[2],high[2],200)))
    # axis aspect ratio
    ax.set_box_aspect(aspect = (high[0]-low[0],high[1]-low[1],high[2]-low[2]))

def plot_func_3d(frame_id,ax,joints_dict,coords,high,low,title):
    # plot links in 3D 
    ax.cla()
    current_frame = coords[frame_id] # current_frame: [3,26]
    prepare_ax(ax,high,low)
    ax.set_title(title)
    ax.scatter3D(current_frame[0], current_frame[1], current_frame[2], c='steelblue', marker='<')
    connect_with_lines(current_frame,ax,joints_dict)

def plot_func_2d(frame_id,ax,dist_plots,dist_time,title):
    # plot distance-lines in 2D
    # attentions:
    #    don't call ax.cla() for 2D projection, because previous plot is needed
    ax.set_title(title)
    x = np.arange(frame_id)
    for idx,dist_plot in enumerate(dist_plots):
        dist_plot.set_data(x, dist_time[idx][:frame_id])

def plot_func(frame_id,ax1,joints_dict,coords_ori,high_1,low_1,title_1,ax2,dist_plots,dist_time,title_2):
    plot_func_3d(frame_id,ax1,joints_dict,coords_ori,high_1,low_1,title_1)
    plot_func_2d(frame_id,ax2,dist_plots,dist_time,title_2)

def plot_ori_data(input_path,args):
        ### prepare ###
        coords,dist_time,joints_dict = get_prepared(input_path,frame_range=[args.start_frame,args.end_frame])
        N_frames = args.end_frame - args.start_frame
        ### plot ###
        fig = plt.figure(figsize=(16,8))
        fig.tight_layout()
        ### ax1: original coordinates in 3D ###
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.view_init(30, 150)
        high_1, low_1 = calc_axis_limit(coords) # (x_high, x_low), (y_high, y_low), (z_high, z_low)
        title_1 = f'Original coordinates in frames {args.start_frame}-{args.end_frame}'
        #### ax2: distance changing in 2D ###
        ax2 = fig.add_subplot(1, 2, 2)
        x = np.arange(N_frames)
        dist_plots = [ax2.plot(x, dist_time[idx],label=f'{links}')[0] for idx,(links,joints) in enumerate(joints_dict.items())]
        plt.legend()
        title_2 = f'Distances of limbs in frames {args.start_frame}-{args.end_frame}'
        ### animation ###
        ani1 = animation.FuncAnimation(fig,plot_func,frames=N_frames,fargs=(ax1,joints_dict,coords,high_1,low_1,title_1,
                                                                            ax2,dist_plots,dist_time,title_2),interval=17)
        plt.show()
        #### save animation.gif ###
        if args.output_anim:
            writergif = animation.PillowWriter(fps=30) 
            ani1.save(os.path.join(args.data_path,f"anim_{dt.now().strftime('%d-%h-%H-%M')}_{args.data_path.split('/')[1]}.gif"), writer=writergif)

################################################################
###### Calculation of all distances between random joints ######
################################################################

def get_joint_index():
    joint_index_dict = {'LWrist':0, 'LElbow':1, 'LShoulder':2,
                       'RWrist':3, 'RElbow':4, 'RShoulder':5,
                       'LToe':6, 'LAnkle':7, 'LKnee':8, 'LHip':9,
                       'RToe':10, 'RAnkle':11, 'RKnee':12, 'RHip':13,
                       'LClavicle':14, 'LHandEnd':15, 'LToesEnd':16,
                       'RClavicle':17, 'RHandEnd':18, 'RToesEnd':19,
                       'spine1':20, 'spine2':21, 'spine3':22, 'spine4':23, 'spine5':24, 'head':25
                      }
    return joint_index_dict

def calc_all_distances(ori_data_path):
    # e,g, dist_feature[3,1,4] means distance between LElbow and RElbow in 3rd frame
    _,coords = get_ori_data(ori_data_path) # coords: [frames,xyz,num_joints] = [18000,3,26]
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
    return all_distances

def examine_distance(ori_data_path,frame,joint_1,joint_2):
    _,coords = get_ori_data(ori_data_path) # coords: [frames,xyz,num_joints] = [18000,3,26]
    all_distances = calc_all_distances(ori_data_path)
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

def get_dist_feature(ori_data_path,desired_dists):
    all_distances = calc_all_distances(ori_data_path)
    joint_index_dict = get_joint_index()
    dist_feature = np.zeros((all_distances.shape[0],len(desired_dists)))
    for idx,desired_dist in enumerate(desired_dists):
        joint1,joint2 = desired_dist.split('_')
        dist_feature[:,idx] = all_distances[:,joint_index_dict[joint1],joint_index_dict[joint2]]
    del all_distances
    return dist_feature

##########################################################
###### Calculation of angles between adjacent links ######
##########################################################

def get_angle_pairs(desized_angles):
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
        if not all_angle_pairs_dict.get(desired_angle) == None:
            angle_pairs_dict[desired_angle] = all_angle_pairs_dict[desired_angle]
    return angle_pairs_dict

def calc_angles(joints_lst,distances):
    # b: the edge in triangle ABC which is opposite to the angle to be calculated
    # a,c: side edges
    # refer to formular: 
    # https://www.mathsisfun.com/algebra/trig-cosine-law.html
    a = distances[joints_lst[0],joints_lst[1]]
    b = distances[joints_lst[0],joints_lst[2]]
    c = distances[joints_lst[1],joints_lst[2]]
    cos_b = (c**2 + a**2 - b**2) / (2*c*a)
    B = np.arccos(cos_b) # radius = np.arccos(cos_value)
    return np.round(B,decimals=5)

def get_angle_feature(ori_data_path,desized_angles):
    all_distances = calc_all_distances(ori_data_path)
    angle_pairs_dict = get_angle_pairs(desized_angles)
    num_frames = all_distances.shape[0]
    num_angles = len(angle_pairs_dict)
    angle_feature = np.zeros((num_frames,num_angles))
    for frame, distances in enumerate(all_distances):
        for angle_idx, (angle_name,joints_lst) in enumerate(angle_pairs_dict.items()):
            angle_feature[frame,angle_idx] = calc_angles(joints_lst,distances)
    return angle_feature

##############################
###### generate dataset ######
##############################

def get_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        features = yaml.safe_load(file)
    dists = features['desired_dists']
    angles = features['desized_angles']
    return dists,angles

def get_all_features(ori_data_path,desired_dists,desized_angles):
    # get features
    dist_feature = get_dist_feature(ori_data_path,desired_dists)
    angle_feature = get_angle_feature(ori_data_path,desized_angles)
    all_features = np.concatenate((dist_feature, angle_feature), axis=1)
    return all_features

def output_dataset(ori_data_path,desired_dists,desized_angles,output_name='UpperBody', output_npy=False):
    all_features = get_all_features(ori_data_path,desired_dists,desized_angles)
    x_data = np.concatenate((all_features[200:3700],
                            all_features[3900:7200],
                            all_features[7400:10700],
                            all_features[10900:14400],
                            all_features[14600:]),axis=0)
    y_data = np.concatenate((np.full((3700-200),1),
                            np.full((7200-3900),2),
                            np.full((10700-7400),3),
                            np.full((14400-10900),4),
                            np.full((18000-14600),5)),axis=0)
    if output_npy:
        output_path = os.path.join(ori_data_path.split('/')[0],ori_data_path.split('/')[1])
        print(f'type: {type(x_data)}, shape: {x_data.shape}')
        np.save(os.path.join(output_path,f'x_data_{output_name}.npy'), x_data)
        print(f'type: {type(y_data)}, shape: {y_data.shape}')
        np.save(os.path.join(output_path,f'y_data_{output_name}.npy'), y_data)
    else:
        return [x_data,y_data]
    
#########################################
###### verify dataset to be output ######
#########################################

def verification_dataset(ori_data_path,desired_dists,desized_angles):
    _,coords = get_ori_data(ori_data_path)
    x_data,y_data = output_dataset(ori_data_path,desired_dists,desized_angles)
    skeletons = np.concatenate((coords[200:3700],
                               coords[3900:7200],
                               coords[7400:10700],
                               coords[10900:14400],
                               coords[14600:]),axis=0)
    return x_data,y_data,skeletons

def calc_dist_verify(skeleton,joint_1_idx,joint_2_idx):
    dist = np.sqrt((skeleton[0,joint_1_idx] - skeleton[0,joint_2_idx])**2 
                        + (skeleton[1,joint_1_idx] - skeleton[1,joint_2_idx])**2 
                        + (skeleton[2,joint_1_idx] - skeleton[2,joint_2_idx])**2
                       )
    return np.around(dist,decimals=5)
    
def dist_equal_or_not(skeleton,dist_feature,desired_dists):
    joint_index_dict = get_joint_index()
    dist_equal_lst = []
    for idx,desired_dist in enumerate(desired_dists):
        dist_dataset = dist_feature[idx]
        joint1,joint2 = desired_dist.split('_')
        joint_1_idx = joint_index_dict[joint1]
        joint_2_idx = joint_index_dict[joint2]
        dist_skeleton = calc_dist_verify(skeleton,joint_1_idx,joint_2_idx)
        dist_equal_lst.append(dist_dataset == dist_skeleton)
        if not dist_dataset == dist_skeleton:
            print(f'dist_dataset={dist_dataset},dist_skeleton={dist_skeleton}')
    return np.all(np.array(dist_equal_lst))

def calc_angle_verify(skeleton,joints_lst):
    # b: the edge in triangle ABC which is opposite to the angle to be calculated
    # a,c: side edges
    # refer to formular: 
    # https://www.mathsisfun.com/algebra/trig-cosine-law.html
    a = calc_dist_verify(skeleton,joints_lst[0],joints_lst[1])
    b = calc_dist_verify(skeleton,joints_lst[0],joints_lst[2])
    c = calc_dist_verify(skeleton,joints_lst[1],joints_lst[2])
    cos_b = (c**2 + a**2 - b**2) / (2*c*a)
    B = np.arccos(cos_b) # radius = np.arccos(cos_value)
    return np.around(B,decimals=5)

def angles_equal_or_not(skeleton,angle_feature,desized_angles):
    angle_pairs_dict = get_angle_pairs(desized_angles)
    angle_equal_lst = []
    for idx, (angle_name,joints_lst) in enumerate(angle_pairs_dict.items()):
        angle_dataset = angle_feature[idx]
        angle_skeleton = calc_angle_verify(skeleton,joints_lst)
        angle_equal_lst.append(angle_dataset == angle_skeleton)
        if not angle_dataset == angle_skeleton:
            print(f'angle_dataset={angle_dataset},angle_skeleton={angle_skeleton}')
    return np.all(np.array(angle_equal_lst))

def verification(ori_data_path,yaml_path,dataset_path=None):
    desired_dists,desized_angles = get_yaml(yaml_path)
    if not dataset_path == None:
        x_data_path, y_data_path = dataset_path[0], dataset_path[1]
        with open(x_data_path, 'rb') as xf:
            x_data = np.load(xf)
        with open(y_data_path, 'rb') as yf:
            y_data = np.load(yf)
        _,_,skeletons = verification_dataset(ori_data_path,desired_dists,desized_angles)
        assert x_data.shape[1] == len(desired_dists) + len(desized_angles), 'Number of features of loaded dataset is different from verified features'
        print(f'Loaded x_data with shape {x_data.shape}')
        print(f'Loaded y_data with shape {y_data.shape}')
    else:
        x_data,y_data,skeletons = verification_dataset(ori_data_path,desired_dists,desized_angles)
        print(f'Generated x_data with shape {x_data.shape}')
        print(f'Generated y_data with shape {y_data.shape}')
    # select random frames
    choices = np.random.randint(17000, size = 16)
    print(f'These frames will be verified: {choices}')
    # plot
    links_dict = get_links_dict()
    fig = plt.figure(figsize=(15,15))
    for idx,(frame,feature,label,skeleton) in enumerate(zip(choices,x_data[choices],y_data[choices],skeletons[choices])):
        # verify whether distances and angles are the same with dataset
        dist_equal = dist_equal_or_not(skeleton,feature[:len(desired_dists)],desired_dists)
        angles_equal = angles_equal_or_not(skeleton,feature[len(desired_dists):],desized_angles)
        # plot skeleton
        ax = fig.add_subplot(4, 4, idx+1, projection='3d')
        high,low = calc_axis_limit(np.expand_dims(skeleton, axis=0))
        prepare_ax(ax,high,low)
        ax.scatter3D(skeleton[0], skeleton[1], skeleton[2], c='steelblue', marker='<')
        connect_with_lines(skeleton,ax,links_dict)
        ax.set_title(f'label: {label}, dist: {dist_equal}, angles: {angles_equal}')
        ax.set_axis_off()
    fig.tight_layout()
    plt.show()