from datetime import datetime as dt
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from util.utils import get_ori_data,output_dataset,get_feature_selection
from util.features import get_joint_index,get_angle_pairs,get_scales_dict,calc_all_distances,get_dist_feature


def calc_distances_timeline(coords,scales_dict):
    """
    calculate the sum of length of desired limbs along with timeline
    """
    # coords: [#frames,3,26]
    dist_time = np.zeros((len(scales_dict),coords.shape[0])) # dist_time: [5,#frames]
    distances = calc_all_distances(coords)
    for idx,(_,links_idx) in enumerate(scales_dict.items()):
        # dist_feature: [#frames,#features] -> [#frames,] -> [1,#frames]
        dist_feature = np.expand_dims(np.sum(get_dist_feature(distances,links_idx),axis=1),axis=0)
        dist_time[idx,:] = dist_feature
    return dist_time

def connect_with_lines(frame,ax,joints_dict):
    """
    plot links in 3D
    """
    # frame: [3,26]
    for _,joints in joints_dict.items():
        ax.plot(frame[0,joints], frame[1,joints], frame[2,joints], color='red')

def plot_distances(dist_time,frame_id,ax):
    """
    plot distance-lines in 2D
    """
    for joints in dist_time:
        ax.plot(np.arange(frame_id),dist_time[joints][:frame_id+1])

def joints_name2index(joints_name_lst):
    """
    convert joint_name into joint_idx from list
    """
    joint_index_dict = get_joint_index()
    joint_idx_lst = []
    for jName in joints_name_lst:
        joint_idx_lst.append(joint_index_dict[jName])
    return joint_idx_lst

def get_links_dict():
    """
    get joint indices for connecting joints in 3D projection plot
    """
    joint_names_dict = {
        'left arm': ['LHandEnd','LWrist','LElbow','LShoulder','LClavicle','spine5'],
        'right arm': ['RHandEnd','RWrist','RElbow','RShoulder','RClavicle','spine5'],
        'left leg': ['LToesEnd','LToe','LAnkle','LKnee','LHip','spine1'],
        'right leg': ['RToesEnd','RToe','RAnkle','RKnee','RHip','spine1'],
        'spine': ['spine1','spine2','spine3','spine4','spine5','head']
        }
    links_dict = {}
    for link_name,joints_name_lst in joint_names_dict.items():
        links_dict[link_name] = joints_name2index(joints_name_lst)
    
    return links_dict

def prepare_for_plot(path,frame_range=None):
    """
    return coordinates and distances
    """
    _,coords = get_ori_data(path=path) 
    # if frame_range is designated, then return designated ones
    if not frame_range is None:
        coords = coords[frame_range[0]:frame_range[1]]
    # change of distances
    dist_time = calc_distances_timeline(coords,get_scales_dict())
    return coords,dist_time

def calc_axis_limit(coords):
    """
    calculatet the limits for each axis, i.e. x,y,z
    """
    high = np.array(list(int(np.ceil(coords[:,i,:].max()*1000/200.0))*200 for i in range(coords.shape[1])))
    low = np.array(list(int(np.ceil(coords[:,i,:].min()*1000/200.0))*200 for i in range(coords.shape[1])))
    return high,low

def prepare_ax(ax,high,low):
    """
    define the axis label, axis limit, axis scale and zooming ratio
    """
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
    """
    plot links in 3D
    """
    ax.cla()
    # ax.set_axis_off()
    current_frame = coords[frame_id] # current_frame: [3,26]
    prepare_ax(ax,high,low)
    ax.set_title(title)
    ax.scatter3D(current_frame[0], current_frame[1], current_frame[2], c='steelblue', marker='<')
    connect_with_lines(current_frame,ax,joints_dict)

def plot_func_2d(frame_id,ax,dist_plots,dist_time,title):
    """
    plot distance-lines in 2D
    attentions:
       don't call ax.cla() for 2D projection, because previous plot is needed
    """
    ax.set_title(title)
    x = np.arange(frame_id)
    for idx,dist_plot in enumerate(dist_plots):
        dist_plot.set_data(x, dist_time[idx][:frame_id])

def plot_func(frame_id,ax1,joints_dict,coords_ori,high_1,low_1,title_1,ax2,dist_plots,dist_time,title_2):
    """
    plot 3d projection and 2d plain
    """
    plot_func_3d(frame_id,ax1,joints_dict,coords_ori,high_1,low_1,title_1)
    plot_func_2d(frame_id,ax2,dist_plots,dist_time,title_2)

def plot_ori_data(data_path,start_frame,end_frame,output_anim):
    """
    plot original data without cutting and selecting
    """
    ### prepare ###
    input_path = os.path.join(data_path,'unknown.NoHead.csv')
    coords,dist_time = prepare_for_plot(input_path,frame_range=[start_frame,end_frame])
    scales_dict = get_scales_dict()
    joints_dict = get_links_dict()
    N_frames = end_frame - start_frame
    ### plot ###
    fig = plt.figure(figsize=(16,8))
    fig.tight_layout()
    ### ax1: original coordinates in 3D ###
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.view_init(30, 150)
    high_1, low_1 = calc_axis_limit(coords) # (x_high, x_low), (y_high, y_low), (z_high, z_low)
    title_1 = f'Original coordinates in frames {start_frame}-{end_frame}'
    #### ax2: distance changing in 2D ###
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(N_frames)
    dist_plots = [ax2.plot(x, dist_time[idx],label=f'{links}')[0] for idx,(links,joints) in enumerate(scales_dict.items())]
    title_2 = f'Distances of limbs in frames {start_frame}-{end_frame}'
    plt.legend()
    plt.grid(True)
    ### animation ###
    ani1 = animation.FuncAnimation(fig,plot_func,frames=N_frames,fargs=(ax1,joints_dict,coords,high_1,low_1,title_1,
                                                                        ax2,dist_plots,dist_time,title_2),interval=17)
    plt.show()
    #### save animation.gif ###
    if output_anim:
        writergif = animation.PillowWriter(fps=30) 
        ani1.save(os.path.join(data_path,f"anim_{dt.now().strftime('%d-%h-%H-%M')}_{data_path.split('/')[1]}.gif"), writer=writergif)

##########################################################
###### verify dataset to be output by visualization ######
##########################################################

def calc_dist_verify(skeleton,joint_1_idx,joint_2_idx):
    """
    skeleton contains several frames: [#frames,xyz,joint]
    """
    if len(skeleton.shape) == 2:
        skeleton = np.expand_dims(skeleton,axis=0)
    dist = np.sqrt(np.sum((skeleton[...,joint_1_idx] - skeleton[...,joint_2_idx])**2,axis=1))
    return np.around(dist,decimals=5)
    
def dist_equal_or_not(skeleton,dist_feature,desired_dists):
    """
    check whether all distances in samples are equal
    """
    joint_index_dict = get_joint_index()
    for idx,desired_dist in enumerate(desired_dists):
        dist_dataset = dist_feature[:,idx]
        joint1,joint2 = desired_dist.split('_')
        joint_1_idx = joint_index_dict[joint1]
        joint_2_idx = joint_index_dict[joint2]
        dist_skeleton = calc_dist_verify(skeleton,joint_1_idx,joint_2_idx)
        all_equal = np.all(np.equal(dist_dataset,dist_skeleton))
        if not all_equal:
            NoEqual_index = np.argwhere(dist_dataset != dist_skeleton)
            print(f'{NoEqual_index}th frame of feature: {desired_dist} in this window has difference:')
            print(f'dist_dataset={dist_dataset[NoEqual_index]}, dist_skeleton={dist_skeleton[NoEqual_index]}')
            print()

def calc_angle_verify(skeleton,joints_lst):
    """
    calculate angles when only 1 frame of original skeleton data is provided
        b: the edge in triangle ABC which is opposite to the angle to be calculated
        a,c: side edges
        refer to formular: 
        https://www.mathsisfun.com/algebra/trig-cosine-law.html
    """
    a = calc_dist_verify(skeleton,joints_lst[0],joints_lst[1])
    b = calc_dist_verify(skeleton,joints_lst[0],joints_lst[2])
    c = calc_dist_verify(skeleton,joints_lst[1],joints_lst[2])
    cos_b = (c**2 + a**2 - b**2) / (2*c*a)
    B = np.arccos(cos_b) # radius = np.arccos(cos_value)
    return np.around(B,decimals=5)

def angles_equal_or_not(skeleton,angle_feature,desired_angles):
    """
    check whether all calculated angles are correct
    """
    angle_pairs_dict = get_angle_pairs(desired_angles)
    for idx, (angle_name,joints_lst) in enumerate(angle_pairs_dict.items()):
        angle_dataset = angle_feature[:,idx]
        angle_skeleton = calc_angle_verify(skeleton,joints_lst)
        all_equal = np.all(np.equal(angle_dataset,angle_skeleton))
        if not all_equal:
            NoEqual_index = np.argwhere(angle_dataset != angle_skeleton)
            print(f'{NoEqual_index}th frame of feature: {angle_name} in this window has difference:')
            print(f'angle_dataset={angle_dataset[NoEqual_index]}, angle_skeleton={angle_skeleton[NoEqual_index]}')
            print()

def dynamic_plot_func(frame_id,anim_configs_lst):
    """
    plot_configs_lst consists of lists, each of list contains:
        ax,coords,high,low,title
    """
    joints_dict = get_links_dict()
    for anim_configs in anim_configs_lst:
        ax1,coords,high,low,title = anim_configs
        plot_func_3d(frame_id,ax1,joints_dict,coords,high,low,title)

def crossLine(start,start_index,end_index,win_len):
    # add end_index into start_index
    end_index = np.expand_dims(end_index,axis=0)
    start_index = np.concatenate((start_index,end_index),axis=0)
    # get distances between start and each start index
    dist = start_index - start
    # find the smallest positive distance, which is closest to start and on the right
    closest_dist = dist[dist>0][0]
    # check whether it is larger than window length, i.e. whether current window crosses line
    cross = False if closest_dist >= win_len else True
    return cross

def verification(ori_data_paths,feature_path,split_method_paths,win_len=1):
    """
    verify whether output datasets have wrong calculated values, and whether the label is correct
    """
    ### get data
    desired_dists,desired_angles = get_feature_selection(feature_path)
    x_data,y_data,skeletons = output_dataset(ori_data_paths,
                                             split_method_paths,
                                             desired_dists,
                                             desired_angles,
                                             standard='no_scale')
    print(f'Trial x_data with shape {x_data.shape}')
    print(f'Trial y_data with shape {y_data.shape}')

    ### select random frames
    choice_starts = np.random.randint(x_data.shape[0], size = 3)
    _,start_index = np.unique(y_data,return_index=True)
    print(f'start_index here: {start_index}')
    choices_lst = []
    for start in choice_starts:
        # if distance between choice and start index of each activity
        # is less than window, then a window doesn't cross broundries
        if not crossLine(start,start_index,len(y_data),win_len):
            choices_lst.append(range(start,start+win_len))
        # if a window crosses 2 segments belonging to different activities
        # then take the window in reserve direction
        else:
            choices_lst.append(range(start-win_len,start))
    print(f'These frames will be verified: {choices_lst}')

    ### plot
    anim_configs_lst = []
    fig = plt.figure(figsize=(24,12))
    for idx,choices in enumerate(choices_lst):
        feature = x_data[choices]
        label = y_data[choices[len(choices)//2]]
        skeleton = skeletons[choices]
        # verify whether distances and angles are the same with dataset
        dist_equal_or_not(skeleton,feature[:,:len(desired_dists)],desired_dists)
        angles_equal_or_not(skeleton,feature[:,2*len(desired_dists):2*len(desired_dists)+len(desired_angles)],desired_angles)
        # plot animation of skeleton
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        ax.view_init(30, 150)
        high,low = calc_axis_limit(skeleton) # (x_high, x_low), (y_high, y_low), (z_high, z_low)
        title = f'label: {label} in frames {choices[0]}-{choices[-1]}'
        anim_configs_lst.append([ax,skeleton,high,low,title])
    _ = animation.FuncAnimation(fig,dynamic_plot_func,frames=win_len,fargs=(anim_configs_lst,),interval=17)
    plt.show()

def show_misclassified_frames(skeleton,MisCls_frames):
    """
    examine the misclassified windows from result of experiment
    """
    start_frame, end_frame = MisCls_frames
    skeleton = skeleton[start_frame:end_frame]
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.view_init(30, 150)
    joints_dict = get_links_dict()
    high, low = calc_axis_limit(skeleton) # (x_high, x_low), (y_high, y_low), (z_high, z_low)
    title = f'Misclassified frames {start_frame}-{end_frame}'
    _ = animation.FuncAnimation(fig,
                                plot_func_3d,
                                frames=end_frame-start_frame,
                                fargs=(ax1,joints_dict,skeleton,high,low,title),
                                interval=17)
    fig.tight_layout()
    plt.show()
