import os
import pandas as pd
import numpy as np

#################################
###### calculate distances ######
#################################
        
def calc_dist(coords,joints_idx):
    dist_sum = np.zeros(coords.shape[0])
    for i,j in zip(joints_idx[:-1],joints_idx[1:]):
        # coords[:,0,i]: all frames, x, i-th joint
        dist_sum += np.sqrt((coords[:,0,i] - coords[:,0,j])**2 + (coords[:,1,i] - coords[:,1,j])**2 + (coords[:,2,i] - coords[:,2,j])**2)
    return dist_sum
    
def calc_distances_timeline(coords,joints_dict):
    # coords: [#frames,3,26]
    dist_time = np.zeros((len(joints_dict),coords.shape[0])) # dist_time: [5,#frames]
    # dist_time = []
    for idx,(_,joints_idx) in enumerate(joints_dict.items()):
        dist_time[idx,:] = calc_dist(coords,joints_idx)
    print(f'dist_time shape: {dist_time.shape}')
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

def get_ori_data(path,save):
    
    csv_df = pd.read_csv(path ,header=None, delimiter=';',low_memory=False)
    csv_np = csv_df.to_numpy()
    csv_np = csv_np.astype(np.float32)
    csv_np = csv_np / 1000.0 # mm -> m
    output_path = './Dataset_Proposal'

    # index in coords: x_pos, y_pos, z_pos:
    # original coordinates are different in y and z, so reserve them, both cog and other coordinates
    cog = csv_np[:,3:6]
    cog[:,[2,1]] = cog[:,[1,2]]
    x_pos_ori = csv_np[:,[6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,168]] # x_pos_ori: [#frames,26]
    z_pos_ori = csv_np[:,[7,13,19,25,31,37,43,49,55,61,67,73,79,85,91,97,103,109,115,121,127,133,139,145,151,169]]
    y_pos_ori = csv_np[:,[8,14,20,26,32,38,44,50,56,62,68,74,80,86,92,98,104,110,116,122,128,134,140,146,152,170]]
    print(f'x_pos shape: {x_pos_ori.shape}')
    
    coords = np.concatenate((np.expand_dims(x_pos_ori, axis=1),
                             np.expand_dims(y_pos_ori, axis=1),
                             np.expand_dims(z_pos_ori, axis=1)), axis=1) # coords: [#frames,3,26]
    cog = np.expand_dims(cog,axis=1)
    print(f'coords shape: {coords.shape}')
    print(f'cog shape: {cog.shape}')
    
    return cog,coords

def get_prepared(path,frame_range=None):

    cog,coords = get_ori_data(path=path,save='no') 

    # if frame_range is designated, then return designated ones
    if not frame_range == None:
        coords = coords[frame_range[0]:frame_range[1]]
    
    # joints list
    joints_dict = {'left arm': [15,0,1,2,14,24],
                  'right arm': [18,3,4,5,17,24],
                  'left leg': [16,6,7,8,9,20],
                  'right leg': [19,10,11,12,13,20],
                  'spine': [20,21,22,23,24,25]
                 }
    # change of distances
    dist_time = calc_distances_timeline(coords,joints_dict)
    
    return coords,dist_time,joints_dict

##################################
###### plot with each frame ######
##################################

def calc_axis_limit(coords):
    x_high, x_low = int(np.ceil(coords[:,0,:].max()*1000/200.0))*200, int(np.floor(coords[:,0,:].min()*1000/200.0))*200
    y_high, y_low = int(np.ceil(coords[:,1,:].max()*1000/200.0))*200, int(np.floor(coords[:,1,:].min()*1000/200.0))*200
    z_high, z_low = int(np.ceil(coords[:,2,:].max()*1000/200.0))*200, int(np.floor(coords[:,2,:].min()*1000/200.0))*200
    high = np.array([x_high,y_high,z_high])
    low = np.array([x_low,y_low,z_low])
    del x_high, x_low, y_high, y_low, z_high, z_low
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