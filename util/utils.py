import os
import pandas as pd
import numpy as np

#################################
###### calculate distances ######
#################################
        
def calc_dist(coords,joints):
    dist_sum = np.zeros(coords[0].shape[0])
    for i,j in zip(joints[:-1],joints[1:]):
        dist_sum += np.sqrt((coords[0][:,i] - coords[0][:,j])**2 + (coords[1][:,i] - coords[1][:,j])**2 + (coords[2][:,i] - coords[2][:,j])**2)
    return dist_sum
    
def calc_distances_timeline(coords,joints_dict):
    dist_time = []
    for links,joints in joints_dict.items():
        dist_time.append(calc_dist(coords,joints))
    return dist_time

##############################
###### plot links in 3D ######
##############################

def connect(frame,ax,joints):
    idx_lst = [[],[],[]]
    for i in joints:
        idx_lst[0].append(frame[0][i])
        idx_lst[1].append(frame[1][i])
        idx_lst[2].append(frame[2][i])
        ax.plot(idx_lst[0], idx_lst[1], idx_lst[2], color='red')

def connect_with_lines(frame,ax,joints_dict):
    for links,joints in joints_dict.items():
        connect(frame,ax,joints)

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
    x_pos_ori = csv_np[:,[6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,168]]
    z_pos_ori = csv_np[:,[7,13,19,25,31,37,43,49,55,61,67,73,79,85,91,97,103,109,115,121,127,133,139,145,151,169]]
    y_pos_ori = csv_np[:,[8,14,20,26,32,38,44,50,56,62,68,74,80,86,92,98,104,110,116,122,128,134,140,146,152,170]]
    print(f'x_pos shape: {x_pos_ori.shape}')
    
    # save the original position vectors into csv, or not
    if save == 'yes':
        num_row = x_pos_ori.shape[0]
        num_col = x_pos_ori.shape[1] * 3 + 3
        output_df = np.zeros((num_row,num_col))
        for frame,(C,X,Y,Z) in enumerate(zip(cog,x_pos_ori,y_pos_ori,z_pos_ori)):
            output_df[frame,0:3] = C # center of gravity
            output_df[frame,3::3] = X # x position
            output_df[frame,4::3] = Y # y position
            output_df[frame,5::3] = Z # z position
        df = pd.DataFrame(output_df)
        df.to_csv(os.path.join(output_path,'coordinates_yz.csv'),index=False,header=False)
    
    return cog,x_pos_ori,y_pos_ori,z_pos_ori
    
def get_prepared(path,frame_range=None):

    cog,x_pos_ori,y_pos_ori,z_pos_ori = get_ori_data(path=path,save='no')

    # if frame_range is not designated, then return all data
    if frame_range == None:
        coords_ori = [x_pos_ori,y_pos_ori,z_pos_ori]
    # if frame_range is designated, then return the designated ones
    else:
        coords_ori = [x_pos_ori[frame_range[0]:frame_range[1]],y_pos_ori[frame_range[0]:frame_range[1]],z_pos_ori[frame_range[0]:frame_range[1]]]
    
    # joints list
    joints_dict = {'left arm': [15,0,1,2,14,24],
                  'right arm': [18,3,4,5,17,24],
                  'left leg': [16,6,7,8,9,20],
                  'right leg': [19,10,11,12,13,20],
                  'spine': [20,21,22,23,24,25]
                 }
    # change of distances
    dist_time = calc_distances_timeline(coords_ori,joints_dict)
    
    return coords_ori,dist_time,joints_dict

##################################
###### plot with each frame ######
##################################

def calc_axis_limit(coords):
    x_high, x_low = int(np.ceil(coords[0].max()*1000/200.0))*200, int(np.floor(coords[0].min()*1000/200.0))*200
    y_high, y_low = int(np.ceil(coords[1].max()*1000/200.0))*200, int(np.floor(coords[1].min()*1000/200.0))*200
    z_high, z_low = int(np.ceil(coords[2].max()*1000/200.0))*200, int(np.floor(coords[2].min()*1000/200.0))*200
    return [[x_high, x_low],[y_high, y_low],[z_high, z_low]]

def prepare_ax(coords,ax,limits):
    # axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # axis limit
    x_high, x_low = limits[0][0],limits[0][1]
    y_high, y_low = limits[1][0],limits[1][1]
    z_high, z_low = limits[2][0],limits[2][1]
    ax.axes.set_xlim3d(left=x_low/1000.0, right=x_high/1000.0) 
    ax.axes.set_ylim3d(bottom=y_low/1000.0, top=y_high/1000.0)
    ax.axes.set_zlim3d(bottom=z_low/1000.0, top=z_high/1000.0)
    # axis scale
    ax.set_xticks(list(i/1000.0 for i in range(x_low,x_high,200)))
    ax.set_yticks(list(i/1000.0 for i in range(y_low,y_high,200)))
    ax.set_zticks(list(i/1000.0 for i in range(z_low,z_high,200)))
    # axis aspect ratio
    ax.set_box_aspect(aspect = (x_high-x_low,y_high-y_low,z_high-z_low))

def plot_func_3d(frame_id,ax,joints_dict,coords,limits,title):
    # plot links in 3D 
    ax.cla()
    current_frame = [coords[0][frame_id], coords[1][frame_id], coords[2][frame_id]] # x_pos,z_pos,y_pos
    prepare_ax(coords,ax,limits)
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

def plot_func(frame_id,ax1,joints_dict,coords_ori,limits_1,title_1,ax2,dist_plots,dist_time,title_2):
    plot_func_3d(frame_id,ax1,joints_dict,coords_ori,limits_1,title_1)
    plot_func_2d(frame_id,ax2,dist_plots,dist_time,title_2)