import os
import yaml
import pandas as pd
import numpy as np
from util.features import get_all_features

#########################################
###### load original skeleton data ######
#########################################

def get_ori_data(path):
    """
    get coordinates, distances, and joints list
    """
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
    print(f'coords shape: {coords.shape}')
    print(f'cog shape: {cog.shape}')
    return cog,coords

##############################
###### generate dataset ######
##############################

def get_feature_selection(yaml_path):
    with open(yaml_path, "r") as file:
        features = yaml.safe_load(file)
    dists = features['desired_dists']
    angles = features['desired_angles']
    return dists,angles

def get_splilt_method(yaml_path,show=False):
    with open(yaml_path, "r") as file:
        split_method = yaml.safe_load(file)
    if show:
        print(f'split method: {split_method}')
    return split_method

def output_dataset(ori_data_paths,
                   split_method_paths,
                   desired_dists,
                   desired_angles,
                   output_path=None,
                   output_name='UpperBody'):
    out_dict = {}
    for split_path,data_path in zip(split_method_paths,ori_data_paths):
        _,coords = get_ori_data(data_path)
        split_method = get_splilt_method(split_path)
        all_features = get_all_features(coords,desired_dists,desired_angles)
        # e.g. dynamic split_method = {'Boxing1': {'start': 200, 'end': 3700, 'label': 1}}
        for act_name,config in split_method.items():
            start,end,label = list(i for _,i in config.items())
            if not act_name[:-1] in out_dict:
                out_dict[act_name[:-1]] = {'x_data':[],'y_data':[],'skeleton':[]}
            out_dict[act_name[:-1]]['x_data'].append(all_features[start:end])
            out_dict[act_name[:-1]]['y_data'].append(np.full((end-start),label))
            out_dict[act_name[:-1]]['skeleton'].append(coords[start:end])
    # concatenate all activities
    x_data_lst = []
    y_data_lst = []
    skeletons_lst = []
    for act,data in out_dict.items():
        x_data_tmp = np.concatenate(data['x_data'],axis=0)
        y_data_tmp = np.concatenate(data['y_data'],axis=0)
        skeleton_tmp = np.concatenate(data['skeleton'],axis=0)
        x_data_lst.append(x_data_tmp)
        y_data_lst.append(y_data_tmp)
        skeletons_lst.append(skeleton_tmp)
    x_data = np.concatenate(x_data_lst,axis=0)
    y_data = np.concatenate(y_data_lst,axis=0)
    skeletons = np.concatenate(skeletons_lst,axis=0)
    del x_data_lst,y_data_lst,skeletons_lst,x_data_tmp,y_data_tmp,skeleton_tmp,out_dict

    if not output_path is None:
        # skeleton is only used for varification
        # so don't output skeleton
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        print(f'type: {type(x_data)}, shape: {x_data.shape}')
        np.save(os.path.join(output_path,f'x_data_{output_name}.npy'),x_data)
        print(f'type: {type(y_data)}, shape: {y_data.shape}')
        np.save(os.path.join(output_path,f'y_data_{output_name}.npy'),y_data)
    else:
        return x_data,y_data,skeletons
