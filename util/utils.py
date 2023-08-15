import re
import os
import yaml
import pandas as pd
import numpy as np
from util.features import get_all_features,get_ActName

def extract_path(group_path,train_or_test):
    """
    param:
        group_path: string, the root path of current group of experiment
        train_or_test: string, 'train' or 'test'
    """
    # trainset
    set_path = os.path.join(group_path,train_or_test)
    set_path_lst = os.listdir(set_path)
    set_path_lst.sort()
    split_method_paths = []
    data_paths = []
    for train_path in set_path_lst:
        split_method_paths.append(os.path.join(set_path,train_path,'split_method.yaml'))
        data_paths.append(os.path.join(set_path,train_path,'unknown.NoHead.csv'))
    return split_method_paths,data_paths

def get_paths(args):
    """
    One group of experiment consists of:
        - trainset from N .csv files
        - testset from M .csv files
        (N>=1, M>=1)
    """

    group_path = os.path.join('dataset',args.exp_group)
    # trainset
    args.train_split_method_paths,args.trainset_paths = extract_path(group_path=group_path,train_or_test='trainset')
    # testset
    args.test_split_method_paths,args.testset_paths = extract_path(group_path=group_path,train_or_test='testset')
    # whether use testset outside from trainset or not
    if not args.outside_test:
        args.test_split_method_paths = None
        args.testset_paths = None

    return args

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
                   standard):
    out_dict = {}
    AccCount = 0 # accumulated counts
    for split_path,data_path in zip(split_method_paths,ori_data_paths):
        _,coords = get_ori_data(data_path)
        split_method = get_splilt_method(split_path)
        all_features = get_all_features(coords,desired_dists,desired_angles,standard)
        # e.g. dynamic split_method = {'Boxing1': {'start': 200, 'end': 3700, 'label': 1}}
        for act_name,config in split_method.items():
            start,end,label = list(i for _,i in config.items())
            act_name = get_ActName(act_name)
            if not act_name in out_dict:
                out_dict[act_name] = {'x_data':[],'y_data':[],'y_ori_idx':[],'skeleton':[]}
            out_dict[act_name]['x_data'].append(all_features[start:end])
            out_dict[act_name]['y_data'].append(np.full((end-start),label))
            out_dict[act_name]['y_ori_idx'].append(np.arange(start+AccCount,end+AccCount))
            out_dict[act_name]['skeleton'].append(coords[start:end])
        AccCount += len(coords)
    # concatenate all activities
    x_data_lst = []
    y_data_lst = []
    y_ori_idx_lst = []
    skeletons_lst = []
    for act,data in out_dict.items():
        x_data_tmp = np.concatenate(data['x_data'],axis=0)
        y_data_tmp = np.concatenate(data['y_data'],axis=0)
        y_ori_idx_tmp = np.concatenate(data['y_ori_idx'],axis=0)
        skeleton_tmp = np.concatenate(data['skeleton'],axis=0)
        x_data_lst.append(x_data_tmp)
        y_data_lst.append(y_data_tmp)
        y_ori_idx_lst.append(y_ori_idx_tmp)
        skeletons_lst.append(skeleton_tmp)
    x_data = np.concatenate(x_data_lst,axis=0)
    y_data = np.concatenate(y_data_lst,axis=0)
    y_ori_idx = np.concatenate(y_ori_idx_lst,axis=0)
    skeletons = np.concatenate(skeletons_lst,axis=0)
    del x_data_lst,y_data_lst,y_ori_idx_lst,skeletons_lst,x_data_tmp,y_data_tmp,y_ori_idx_tmp,skeleton_tmp,out_dict

    return x_data,y_data,skeletons,y_ori_idx
