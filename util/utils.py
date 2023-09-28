import os
import pickle
import yaml
import pandas as pd
import numpy as np
from util.features import get_all_features

##########################################
###### generate paths for exp_group ######
##########################################

def extract_path(group_path,train_or_test):
    """
    param:
        group_path: string, the root path of current group of experiment
        train_or_test: string, 'train' or 'test'
    """
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
    """
    get pre-selected features for creation of dataset, including distance and angle features
    """
    with open(yaml_path, "r") as file:
        features = yaml.safe_load(file)
    dists = features['desired_dists']
    angles = features['desired_angles']
    return dists,angles

def get_splilt_method(yaml_path,show=False):
    """
    load split_mehthods.yaml as dict, containing starting frame, end frame, label for each shot
    """
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
    """
    return:
        x_data:
            Frame Feature Array Arr_ff, containing features for each frame
        y_data:
            labels/classes for frames
        skeleton:
            Original Data Arr_ori, containig original coordinates for skeleton joints
        general_split_methods:
            split methods for all segments of recorded shot, for trainset or testset
    """
    AccFrame = 0 # accumulated counts
    x_data_lst = []
    y_data_lst = []
    skeleton_lst = []
    general_split_methods = {}
    ### iterate all split_methods.yaml
    for split_path,data_path in zip(split_method_paths,ori_data_paths):
        ### save split methods
        ori_split_method = get_splilt_method(split_path)
        # check if it is trainset or testset:
        # if not only 1 split methods, then it is trainset, accumulate corresponding number of frame
        if not len(split_method_paths) == 1:
            actName = list(ori_split_method.keys())[0]
            actConfig = list(ori_split_method.values())[0]
            current_split_method = {actName: {'start': actConfig['start']+AccFrame, 'end': actConfig['end']+AccFrame, 'label': actConfig['label']}}
            AccFrame += (actConfig['end'] - actConfig['start'])
        # if only 1 split methods, then it is testset, just save it
        else:
            current_split_method = ori_split_method
        general_split_methods.update(current_split_method)
        ### calculate features
        _,coords = get_ori_data(data_path)
        all_features = get_all_features(coords,desired_dists,desired_angles,standard)
        ### iterate all activities in current split_methods.yaml
        # e.g. dynamic split_method = {'actName': {'start': 200, 'end': 3700, 'label': 1}}
        for _,actConfig in ori_split_method.items():
            start,end,label = list(i for _,i in actConfig.items())
            x_data_lst.append(all_features[start:end])
            y_data_lst.append(np.full((end-start),label))
            skeleton_lst.append(coords[start:end])
    # concatenate all data
    x_data = np.concatenate(x_data_lst,axis=0)
    y_data = np.concatenate(y_data_lst,axis=0)
    skeletons = np.concatenate(skeleton_lst,axis=0)
    del x_data_lst,y_data_lst,skeleton_lst

    return x_data,y_data,skeletons,general_split_methods


################################
###### save & load models ######
################################

def get_output_name(args):
    """
    output name for folders containing corresponding model and args.yaml
    """
    output_name = f"{args.start_time}-{args.exp_group}-{args.model}-wl{args.window_size}"
    if args.model == 'KNN':
        output_folder = output_name + f"-NNeighbor{args.n_neighbor}"
    elif args.model == 'RandomForest':
        output_folder = output_name + f"-MaxDepth{args.max_depth}-RandomState{args.random_state}"
    elif args.model == 'SVM':
        output_folder = output_name
    return output_folder

def save_config(save_path, args):
    """
    save configuration of current experiment from args into .yaml
    """
    arg_dict = {}
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        arg_dict[arg_name] = arg_value
    save_path = os.path.join(save_path,f'args.yaml')
    with open(save_path, 'w') as yaml_file:
        yaml.dump(arg_dict, yaml_file, default_flow_style=False)

def save_model(args,model):
    """
    save the trained model as .pickle for cross testing
    """
    save_path = 'save'
    os.makedirs(save_path, exist_ok=True)
    output_folder = get_output_name(args)
    save_path = os.path.join(save_path,output_folder)
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(model.model, open(os.path.join(save_path,f'model.pickle'), "wb"))
    save_config(save_path, args)

def load_config(args):
    """
    load the configuration of the experiments for trained model,
    make sure the cross-tested experiment have the same configuration
    """
    # e.g. model_path = 'save/06_Sep_20_30-Agree-KNN-wl5-NNeighbor20'
    yaml_path = os.path.join(args.load_model,f'args.yaml')
    with open(yaml_path, "r") as file:
        features = yaml.safe_load(file)
    args.desired_features = features['desired_features']
    args.window_size = features['window_size']
    args.save_res = features['save_res']
    args.standard = features['standard']
    args.model = features['model']
    args.n_neighbor = features['n_neighbor']
    args.max_depth = features['max_depth']
    args.random_state = features['random_state']
    return args
