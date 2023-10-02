import os
import pickle
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    for shot_path in set_path_lst:
        split_method_paths.append(os.path.join(set_path,shot_path,'split_method.yaml'))
        data_paths.append(os.path.join(set_path,shot_path,'unknown.NoHead.csv'))
    return split_method_paths,data_paths

def get_paths(args):
    """
    One group of experiment consists of:
        - trainset from N .csv files
        - testset from M .csv files
        (N>=1, M>=1)
    """
    if not args.cross_test:
        args.test_exp_group = args.train_exp_group
    train_group_path = os.path.join('dataset',args.train_exp_group)
    test_group_path = os.path.join('dataset',args.test_exp_group)
    # trainset
    args.train_split_method_paths,args.trainset_paths = extract_path(group_path=train_group_path,train_or_test='trainset')
    # testset
    args.test_split_method_paths,args.testset_paths = extract_path(group_path=test_group_path,train_or_test='testset')
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

def single_splilt_method(yaml_path,show=False):
    """
    load split_mehthods.yaml as dict, containing starting frame, end frame, label for each shot
    """
    with open(yaml_path, "r") as file:
        split_method = yaml.safe_load(file)
    if show:
        print(f'split method: {split_method}')
    return split_method

def get_split_methods(split_method_paths):
    """
    get general split methods for trainset or testset
    """
    AccFrame = 0 # accumulated counts
    general_split_methods = {}
    ### iterate all split_methods.yaml
    for split_path in split_method_paths:
        ### save split methods
        ori_split_method = single_splilt_method(split_path)
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
    return general_split_methods

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
    x_data_lst = []
    y_data_lst = []
    skeleton_lst = []
    ### iterate all split_methods.yaml
    for split_path,data_path in zip(split_method_paths,ori_data_paths):
        ### calculate features
        ori_split_method = single_splilt_method(split_path)
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

    return x_data,y_data,skeletons


##########################
###### save results ######
##########################

def get_output_name(args):
    """
    output name for folders containing corresponding model and args.yaml
    """
    ### load trained model or not
    if args.load_model is None:
        output_name = f'{args.start_time}-Train_{args.train_exp_group}-Test_{args.test_exp_group}-{args.model}-wl{args.window_size}'
        if args.model == 'KNN':
            output_folder = output_name + f"-NNeighbor{args.n_neighbor}"
        elif args.model == 'RandomForest':
            output_folder = output_name + f"-MaxDepth{args.max_depth}-RandomState{args.random_state}"
        elif args.model == 'SVM':
            output_folder = output_name
    else:
        output_folder = '{}-load:{}-test_{}'.format(args.start_time,args.load_model.split('/')[1],args.test_exp_group)
    return output_folder

def save_model(save_path, model):
    """
    save the trained model as .pickle for cross testing
    """
    pickle.dump(model, open(os.path.join(save_path,f'model.pickle'), "wb"))

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

def save_plot(save_path, args, acc, plot_pred,plot_truth,actLabel_actName_dict):
    """
    save performance of current experiment as plot for prediction and target
    """
    sample_numbers = np.arange(plot_pred.shape[0])#/60 # frame_rate=60
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,13))
    for idx,(actLabel,_) in enumerate(actLabel_actName_dict.items()):
        ax1.plot(sample_numbers, plot_pred[:, idx], label=f'{actLabel_actName_dict[actLabel]}')
        truth = np.where(plot_truth==actLabel,1,0)
        ax2.plot(sample_numbers, truth, label=f'{actLabel_actName_dict[actLabel]}')
    ax1.set_title(f'Result of classification',fontsize=20)
    ax1.set_ylabel(f'Probability of Classification',fontsize=20)
    ax2.set_title(f'Ground truth',fontsize=20)
    ax2.set_xlabel(f'Windows',fontsize=20)
    ax2.set_ylabel(f'Probability of Classification',fontsize=20)
    plt.legend(fontsize=15)
    if args.load_model is None:
        output_image = f"{args.start_time}-Train_{args.train_exp_group}-Test_{args.test_exp_group}-{args.model}-wl{args.window_size}-Acc{round(acc, 3)}.png"
    else:
        output_image = '{}-load[{}]-test_{}-Acc{}.png'.format(args.start_time,args.load_model.split('/')[1],args.test_exp_group,round(acc, 3))
    plt.savefig(os.path.join(save_path,output_image))

def save_miscls_index(save_path,
                      miscls_win_index,
                      examine_frame_index,
                      true_labels,
                      pred_labels):
    """
    record the misclassified index of window and corresponding frames, examine with data_visualization.py
    """
    file_name = os.path.join(save_path,'miscls_index.txt')
    with open(file_name, 'w') as file:
        file.write(f'idx of misclassified window | check on dataset with:[start_frame, end_frame] | truth | prediction' + '\n')
        for mis_idx,exm_idxs,tru,pre in zip(miscls_win_index,examine_frame_index,true_labels,pred_labels):
            file.write(f'{mis_idx} | {exm_idxs} | {tru} | {pre}' + '\n')

def save_result(args,
                model,
                acc,
                plot_pred,
                plot_truth,
                actLabel_actName_dict,
                miscls_win_index,
                examine_frame_index,
                true_labels,
                pred_labels):
    """
    save result of current experiment: trained model, args config, performance plot
    """
    # define output folder
    save_path = 'save'
    os.makedirs(save_path, exist_ok=True)
    output_folder = get_output_name(args)
    save_path = os.path.join(save_path,output_folder)
    os.makedirs(save_path, exist_ok=True)
    # save results
    save_model(save_path,model)
    save_config(save_path,args)
    save_plot(save_path,args,acc,plot_pred,plot_truth,actLabel_actName_dict)
    save_miscls_index(save_path,miscls_win_index,examine_frame_index,true_labels,pred_labels)

#################################
###### load trained models ######
#################################

def load_config(args):
    """
    load the configuration of the experiments for trained model,
    make sure the cross-tested experiment have the same configuration
    """
    # e.g. model_path = 'save/06_Sep_20_30-Agree-KNN-wl5-NNeighbor20'
    yaml_path = os.path.join(args.load_model,f'args.yaml')
    with open(yaml_path, "r") as file:
        features = yaml.safe_load(file)
    args.cross_test = features['cross_test']
    args.train_exp_group = features['train_exp_group']
    args.desired_features = features['desired_features']
    args.window_size = features['window_size']
    args.standard = features['standard']
    args.model = features['model']
    args.n_neighbor = features['n_neighbor']
    args.max_depth = features['max_depth']
    args.random_state = features['random_state']
    return args
