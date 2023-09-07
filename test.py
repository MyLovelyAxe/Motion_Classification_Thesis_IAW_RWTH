import os
import pickle
from datetime import datetime as dt
import argparse
from model.ML_models import create_model
from util.utils import get_paths, load_config

def default_args():

    parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

    ###### datasets parameters ######
    parser.add_argument('--exp_group',type=str,default='Dynamic_Apostolos',
                        choices=['Dynamic','Agree','Static',
                                 'Dynamic_Jialei','Dynamic_Apostolos',
                                 'Static_Jialei','Static_Apostolos'],
                        help='Select one group of training & testing')
    parser.add_argument('--train_split_method_paths', type=list, help='paths of split methods for trainset')
    parser.add_argument('--trainset_paths', type=list, help='paths of data for trainset')
    parser.add_argument('--test_split_method_paths', type=list, help='paths of split methods for testset')
    parser.add_argument('--testset_paths', type=list, help='paths of data for testset')
    
    ###### training configuration ######
    parser.add_argument('--desired_features',type=str)
    parser.add_argument('--split_ratio', type=float)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--outside_test',type=int)
    parser.add_argument('--save_res',type=int)
    parser.add_argument('--start_time',type=str)
    parser.add_argument('--standard', type=str)

    ###### models configuration ######
    # select a model
    parser.add_argument('--model', type=str)
    # for KNN
    parser.add_argument('--n_neighbor', type=int)
    # for RandomForest
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--random_state', type=int)

    ###### load model ######
    parser.add_argument('--load_model', type=str, default='save/06_Sep_22_10-Dynamic_Jialei-RandomForest-wl100-MaxDepth6-RandomState0')

    args = parser.parse_args()
    return args

def main(ext_args=None):
    """
    ext_args: args
        if None: use default args
        if not None: use external args
    """
    ### get args
    if not ext_args:
        args = default_args()
    else: 
        args = ext_args
    ### get paths for current experiment
    args = load_config(args)
    args = get_paths(args)
    ### get time
    args.start_time = f"{dt.now().strftime('%d_%h_%H_%M')}"
    ### create model for new testset
    test_model = create_model(args)
    ### load trained model
    loaded_model = pickle.load(open(os.path.join(args.load_model,f'model.pickle'), "rb"))
    # replace testset of loaded model with the one of test_model
    loaded_model.dynamic_data.y_data_ori = test_model.dynamic_data.y_data_ori
    loaded_model.dynamic_data.x_test = test_model.dynamic_data.x_test
    loaded_model.dynamic_data.y_test = test_model.dynamic_data.y_test
    loaded_model.dynamic_data.y_MisClsExm = test_model.dynamic_data.y_MisClsExm
    loaded_model.dynamic_data.y_ori_idx_win = test_model.dynamic_data.y_ori_idx_win

    ### use loaded model to test new testset
    loaded_model.test()
    ### show results
    print(f'Result on {args.model}:')
    loaded_model.show_result(args)
    return loaded_model

if __name__ == '__main__':

    loaded_model = main()