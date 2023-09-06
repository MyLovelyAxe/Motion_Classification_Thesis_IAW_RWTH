import pickle
from datetime import datetime as dt
import argparse
from model.ML_models import create_model
from util.utils import get_paths, load_config

def default_args():

    parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

    ###### datasets parameters ######
    parser.add_argument('--exp_group',type=str)
    parser.add_argument('--train_split_method_paths', type=list)
    parser.add_argument('--trainset_paths', type=list)
    parser.add_argument('--test_split_method_paths', type=list)
    parser.add_argument('--testset_paths', type=list)
    
    ###### training configuration ######
    parser.add_argument('--desired_features',type=str)
    parser.add_argument('--split_ratio', type=float)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--outside_test',type=int)
    parser.add_argument('--save_res',type=int)
    parser.add_argument('--loaded_model',type=int)
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
    parser.add_argument('--load_model', type=str, default='save/06_Sep_20_30-Dynamic_Apostolos-KNN-wl5-NNeighbor20')

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
    ### get time
    args.start_time = f"{dt.now().strftime('%d_%h_%H_%M')}"
    ### create model
    cls_model = create_model(args)
    ### test external model
    cls_model.train()
    cls_model.test()
    ### show results
    print(f'Result on {args.model}:')
    cls_model.show_result(args)
    return cls_model

if __name__ == '__main__':

    cls_model = main()
