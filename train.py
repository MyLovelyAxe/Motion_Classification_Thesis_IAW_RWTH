from datetime import datetime as dt
import argparse
from model.ML_models import Exp
from util.utils import get_paths

################################# Instruction for usage ######################################
#
# Edit the following arguments to define training configuration:
#
#    1. cross_test:
#          True:
#              then define both train_exp_group and test_exp_group
#          False:
#              then only define train_exp_group
#
#    2. window_size:
#          how many frames are in a window
#
#    3. model:
#          select one model for classification: KNN, Random Forest, SVM
#
#    4. parameters for corresponding model:
#          see details in "help" of arguments of parameters
#
################################# Instruction for usage ######################################

def default_args():

    parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

    ###### select exp_group ######
    parser.add_argument('--cross_test', type=bool, default=True,
                        help='True: train with user1 trainset, test with user2 testset; False: train and test with data of same user')
    parser.add_argument('--train_exp_group',type=str,default='Static_User1',
                        choices=['Dynamic_User1','Dynamic_User2',
                                 'Static_User1','Static_User2'],
                        help='Select one group of training')
    parser.add_argument('--test_exp_group',type=str,default='Static_User2',
                        choices=['Dynamic_User1','Dynamic_User2',
                                 'Static_User1','Static_User2'],
                        help='Select one group of testing')
    
    ###### path of data and config files ######
    parser.add_argument('--train_split_method_paths', type=list, help='paths of split methods for trainset')
    parser.add_argument('--trainset_paths', type=list, help='paths of data for trainset')
    parser.add_argument('--test_split_method_paths', type=list, help='paths of split methods for testset')
    parser.add_argument('--testset_paths', type=list, help='paths of data for testset')
    
    ###### training configuration ######
    parser.add_argument('--desired_features',type=str,default='config/desired_features.yaml',help='load features name from .yaml')
    parser.add_argument('--window_size', type=int, default=100, help='the ratio for number of samples in trainset')
    parser.add_argument('--start_time',type=str,help='starting time of current process')
    parser.add_argument('--load_model',type=str,default=None,help='path of loaded model')
    parser.add_argument('--standard', type=str, default='no_scale',
                         choices={'len_spine','neck_height','len_arm','len_shoulder','no_scale'},
                         help='standarize with which scaling factor')

    ###### models configuration ######
    # select a model
    parser.add_argument('--model', type=str, default='RandomForest', choices=['KNN','RandomForest','SVM'])
    # for KNN
    parser.add_argument('--n_neighbor', type=int, default=20, help='number of neighbours, only for KNN')
    # for RandomForest
    parser.add_argument('--max_depth', type=int, default=6, help='max depth for random forest')
    parser.add_argument('--random_state', type=int, default=0, help='random state for random forest')

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
    args = get_paths(args)
    ### get time
    args.start_time = f"{dt.now().strftime('%d_%h_%H_%M')}"
    ### create model
    exp_model = Exp(args)
    ### train & test
    exp_model.train()
    exp_model.test()
    ### save results
    print(f'Result on {args.model}:')
    exp_model.result(args)
    return exp_model

if __name__ == '__main__':

    exp_model = main()
