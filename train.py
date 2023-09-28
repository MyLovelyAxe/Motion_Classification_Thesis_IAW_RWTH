from datetime import datetime as dt
import argparse
from model.ML_models import DynamicClassModel
from util.utils import get_paths,save_model

######################################################################################
#
# in README.md:
#       note that in order to do NonCross experiment, just run experiment in train.py
#       the result is just trained and tested on the same exp_group
#
######################################################################################

######################################################################################
#
# 1. select training method: cross test or not
#       if cross_test is True:
#           then define both train_exp_group and test_exp_group
#       if cross_test is False:
#           then only define train_exp_group
#
######################################################################################

def default_args():

    parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

    ###### select exp_group ######
    parser.add_argument('--cross_test', type=bool, default=True,
                        help='True: train with user1 trainset, test with user2 testset; False: train and test with data of same user')
    parser.add_argument('--train_exp_group',type=str,default='Static_Jialei',
                        choices=['Dynamic','Agree','Static',
                                 'Dynamic_Jialei','Dynamic_Apostolos',
                                 'Static_Jialei','Static_Apostolos'],
                        help='Select one group of training & testing')
    parser.add_argument('--test_exp_group',type=str,default='Static_Apostolos',
                        choices=['Dynamic','Agree','Static',
                                 'Dynamic_Jialei','Dynamic_Apostolos',
                                 'Static_Jialei','Static_Apostolos'],
                        help='Select one group of training & testing')
    
    ###### path of data and config files ######
    parser.add_argument('--train_split_method_paths', type=list, help='paths of split methods for trainset')
    parser.add_argument('--trainset_paths', type=list, help='paths of data for trainset')
    parser.add_argument('--test_split_method_paths', type=list, help='paths of split methods for testset')
    parser.add_argument('--testset_paths', type=list, help='paths of data for testset')
    
    ###### training configuration ######
    parser.add_argument('--desired_features',type=str,default='config/desired_features.yaml',help='load features name from .yaml')
    parser.add_argument('--window_size', type=int, default=100, help='the ratio for number of samples in trainset')
    parser.add_argument('--save_res',type=int,default=1,help='1: save plot; 0: show plot')
    parser.add_argument('--save_model',type=int,default=1,help='1: save trained model; 0: not save model')
    parser.add_argument('--start_time',type=str,help='starting time of current process')
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
    cls_model = DynamicClassModel(args)
    ### train & test
    cls_model.train()
    cls_model.test()
    ### show results
    print(f'Result on {args.model}:')
    cls_model.show_result(args)
    return cls_model

if __name__ == '__main__':

    cls_model = main()
