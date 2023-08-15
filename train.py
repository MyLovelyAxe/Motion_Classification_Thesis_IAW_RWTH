import argparse
from model.ML_models import KNN,RandomForest,SVM
from util.utils import get_paths

def default_args():

    parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

    ###### datasets parameters ######
    parser.add_argument('--exp_group',type=str,default='Dynamic',
                        choices=['Dynamic','Agree','Static'],
                        help='Select one group of training & testing')
    parser.add_argument('--train_split_method_paths', type=list, help='paths of split methods for trainset')
    parser.add_argument('--trainset_paths', type=list, help='paths of data for trainset')
    parser.add_argument('--test_split_method_paths', type=list, help='paths of split methods for testset')
    parser.add_argument('--testset_paths', type=list, help='paths of data for testset')
    
    ###### training configuration ######
    parser.add_argument('--desired_features',type=str,default='dataset/desired_features.yaml',help='load features name from .yaml')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='the ratio for number of samples in trainset')
    parser.add_argument('--window_size', type=int, default=100, help='the ratio for number of samples in trainset')
    parser.add_argument('--outside_test',type=int,default=1,help='1: use extra testset; 0: extract testset from trainset')
    parser.add_argument('--save_res',type=int,default=1,help='True: save plot; False: show plot')
    parser.add_argument('--standard', type=str, default='height',
                         choices={'len_spine','height','len_spine_rate','height_rate','no_scale'},
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

    ### create model
    if args.model == 'KNN':
        cls_model = KNN(args)

    elif args.model == 'RandomForest':
        cls_model = RandomForest(args)

    elif args.model == 'SVM':
        cls_model = SVM(args)
        
    ### train & test & show result
    cls_model.train()
    cls_model.test()
    print(f'Result on {args.model}:')
    cls_model.show_result(args)

    return cls_model

if __name__ == '__main__':

    cls_model = main()
