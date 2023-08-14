import argparse
from model.ML_models import KNN,RandomForest,SVM
from util.utils import get_paths

def default_args():

    parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

    ###### datasets parameters ######
    parser.add_argument('--exp_group',type=str,default='Agree',
                        choices=['Dynamic','Agree','Static'],
                        help='Select one group of training & testing')
    parser.add_argument('--desired_features',type=str,default='dataset/desired_features.yaml',help='load features name from .yaml')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='the ratio for number of samples in trainset')
    parser.add_argument('--window_size', type=int, default=100, help='the ratio for number of samples in trainset')
    parser.add_argument('--outside_test',type=int,default=1,help='1: use extra testset; 0: extract testset from trainset')
    parser.add_argument('--save_res',type=int,default=1,help='True: save plot; False: show plot')

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
    train_split_method_paths,trainset_paths,test_split_method_paths,testset_paths = get_paths(args.exp_group,args.outside_test)

    ### create model
    if args.model == 'KNN':
        cls_model = KNN(N_neighbor=args.n_neighbor,
                        Window_Size=args.window_size,
                        Train_Split_Method_Paths=train_split_method_paths,
                        Trainset_Paths=trainset_paths,
                        Test_Split_Method_Paths=test_split_method_paths,
                        Testset_Paths=testset_paths,
                        Split_Ratio=args.split_ratio,
                        Desired_Features=args.desired_features
                        )
    elif args.model == 'RandomForest':
        cls_model = RandomForest(Max_Depth=args.max_depth,
                                 Random_State=args.random_state,
                                 Window_Size=args.window_size,
                                 Train_Split_Method_Paths=train_split_method_paths,
                                 Trainset_Paths=trainset_paths,
                                 Test_Split_Method_Paths=test_split_method_paths,
                                 Testset_Paths=testset_paths,
                                 Split_Ratio=args.split_ratio,
                                 Desired_Features=args.desired_features
                                 )
    elif args.model == 'SVM':
        cls_model = SVM(Window_Size=args.window_size,
                        Train_Split_Method_Paths=train_split_method_paths,
                        Trainset_Paths=trainset_paths,
                        Test_Split_Method_Paths=test_split_method_paths,
                        Testset_Paths=testset_paths,
                        Split_Ratio=args.split_ratio,
                        Desired_Features=args.desired_features
                        )
        
    ### train & test & show result
    cls_model.train()
    cls_model.test()
    print(f'Result on {args.model}:')
    cls_model.show_result(args)

    return cls_model

if __name__ == '__main__':

    cls_model = main()
