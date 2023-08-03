import argparse
from model.static_models import KNN,RandomForest,SVM

parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

###### datasets parameters ######
parser.add_argument('--train_split_method_paths', type=str,nargs='+',
                    default=['dataset/chor2_20230609/split_method.yaml'],
                    help='split method for extracting labels and names of activities')
parser.add_argument('--trainset_path',type=str,nargs='+',
                    default=['dataset/chor2_20230609/x_data_UpperLowerBody.npy',
                             'dataset/chor2_20230609/y_data_UpperLowerBody.npy'],
                    help='path of training dataset')

parser.add_argument('--test_split_method_paths', type=str,nargs='+',
                    default=['dataset/testset_20230627/split_method.yaml'],
                    help='split method for extracting labels and names of activities')
parser.add_argument('--testset_path',type=str,nargs='+',
                    default=['dataset/testset_20230627/x_data_UpperLowerBody.npy',
                             'dataset/testset_20230627/y_data_UpperLowerBody.npy'],
                    help='path of extra testing dataset from outside')
parser.add_argument('--train_len', type=int, default=10000, help='length of train set')
parser.add_argument('--test_len', type=int, default=100, help='length of test set, only useful when there is no outside testset')

parser.add_argument('--outside_test',type=int,default=1,help='1: use extra testset; 0: extract testset from trainset')
parser.add_argument('--save_res',type=int,default=1,help='True: save plot; False: show plot')

###### models configuration ######
# select a model
parser.add_argument('--model', type=str, default='SVM', choices=['KNN','RandomForest','SVM'])
# for KNN
parser.add_argument('--n_neighbor', type=int, default=20, help='number of neighbours, only for KNN')
# for RandomForest
parser.add_argument('--max_depth', type=int, default=6, help='max depth for random forest')
parser.add_argument('--random_state', type=int, default=0, help='random state for random forest')

args = parser.parse_args()

if __name__ == '__main__':

    ### whether use testset outside from trainset or not
    if not args.outside_test:
        args.test_split_method_paths = None
        args.testset_path = None

    ### create model
    if args.model == 'KNN':
        cls_model = KNN(N_neighbor=args.n_neighbor,
                        Train_Len=args.train_len,
                        Test_Len=args.test_len,
                        Train_Split_Method_Paths=args.train_split_method_paths,
                        Trainset_Path=args.trainset_path,
                        Test_Split_Method_Paths=args.test_split_method_paths,
                        Testset_Path=args.testset_path
                        )
    elif args.model == 'RandomForest':
        cls_model = RandomForest(Max_Depth=args.max_depth,
                                 Random_State=args.random_state,
                                 Train_Len=args.train_len,
                                 Test_Len=args.test_len,
                                 Train_Split_Method_Paths=args.train_split_method_paths,
                                 Trainset_Path=args.trainset_path,
                                 Test_Split_Method_Paths=args.test_split_method_paths,
                                 Testset_Path=args.testset_path
                                 )
    elif args.model == 'SVM':
        cls_model = SVM(Train_Len=args.train_len,
                        Test_Len=args.test_len,
                        Train_Split_Method_Paths=args.train_split_method_paths,
                        Trainset_Path=args.trainset_path,
                        Test_Split_Method_Paths=args.test_split_method_paths,
                        Testset_Path=args.testset_path
                        )
        
    ### train & test & show result
    cls_model.train()
    cls_model.test()
    print(f'Result on {args.model}:')
    cls_model.show_result(args)
