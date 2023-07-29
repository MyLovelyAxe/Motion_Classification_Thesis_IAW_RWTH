import argparse
from model.static_models import KNN,RandomForest,SVM


parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

###### datasets parameters ######
parser.add_argument('--trainset_path',type=list,
                    default=['dataset/chor2_20230609/x_data_UpperLowerBody.npy',
                             'dataset/chor2_20230609/y_data_UpperLowerBody.npy'],
                    help='path of training dataset')
parser.add_argument('--testset_path',type=list,
                    default=['dataset/testset_20230627/x_data_UpperLowerBody.npy',
                             'dataset/testset_20230627/y_data_UpperLowerBody.npy'],
                    help='path of extra testing dataset from outside')
parser.add_argument('--train_len', type=int, default=10000, help='length of train set')
parser.add_argument('--test_len', type=int, default=100, help='length of test set, only useful when there is no outside testset')

###### models configuration ######
# select a model
parser.add_argument('--model', type=str, default='RandomForest', choices=['KNN','RandomForest','SVM'])
# for KNN
parser.add_argument('--n_neighbor', type=int, default=1, help='number of neighbours, only for KNN')
# for RandomForest
parser.add_argument('--max_depth', type=int, default=2, help='max depth for random forest')
parser.add_argument('--random_state', type=int, default=0, help='random state for random forest')

args = parser.parse_args([])

if __name__ == '__main__':

    if args.model == 'KNN':
        cls_model = KNN(N_neighbor=args.n_neighbor,
                        Train_Len=args.train_len,
                        Test_Len=args.test_len,
                        Trainset_Path=args.trainset_path,
                        Testset_Path=args.testset_path
                        )
    elif args.model == 'RandomForest':
        cls_model = RandomForest(Max_Depth=args.max_depth,
                                 Random_State=args.random_state,
                                 Train_Len=args.train_len,
                                 Test_Len=args.test_len,
                                 Trainset_Path=args.trainset_path,
                                #  Testset_Path=args.testset_path
                                 )
    elif args.model == 'SVM':
        cls_model = SVM(Train_Len=args.train_len,
                        Test_Len=args.test_len,
                        Trainset_Path=args.trainset_path,
                        Testset_Path=args.testset_path
                        )
    cls_model.train()
    cls_model.test()
    print(f'Result on {args.model}:')
    cls_model.show_result()
