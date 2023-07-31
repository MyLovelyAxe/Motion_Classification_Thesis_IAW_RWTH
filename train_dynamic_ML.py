import argparse
from model.dynamic_models import KNN,RandomForest,SVM


parser = argparse.ArgumentParser(description='Machine learning method on classification of human activities from skeleton data')

###### datasets parameters ######
parser.add_argument('--trainset_path',type=str,nargs='+',
                    default=['dataset/dynamic_dataset/x_data_UpperLowerBody.npy',
                             'dataset/dynamic_dataset/y_data_UpperLowerBody.npy'],
                    help='path of training dataset')
parser.add_argument('--testset_path',type=str,nargs='+',
                    default=['dataset/testset_20230627/x_data_UpperLowerBody.npy',
                             'dataset/testset_20230627/y_data_UpperLowerBody.npy'],
                    help='path of extra testing dataset from outside')
parser.add_argument('--split_method_paths', type=str,nargs='+',
                    default=['dataset/dynamic1_20230706/split_method.yaml',
                             'dataset/dynamic2_20230706/split_method.yaml',
                             'dataset/dynamic3_20230706/split_method.yaml'],
                    help='split method for extracting labels and names of activities')
parser.add_argument('--split_ratio', type=float, default=0.8, help='the ratio for number of samples in trainset')
parser.add_argument('--window_size', type=int, default=250, help='the ratio for number of samples in trainset')

###### models configuration ######
# select a model
parser.add_argument('--model', type=str, default='RandomForest', choices=['KNN','RandomForest','SVM'])
# for KNN
parser.add_argument('--n_neighbor', type=int, default=20, help='number of neighbours, only for KNN')
# for RandomForest
parser.add_argument('--max_depth', type=int, default=6, help='max depth for random forest')
parser.add_argument('--random_state', type=int, default=0, help='random state for random forest')

args = parser.parse_args()

if __name__ == '__main__':

    if args.model == 'KNN':
        cls_model = KNN(N_neighbor=args.n_neighbor,
                        Window_Size=args.window_size,
                        Split_Method_Paths=args.split_method_paths,
                        Trainset_Path=args.trainset_path,
                        Testset_Path=None,
                        Split_Ratio=args.split_ratio
                        )
    elif args.model == 'RandomForest':
        cls_model = RandomForest(Max_Depth=args.max_depth,
                                 Random_State=args.random_state,
                                 Window_Size=args.window_size,
                                 Split_Method_Paths=args.split_method_paths,
                                 Trainset_Path=args.trainset_path,
                                 Testset_Path=None,
                                 Split_Ratio=args.split_ratio
                                 )
    elif args.model == 'SVM':
        cls_model = SVM(Window_Size=args.window_size,
                        Split_Method_Paths=args.split_method_paths,
                        Trainset_Path=args.trainset_path,
                        Testset_Path=None,
                        Split_Ratio=args.split_ratio
                        )
    cls_model.train()
    cls_model.test()
    print(f'Result on {args.model}:')
    cls_model.show_result(args.model,save=False)
