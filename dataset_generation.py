# in order to get access to functions from other parent folder
# add the current path into system variable
# then current path inside this script is root path
# from inspect import getsourcefile
# import os
# import sys
# import argparse
# current_path = os.path.abspath(getsourcefile(lambda:0))
# current_dir = os.path.dirname(current_path)
# parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
# sys.path.insert(0, parent_dir)

import os
import argparse
from util.utils import get_feature_selection,output_dataset

parser = argparse.ArgumentParser(description='generate dataset for training')

################# Attention #################
##
## Please edit these arguments in order:
##
##      1. if --type is 'static'
##          edit --static_data_path
##          edit --static_output_path
##
##      2. if --type is 'dynamic'
##          edit --dynamic_data_path
##          edit --dynamic_output_path
##
## Attention:
##
##      Both static and dynamic datasets can
##      be multiple and insert together
#############################################


parser.add_argument('--type',type=str,default='static',choices=['static','dynamic'])
# static dataset
parser.add_argument('--static_data_path',type=str,nargs="+",
                    help='the list containing all static dataset to be merged',
                    choices=['dataset/testset_20230627',
                             'dataset/chor2_20230609'
                             ],
                    default=['dataset/testset_20230627',
                             ])
parser.add_argument('--static_output_path',type=str,default='dataset/testset_20230627')
# dynamic dataset
parser.add_argument('--dynamic_data_path',type=str,nargs="+",
                    help='the list containing all dynamic dataset to be merged',
                    choices=['dataset/dynamic1_20230706',
                             'dataset/dynamic2_20230706',
                             'dataset/dynamic3_20230706',
                             ],
                    default=['dataset/dynamic1_20230706',
                             'dataset/dynamic2_20230706',
                             'dataset/dynamic3_20230706',
                             ])
parser.add_argument('--dynamic_output_path',type=str,default='dataset/dynamic_dataset')
# common parameters
parser.add_argument('--output_name',type=str,default='UpperLowerBody')
parser.add_argument('--desired_features',type=str, 
                    default='dataset/desired_features.yaml',help='load features name from .yaml')

# args = parser.parse_args([])
args = parser.parse_args()

if __name__ == '__main__':

    if args.type == 'static':

        input_paths = [os.path.join(data_path,'unknown.NoHead.csv') for data_path in args.static_data_path]
        split_method_paths = [os.path.join(data_path,'split_method.yaml') for data_path in args.static_data_path]
        dists,angles = get_feature_selection(args.desired_features)

        output_dataset(ori_data_paths=input_paths,
                       split_method_paths=split_method_paths,
                       desired_dists=dists,
                       desired_angles=angles,
                       output_path=args.static_output_path,
                       output_name=args.output_name)
        
    elif args.type == 'dynamic':

        input_paths = [os.path.join(data_path,'unknown.NoHead.csv') for data_path in args.dynamic_data_path]
        split_method_paths = [os.path.join(data_path,'split_method.yaml') for data_path in args.dynamic_data_path]
        dists,angles = get_feature_selection(args.desired_features)

        output_dataset(ori_data_paths=input_paths,
                       desired_dists=dists,
                       desired_angles=angles,
                       split_method_paths=split_method_paths,
                       output_path=args.dynamic_output_path,
                       output_name=args.output_name)
