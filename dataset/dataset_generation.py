# in order to get access to functions from other parent folder
# add the current path into system variable
# then current path inside this script is root path
from inspect import getsourcefile
import os
import sys
import argparse
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from util.utils import output_dataset_static,output_dataset_dynamic,get_feature_selection

### change:
#       --data_path
#       --output_path
#       --output_name
#       --type

parser = argparse.ArgumentParser(description='generate dataset for training')
# parser.add_argument('--data_path', type=str, default='dataset/chor2_20230609')
parser.add_argument('--data_path', type=str, default='dataset/testset_20230627')
parser.add_argument('--output_path', type=str, default='dataset/testset_20230627')
parser.add_argument('--output_name', type=str, default='UpperLowerBody')
parser.add_argument('--type', type=str, default='dynamic',choices=['static','dynamic'])
parser.add_argument('--desired_features', type=str, 
                    default='dataset/desired_features.yaml', help='load features name from .yaml')
args = parser.parse_args([])

if __name__ == '__main__':

    if args.type == 'static':

        input_paths = [os.path.join(args.data_path,'unknown.NoHead.csv')]
        split_method_paths = [os.path.join(args.data_path,'split_method.yaml')]
        dists,angles = get_feature_selection(args.desired_features)

        output_dataset_static(ori_data_paths=input_paths,
                              desired_dists=dists,
                              desized_angles=angles,
                              split_method_paths=split_method_paths,
                              output_path=args.output_path,
                              output_name=args.output_name)
        
    elif args.type == 'dynamic':

        input_paths = ['dataset/dynamic1_20230706/unknown.NoHead.csv',
                       'dataset/dynamic2_20230706/unknown.NoHead.csv',
                       'dataset/dynamic3_20230706/unknown.NoHead.csv']
        split_method_paths = ['dataset/dynamic1_20230706/split_method.yaml',
                              'dataset/dynamic2_20230706/split_method.yaml',
                              'dataset/dynamic3_20230706/split_method.yaml']
        dists,angles = get_feature_selection(args.desired_features)

        output_dataset_dynamic(ori_data_paths=input_paths,
                               desired_dists=dists,
                               desized_angles=angles,
                               split_method_paths=split_method_paths,
                               output_path='dataset/dynamic_dataset',
                               output_name='qqq')
