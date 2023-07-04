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

from util.utils import output_dataset,get_feature_selection

parser = argparse.ArgumentParser(description='Visualize original csv data')
# parser.add_argument('--data_path', type=str, default='dataset/chor2_20230609')
parser.add_argument('--data_path', type=str, default='dataset/testset_20230627')
parser.add_argument('--output_name', type=str, default='UpperLowerBody')
parser.add_argument('--desired_features', type=str, 
                    default='dataset/desired_features.yaml', help='load features name from .yaml')
args = parser.parse_args([])

if __name__ == '__main__':

    input_path = os.path.join(args.data_path,'unknown.NoHead.csv')
    split_method_path = os.path.join(args.data_path,'split_method.yaml')
    dists,angles = get_feature_selection(args.desired_features)
    output_dataset(input_path,dists,angles,split_method_path,output_name=args.output_name,output_npy=True)
