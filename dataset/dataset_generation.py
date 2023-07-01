import os
import argparse
import numpy as np

# in order to get access to functions from other parent folder
# add the current path into system variable
# then current path inside this script is root path
from inspect import getsourcefile
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from util.utils import output_dataset,get_yaml


parser = argparse.ArgumentParser(description='Visualize original csv data')
parser.add_argument('--data_path', type=str, default='dataset/chor2_20230609')
# parser.add_argument('--data_path', type=str, default='dataset/testset_20230627')
parser.add_argument('--output_name', type=str, default='UpperBody')
parser.add_argument('--desired_features', type=str, 
                    default='dataset/yaml/desired_features.yaml', help='load features name from .yaml')
args = parser.parse_args([])

if __name__ == '__main__':

    input_path = os.path.join(args.data_path,'unknown.NoHead.csv')
    dists,angles = get_yaml(args.desired_features)
    output_dataset(input_path,dists,angles,output_name=args.output_name, output_npy=True)