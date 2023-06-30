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

from util.utils import get_ori_data, get_dist_feature, get_angle_feature, get_all_features

####################################
###### get general parameters ######
####################################

parser = argparse.ArgumentParser(description='Visualize original csv data')

### general ###
parser.add_argument('--data_path', type=str, default='dataset/chor2_20230609')
# parser.add_argument('--data_path', type=str, default='dataset/testset_20230627')

### feature selection ###
parser.add_argument('--desired_dists', type=list,
                    default=['LHandEnd_head','LWrist_head','LElbow_head','LShoulder_head',
                             'RHandEnd_head','RWrist_head','RElbow_head','RShoulder_head']
                             )
parser.add_argument('--desized_angles', type=list,
                    default=['LHandEnd_LWrist_LElbow',
                             'LWrist_LElbow_LShoulder',
                             'LElbow_LShoulder_LClavicle',
                             'LShoulder_LClavicle_spine5']
                             )

args = parser.parse_args([])

if __name__ == '__main__':

    # get original data
    input_path = os.path.join(args.data_path,'unknown.NoHead.csv')
    cog,coords = get_ori_data(input_path)

    all_features = get_all_features(input_path,args.desired_dists,args.desized_angles)
    print(f'dataset shape: {all_features.shape}')