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

from util.utils import output_dataset

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
                             'RHandEnd_head','RWrist_head','RElbow_head','RShoulder_head',
                             'LHandEnd_spine5','LWrist_spine5','LElbow_spine5','LShoulder_spine5',
                             'RHandEnd_spine5','RWrist_spine5','RElbow_spine5','RShoulder_spine5',
                             'LHandEnd_spine4','LWrist_spine4','LElbow_spine4','LShoulder_spine4',
                             'RHandEnd_spine4','RWrist_spine4','RElbow_spine4','RShoulder_spine4',
                             'LAnkle_spine1','LKnee_spine1','LHip_spine1',
                             'RAnkle_spine1','RKnee_spine1','RHip_spine1']
                             )
parser.add_argument('--desized_angles', type=list,
                    default=['LHandEnd_LWrist_LElbow',
                             'LWrist_LElbow_LShoulder',
                             'LElbow_LShoulder_LClavicle',
                             'LShoulder_LClavicle_spine5',
                             'LClavicle_spine5_spine4',
                             'LClavicle_spine5_head',
                             'RHandEnd_RWrist_RElbow',
                             'RWrist_RElbow_RShoulder',
                             'RElbow_RShoulder_RClavicle',
                             'RShoulder_RClavicle_spine5',
                             'RClavicle_spine5_spine4',
                             'RClavicle_spine5_head',
                             'spine3_spine4_spine5',
                             'spine4_spine5_head',
                             'LToe_LAnkle_LKnee',
                             'LAnkle_LKnee_LHip',
                             'LKnee_LHip_spine1',
                             'RToe_RAnkle_RKnee',
                             'RAnkle_RKnee_RHip',
                             'RKnee_RHip_spine1']
                             )

args = parser.parse_args([])

if __name__ == '__main__':

    input_path = os.path.join(args.data_path,'unknown.NoHead.csv')
    output_dataset(input_path,args.desired_dists,args.desized_angles,output_name='UpperBody', output_npy=True)