import os
import argparse
import numpy as np
from util.plots import plot_ori_data,verification,output_dataset,get_feature_selection,show_misclassified_frames

############################# Attention ###########################
##
## Please edit these arguments in order:
##
##      1. if --function is check_ori_data or verify_before_output
##              edit --single_data_path
##              edit --start_frame
##              edit --end_frame
##
##      2. if --function is verify_npy
##              edit --source_data_path
##              edit --npy_root
##
##      3. if --function is post_process
##              edit --examine_data_path
##              edit --misclassified_frames
##
###################################################################


####################################
###### get general parameters ######
####################################

parser = argparse.ArgumentParser(description='Visualize original csv data')

parser.add_argument('--function', type=str,
                    default='check_ori_data',
                    help='check_ori_data: visualize original data from Captury Live; \
                          verify_before_output: verify dataset before output into .npy files; \
                          verify_npy: verify the existed .npy files which have been already output',
                    choices=['check_ori_data','verify_before_output','verify_npy','post_process']
                    )

##############################################################
###### function: check_ori_data or verify_before_output ######
##############################################################

parser.add_argument('--single_data_path', type=str,
                    default='dataset/dynamic1_20230706',
                    help='only one single dataset for function check_ori_data and verify_before_output',
                    choices=['dataset/chor2_20230609',
                             'dataset/testset_20230627',
                             'dataset/dynamic1_20230706',
                             'dataset/dynamic2_20230706',
                             'dataset/dynamic3_20230706',
                             'dataset/dynamic_test_20230801',
                             'dataset/agree_20230801',
                             'dataset/agree_test_20230801']
                    )
parser.add_argument('--start_frame', type=int, default=0, help='from which frame to start visualize')
parser.add_argument('--end_frame', type=int, default=300, help='to which frame to end visualize')
parser.add_argument('--wl', type=int, default=51, help='window length for dataset creation, make it as odd number')
parser.add_argument('--output_anim', type=bool, default=False, help='whether to output animation of visualization')
parser.add_argument('--desired_features_trial', type=str, 
                    default='dataset/desired_features_trial.yaml', help='load features name from .yaml')

##################################
###### function: verify_npy ######
##################################

parser.add_argument('--source_data_path', type=str,nargs="+",
                    default=['dataset/agree_test_20230801'],
                    help='original source of generated dataset for function verify_npy',
                    choices=['dataset/chor2_20230609',
                             'dataset/testset_20230627',
                             'dataset/dynamic1_20230706',
                             'dataset/dynamic2_20230706',
                             'dataset/dynamic3_20230706',
                             'dataset/dynamic_test_20230801',
                             'dataset/agree_20230801',
                             'dataset/agree_test_20230801']
                             )
parser.add_argument('--npy_root', type=str, default='dataset/agree_test_20230801')
parser.add_argument('--desired_features', type=str, 
                    default='dataset/desired_features.yaml', help='load features name from .yaml')

####################################
###### function: post_process ######
####################################

parser.add_argument('--examine_data_path', type=str,nargs="+",
                    default=['dataset/testset_20230627'],
                    help='location of examined data for misclassified labels, only external testset',
                    choices=['dataset/testset_20230627',
                             'dataset/dynamic_test_20230801',
                             'dataset/agree_test_20230801']
                             )
parser.add_argument('--misclassified_frames', type=list,default=[961,961+1],help='check misclassified frames')

args = parser.parse_args()

if __name__ == '__main__':

    if args.function == 'check_ori_data':

        input_path = os.path.join(args.single_data_path,'unknown.NoHead.csv')
        plot_ori_data(data_path=args.single_data_path,
                      start_frame=args.start_frame,
                      end_frame=args.end_frame,
                      output_anim=args.output_anim)

    if args.function == 'verify_before_output':

        input_paths = os.path.join(args.single_data_path,'unknown.NoHead.csv')
        split_method_paths = os.path.join(args.single_data_path,'split_method.yaml')
        verification([input_paths],args.desired_features_trial,[split_method_paths],win_len=args.wl)
        # verification([input_paths],args.desired_features_trial,[split_method_paths])

    if args.function == 'verify_npy':

        input_paths = [os.path.join(data_path,'unknown.NoHead.csv') for data_path in args.source_data_path]
        split_method_paths = [os.path.join(data_path,'split_method.yaml') for data_path in args.source_data_path]
        npy_paths = [os.path.join(args.npy_root,'x_data_UpperLowerBody.npy'),os.path.join(args.npy_root,'y_data_UpperLowerBody.npy')]
        verification(input_paths,args.desired_features,split_method_paths,npy_path=npy_paths,win_len=args.wl)

    if args.function == 'post_process':

        input_paths = [os.path.join(data_path,'unknown.NoHead.csv') for data_path in args.examine_data_path]
        split_method_paths = [os.path.join(data_path,'split_method.yaml') for data_path in args.examine_data_path]
        dists,angles = get_feature_selection(args.desired_features)

        _,_,skeleton,_ = output_dataset(ori_data_paths=input_paths,
                                        desired_dists=dists,
                                        desired_angles=angles,
                                        split_method_paths=split_method_paths)
        
        show_misclassified_frames(skeleton,args.misclassified_frames)
