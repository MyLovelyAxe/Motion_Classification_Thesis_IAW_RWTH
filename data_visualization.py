import os
import argparse
from util.examine import plot_ori_data,verification,output_dataset,get_feature_selection,show_misclassified_frames

################################# Instruction for usage ######################################
#
# Please define argument --function for what you want to do and other arguments following:
#
#      1. check_ori_data:
#
#           function:
#               visualize Original Data Arr_ori
#           edit:
#               --single_data_path: which shot to examine
#               --start_frame:      from which frame
#               --end_frame:        to which frame
#
#      2. verify_before_output:
#
#           function:
#               examine Frame Feature Array Arr_ff for calculation error
#           edit:
#               --single_data_path: which shot to examine
#               --start_frame:      from which frame
#               --end_frame:        to which frame
#
#      3. post_process:
#
#           function:
#               check the window which is wrong classified
#           edit:
#               --examine_data_path:    which shot to examine
#               --misclassified_frames: index of frame in this window in miscls_index.txt
#               --end_frame:            to which frame
#
################################# Instruction for usage ######################################

####################################
###### get general parameters ######
####################################

parser = argparse.ArgumentParser(description='Visualize original csv data')

parser.add_argument('--function', type=str,
                    default='post_process',
                    help='check_ori_data: visualize original data from Captury Live; \
                          verify_before_output: verify dataset before output into .npy files; \
                          post_process: examine misclassified windows after testing',
                    choices=['check_ori_data','verify_before_output','post_process']
                    )

##############################################################
###### function: check_ori_data or verify_before_output ######
##############################################################

parser.add_argument('--single_data_path', type=str,
                    default='dataset/Static_Apostolos/testset/Test_Staic_Apostolos',
                    help='only one single dataset for function check_ori_data and verify_before_output')
parser.add_argument('--start_frame', type=int, default=2000, help='from which frame to start visualize')
parser.add_argument('--end_frame', type=int, default=2400, help='to which frame to end visualize')
parser.add_argument('--wl', type=int, default=51, help='window length for dataset creation, make it as odd number')
parser.add_argument('--output_anim', type=bool, default=True, help='whether to output animation of visualization')
parser.add_argument('--desired_features_trial', type=str, 
                    default='config/desired_features_trial.yaml', help='load features name from .yaml')

####################################
###### function: post_process ######
####################################

parser.add_argument('--examine_data_path', type=str,nargs="+",
                    default=[
                             'dataset/Dynamic_User1/trainset/Train_Dynamic_User1_11_Ball',
                             ],
                    help='location of examined data for misclassified labels, only external testset')
parser.add_argument('--misclassified_frames', type=list,default=[0,100],help='check misclassified frames')
parser.add_argument('--desired_features', type=str, 
                    default='config/desired_features.yaml', help='load features name from .yaml')

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
        verification([input_paths],args.desired_features,[split_method_paths],win_len=args.wl)

    if args.function == 'post_process':

        input_paths = [os.path.join(data_path,'unknown.NoHead.csv') for data_path in args.examine_data_path]
        split_method_paths = [os.path.join(data_path,'split_method.yaml') for data_path in args.examine_data_path]
        dists,angles = get_feature_selection(args.desired_features)

        _,_,skeleton = output_dataset(ori_data_paths=input_paths,
                                      desired_dists=dists,
                                      desired_angles=angles,
                                      split_method_paths=split_method_paths,
                                      standard='no_scale')
        
        show_misclassified_frames(skeleton,args)
