import os
import argparse
import pandas as pd

#############################################################################################
#
#   Original csv has redundant information which makes it difficult to process useful data
#   in order to only keep useful data and process them as numpy.array
#   the first 5 lines and last 10 lines should be discard and save as unknown.NoHead.csv
#   the first 5 lines are:
#       (1) recorded information
#       (2) number of frames
#       (3) definition of types of joints
#       (4) definition of data
#       (5) unit of data
#   the last 10 lines are:
#       (-10)       camera calibration information
#       (-9)        camera calibration coordinates
#       (-8->-1)    calication parameters
#   please paste the unknown.NoHead.csv under corresponding exp_group folders
#
#############################################################################################

parser = argparse.ArgumentParser(description='Cut heads and tails in original csv')
parser.add_argument('--exp_group', type=str, default='Static_Jialei_trial',
                    help='cut csv files in which exp_group')
args = parser.parse_args()

def cut_csv(exp_group,train_or_test):
    """
    cut off redundant information in orignal unknown.csv
    """
    print(f'/dataset/{exp_group}/{train_or_test}')
    set_path = os.path.join('dataset',exp_group,train_or_test)
    set_path_lst = os.listdir(set_path)
    set_path_lst.sort()
    for shot_path in set_path_lst:
        print(shot_path)
        unknown_csv_path = os.path.join(set_path,shot_path,'unknown.csv')
        Nohead_csv_path = os.path.join(set_path,shot_path,'unknown.NoHead.csv')
        ori_csv = pd.read_csv(unknown_csv_path ,header=None, low_memory=False)
        noHead_csv = ori_csv.iloc[5:-10]
        noHead_csv.to_csv(Nohead_csv_path,header=None,index=False)

def main():
    """
    cut trainset and testset separately
    """
    # trainset
    cut_csv(exp_group=args.exp_group,train_or_test='trainset')
    # testset
    cut_csv(exp_group=args.exp_group,train_or_test='testset')

if __name__ == '__main__':

    main()
