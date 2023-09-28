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

ori_csv_path = 'unknown.csv'
ori_csv = pd.read_csv(ori_csv_path ,header=None, low_memory=False)
noHead_csv = ori_csv.iloc[5:-10]
noHead_csv.to_csv('unknown.NoHead.csv',header=None,index=False)
