import os
import argparse
import yaml
import pandas as pd

###########################################################################################################
#
#   When a new exp_group is designed:
#
#       1. write corresponding config/class_groupName.yaml to name activities and labels
#       2. change argument 'exp_class' with new 'config/class_groupName.yaml'
#       3. change argument 'exp_group' with name of exp_group according to naming style: groupName_user
#       4. change argument 'record_time' with shot duration of trainset, defaultly 20s
#       5. run this script to generate split_methods.yaml for each activity of trainset for this exp_group
#       6. note that split_methods.yaml of testset needs manually edition
#
###########################################################################################################

def get_arg():

    parser = argparse.ArgumentParser(description='generate split_methods.yaml for trainset of each new exp_group')
    parser.add_argument('--exp_class',type=str,default='config/class_Dynamic.yaml',help='names and labels of activities')
    parser.add_argument('--exp_group',type=str,default='Dynamic_User2',help='name of exp group in dataset, naming style: groupName_user')
    parser.add_argument('--record_time',type=int,default=20,help='number of seconds of recorded shot, unit: sec')
    args = parser.parse_args()
    return args

def cut_csv(from_path, to_path):
    """
    cut off redundant information in orignal unknown.csv
        first 5 lines:
            1 recording information
            2 number of frames
            3 name of joints
            4 coordinate name
            5 unit
        last 10 lines:
            camera calibration information
    Attention:
        please check if every unknown.csv has last lines of camera calibration
        it is possible that Captury Live output unknown.csv without last lines
        although these lines will be cut and not used in training
        program will fail if these lines are not there
        consider re-shoot the csv if the above situation happens
    """
    unknown_csv_path = os.path.join(from_path,'unknown.csv')
    Nohead_csv_path = os.path.join(to_path,'unknown.NoHead.csv')
    ori_csv = pd.read_csv(unknown_csv_path ,header=None, low_memory=False)
    noHead_csv = ori_csv.iloc[5:-10]
    noHead_csv.to_csv(Nohead_csv_path,header=None,index=False)

def process(args,train_or_test):
    """
    create split_method.yaml and cut orginal csv files
    """
    ### prepare root path
    root_from_path = os.path.join('ori_csv',args.exp_group)
    root_to_path = os.path.join('dataset',args.exp_group)
    ### get class-label pairs
    with open(args.exp_class, "r") as file:
        classes = yaml.safe_load(file)

    if train_or_test == 'testset':
        test_from_path = os.path.join(root_from_path,'testset',f'Test_{args.exp_group}')
        test_to_path = os.path.join(root_to_path,'testset',f'Test_{args.exp_group}')
        os.makedirs(test_to_path, exist_ok=True)
        # splid_method
        test_split_methods_yaml = os.path.join(test_to_path,f'split_method.yaml')
        with open(test_split_methods_yaml, 'w') as yaml_file:
            # create a pure empty .yaml for further editing
            pass
        # cut csv
        cut_csv(test_from_path, test_to_path)
    else:
        from_path = os.path.join(root_from_path,'trainset')
        to_path = os.path.join(root_to_path,'trainset')
        os.makedirs(to_path, exist_ok=True)
        for name,label in classes.items():
            str_label = str(label)
            if len(str(label)) == 1:
                str_label = f'0' + str(label)
            trainset_folder = f'Train_{args.exp_group}_{str_label}_{name}'
            trainset_from_path = os.path.join(from_path,trainset_folder)
            trainset_to_path = os.path.join(to_path,trainset_folder)
            os.makedirs(trainset_to_path, exist_ok=True)
            train_split_methods_yaml = os.path.join(trainset_to_path,f'split_method.yaml')
            # the split method to save
            train_split_methods_dict = {f'{name}1':
                                                {
                                                    'start': 0,
                                                    'end': args.record_time * 60,
                                                    'label': label
                                                }
                                        }
            with open(train_split_methods_yaml, 'w') as yaml_file:
                # sort_keys=False ensures the original order of dict when saved into .yaml
                yaml.dump(train_split_methods_dict, yaml_file, default_flow_style=False, sort_keys=False)
            # cut csv
            cut_csv(trainset_from_path, trainset_to_path)

def main():

    args = get_arg()
    process(args=args, train_or_test='trainset')
    process(args=args,  train_or_test='testset')

if __name__ == "__main__":

    main()
