import os
import argparse
import yaml

###########################################################################################################
#
#   When a new exp_group is designed:
#
#       1. write corresponding config/class_groupName.yaml to name activities and labels
#       2. run this script to generate split_methods.yaml for each activity of trainset of this exp_group
#
#   The config/class_Morning.yaml is an example to play with
#
###########################################################################################################

parser = argparse.ArgumentParser(description='generate split_methods.yaml for trainset of each new exp_group')

parser.add_argument('--exp_class',type=str,default='config/class_Morning.yaml',help='names and labels of activities')
parser.add_argument('--exp_group',type=str,default='Morning',help='name of exp group in dataset')
parser.add_argument('--record_time',type=int,default=20,help='number of seconds of recorded shot')


args = parser.parse_args()

if __name__ == "__main__":

    ### get names and labels of activities
    with open(args.exp_class, "r") as file:
        classes = yaml.safe_load(file)

    ### create folders
    exp_path = os.path.join('dataset',args.exp_group,'trainset')
    os.makedirs(exp_path, exist_ok=True)
    trainset_path = ''
    for name,label in classes.items():
        str_label = str(label)
        if len(str(label)) == 1:
            str_label = f'0' + str(label)
        trainset_folder = f'Train_{args.exp_group}_{str_label}_{name}'
        trainset_path = os.path.join(exp_path,trainset_folder)
        os.makedirs(trainset_path, exist_ok=True)
        split_methods_yaml = os.path.join(trainset_path,f'split_method.yaml')
        ### the split method to save
        split_methods_dict = {f'{name}1':{
                                            'start': 0,
                                            'end': args.record_time * 60,
                                            'label': label
                                            }
                                }
        with open(split_methods_yaml, 'w') as yaml_file:
            # sort_keys=False ensures the original order of dict when saved into .yaml
            yaml.dump(split_methods_dict, yaml_file, default_flow_style=False, sort_keys=False)
