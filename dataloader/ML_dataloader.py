import numpy as np
import matplotlib.pyplot as plt
from util.features import meta_features, get_metric_index, get_feature_index_dict, get_act_index_dict
from util.utils import get_feature_selection, output_dataset, get_split_methods


class Windowlize():
    """
    Take specific number of frames as 1 sample, i.e. windowlize,
    and slide along with all shot sequence but within each activity.
    In each window, there are statistic data of each features
    to describe the activity inside.
    """
    def __init__(self,
                 window_size,
                 data_paths,
                 split_method_paths,
                 standard,
                 desired_features=None
                 ):
        self.wl = window_size
        self.data_paths = data_paths
        self.split_method_paths = split_method_paths
        self.standard = standard
        self.desired_features = desired_features
        self.load_data()
        self.create_windows()
        self.calc_meta_features()

    def load_data(self):
        """
        return of output dataset.
            self.x_data_ori:
                Frame Feature Array Arr_ff, samples, containing features for each frame
                shape: [#frames,#features]
            self.y_data_ori:
                labels/classes for frames
                shape: [#frames,]
            self.skeleton:
                Original Data Arr_ori, containig original coordinates for skeleton joints
                shape: [#frames,3,26] # 3 means XYZ, 26 means 26 selected joints
            self.frame_split_method:
                split methods for all segments of recorded shot, for trainset or testset
                type: dict
        """
        dists,angles = get_feature_selection(self.desired_features)
        self.x_data_ori,self.y_data_ori,self.skeleton = output_dataset(ori_data_paths=self.data_paths,
                                                                                               desired_dists=dists,
                                                                                               desired_angles=angles,
                                                                                               split_method_paths=self.split_method_paths,
                                                                                               standard=self.standard)
        self.frame_split_method = get_split_methods(split_method_paths=self.split_method_paths)
        self.num_features = self.x_data_ori.shape[1]
        print(f'original x_data shape: {self.x_data_ori.shape}')
        print(f'original y_data shape: {self.y_data_ori.shape}')
        print()
        
    def create_windows(self):
        """
        Data definition:
            self.x_data_win:
                Windowlized Array Arr_w, samples, containing features for each frame but compressed in each window
            self.y_data_win:
                labels/classes for windows

        Data dimension transformation:
            x_data:
                self.x_data_ori -> self.x_data_win
                shape: [#frames,#features] -> [#win,window_size,#features]
            y_data:
                self.y_data_ori -> self.y_data_win
                shape: [#frames,] -> [#win,]
        """
        x_data_win_lst = []
        y_data_win_lst = []
        win_frame_index_lst = [] # record index of frames inside each window, for checking misclassified window later
        AccWin = 0 # accumulated number of windows
        self.win_split_methods = {}
        for actName,actConfig in self.frame_split_method.items():
            start,end,label = list(i for _,i in actConfig.items())
            # record window split methods
            self.win_split_methods[actName] = {'start':AccWin,'end':AccWin+end-start-self.wl}
            AccWin = AccWin+end-start-self.wl
            # please make sure number of frames of each activitiy in recorded shots greater than window_size
            for step in range(end-start-self.wl):
                window = self.x_data_ori[(start+step):(start+step+self.wl), :]
                x_data_win_lst.append(np.expand_dims(window, axis=0))
                y_data_win_lst.append(np.expand_dims(label, axis=0))
                win_frame_index_lst.append(np.expand_dims(np.array([start+step,start+step+self.wl]), axis=0))
        self.x_data_win = np.concatenate(x_data_win_lst, axis=0)
        self.y_data_win = np.concatenate(y_data_win_lst, axis=0)
        self.win_frame_index = np.concatenate(win_frame_index_lst, axis=0)
        print(f'x_data with window has shape: {self.x_data_win.shape}')
        print(f'y_data with window has shape: {self.y_data_win.shape}')
        print()

    def calc_meta_features(self):
        """
        Data definition:
            self.x_data:
                Window Feature Array Arr_wf, samples, containing features for each window
            self.y_data:
                labels/classes for windows
                
        Data dimension transformation:
            x_data:
                self.x_data_win -> self.x_data
                shape: [#win,window_size,#features] -> [#win,#features*#metrics]
            y_data:
                self.y_data_win -> self.y_data
                shape: [#win,] -> [#win,]
        """
        self.x_data = meta_features(self.x_data_win)
        self.y_data = self.y_data_win
        print(f'x_data with window features has shape: {self.x_data.shape}')
        print(f'y_data with window features has shape: {self.y_data.shape}')
        print()
        del self.x_data_win, self.y_data_win

    def plot_metric_features(self,
                             which_metric:list,
                             check_content:bool=False):
        """
        Make sure that:
            - which_metric only contains 1 metric
            - only plot trainset to have a intuition of situation of different activities
        """
        ### prepare basics
        mIdx = get_metric_index(which_metric)[0]
        self.num_win = self.x_data.shape[0]
        self.num_metrics = int(self.x_data.shape[1] / self.num_features)
        feature_index_dict = get_feature_index_dict(split=True)
        self.actLabel_actName_dict = get_act_index_dict(self.frame_split_method)
        ### prepare data of the metric to plot
        shaped_x_data = self.x_data.copy()
        shaped_x_data = shaped_x_data.reshape(self.num_win, self.num_metrics, self.num_features)
        metric_data = shaped_x_data[:,mIdx,:] # shape: [#win,#frame_features]
        del shaped_x_data
        ### output content to check if anything wrong
        if check_content:
            print(f'Check activity:')
            print('=========================================')
            for aName,aIdx in self.actLabel_actName_dict.items():
                print(f'act name: {aName}, act idx: {aIdx}')
            print()
            print(f'Check features:')
            print('=========================================')
            for catogary,fIdx_dict in feature_index_dict.items():
                print(f'catogary: {catogary}')
                print('----------------------------------------')
                for fName,fIdx in fIdx_dict.items():
                    print(f'feature name: {fName}, feature idx: {fIdx}')
        ### plotting
        x_upperLim = max([aStartEnd['end']-aStartEnd['start'] for _,aStartEnd in self.win_split_methods.items()])
        ncol = len(feature_index_dict) # 4, i.e. dist, rate_dist, angle, rate_angle
        nrow = len(self.actLabel_actName_dict) # number of classes
        fig, axes = plt.subplots(nrow, ncol, figsize=(40,30))
        for fNum,(catogary,fIdx_dict) in enumerate(feature_index_dict.items()):
            ## determine limits for y axis
            current_features_idx = list(f_idx for _,f_idx in fIdx_dict.items())
            y_upperLim = np.max(metric_data[:, current_features_idx])
            y_lowerLim = np.min(metric_data[:, current_features_idx])
            for aNum, (aName, aStartEnd) in enumerate(self.win_split_methods.items()):
                for fName,fIdx in fIdx_dict.items():
                    axes[aNum,fNum].plot(np.arange(0,aStartEnd['end']-aStartEnd['start']), metric_data[aStartEnd['start']:aStartEnd['end'],fIdx])
                if aNum == 0:
                    axes[aNum,fNum].set_title(f'feature: {catogary}',fontsize=20)
                if fNum == 0:
                    axes[aNum,fNum].set_ylabel(f'{aName}',fontsize=20)
                axes[aNum,fNum].set_ylim([y_lowerLim,y_upperLim])
                axes[aNum,fNum].set_xlim([0,x_upperLim])
        fig.suptitle(f'MetaFeature: {which_metric[0]}',fontsize=40)
        fig.tight_layout()
        plt.show()

class TrainTestExp():
    """
    integrate trainset and testset into one instance containing all training and testing data
    """
    def __init__(self,args):
        self.train_data = Windowlize(window_size=args.window_size,
                                     data_paths=args.trainset_paths,
                                     split_method_paths=args.train_split_method_paths,
                                     standard=args.standard,
                                     desired_features=args.desired_features)
        self.test_data = Windowlize(window_size=args.window_size,
                                    data_paths=args.testset_paths,
                                    split_method_paths=args.test_split_method_paths,
                                    standard=args.standard,
                                    desired_features=args.desired_features)
        self.x_train = self.train_data.x_data
        self.y_train = self.train_data.y_data
        self.x_test = self.test_data.x_data
        self.y_test = self.test_data.y_data
        self.frame_split_method = self.train_data.frame_split_method
        self.win_frame_index = self.test_data.win_frame_index
        print(f'x_train shape: {self.x_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print(f'x_test shape: {self.x_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')
        print()

class LoadTestExp():
    """
    integrate trainset and testset into one instance containing all training and testing data
    """
    def __init__(self,args):
        self.test_data = Windowlize(window_size=args.window_size,
                                    data_paths=args.testset_paths,
                                    split_method_paths=args.test_split_method_paths,
                                    standard=args.standard,
                                    desired_features=args.desired_features)
        self.x_test = self.test_data.x_data
        self.y_test = self.test_data.y_data
        self.frame_split_method = get_split_methods(split_method_paths=args.train_split_method_paths)
        self.win_frame_index = self.test_data.win_frame_index
        print(f'x_test shape: {self.x_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')
        print()
