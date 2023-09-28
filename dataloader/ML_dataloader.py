import numpy as np
import matplotlib.pyplot as plt
from util.features import meta_features, get_feature_index, get_metric_index, get_act_index, get_feature_index_dict
from util.utils import get_feature_selection, output_dataset


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
        self.localization()

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

        self.x_data_ori,self.y_data_ori,self.skeleton,self.frame_split_method = output_dataset(ori_data_paths=self.data_paths,
                                                                                               desired_dists=dists,
                                                                                               desired_angles=angles,
                                                                                               split_method_paths=self.split_method_paths,
                                                                                               standard=self.standard)
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
        for _,actConfig in self.frame_split_method.items():
            start,end,label = list(i for _,i in actConfig.items())
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

    def localization(self):
        """
        Get the index and counts of each label in y_data for spliting for training
        """
        self.values, self.starts, self.counts = np.unique(self.y_data, return_counts=True, return_index=True)

    def plot_window_features(self,
                             which_activity: list,
                             win_range: list,
                             which_feature: list,
                             which_metric: list):
        """
        which_activity: [aName_1,aName_2,...]
            a list containing the name of which activities to plot
        win_range: [start_idx,end_idx]
            a list containing the desired range to plot, i.e. from which window to which window
            ps: the range is the index inside each activity, i.e. relatice range, not absolute range
            e.g. which_activity=3, win_range=[0,200] means plot from 0th to 200th window of act_3
                    which_activity=6, win_range=[0,200] means plot from 0th to 200th window of act_6
        which_feature: [fName_1,fName_2,...]
            a list containing names of desired features to plot
        which_metric: [mName_1,mName_2,...]
            a list containing names of desired metrics (e.g. mean, std) to plot
        """
        # prepare
        act_idx_lst = get_act_index(self.split_method_paths, which_activity)
        feature_idx_lst = get_feature_index(which_feature)
        metric_idx_lst = get_metric_index(which_metric)
        # show localization
        print(f'values: {self.values}')
        print(f'starts: {self.starts}')
        print(f'counts: {self.counts}')
        # get shapes
        self.num_win = win_range[0] - win_range[1]
        self.num_metrics = int(self.x_data.shape[1] / self.num_features)
        print(f'{self.num_win} windows will be checked, in each window there are {self.num_features} features, each feature has {self.num_metrics} metrics')
        # plot
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)
        for aIdx, aName in zip(act_idx_lst, which_activity):
            # 1. where is beginning index of this act in self.y_data, based on 'values':
            label_idx = np.where(self.values == aIdx)[0][0]
            print(f'act: {aName}, label_idx: {label_idx}')
            # 2. check if the given win_range exceeds the upper limit of this act's range
            print(f'upper limit of this act: {self.counts[label_idx]}')
            assert win_range[1] <= self.counts[
                label_idx], f'the desired window_range exceeds upper limit of act {aName}'
            # 3. select te segment of windows to plot
            print(f'from {self.starts[label_idx]+win_range[0]}th window to {self.starts[label_idx]+win_range[1]}th window')
            desired_windows = self.x_data[self.starts[label_idx]+win_range[0]:self.starts[label_idx]+win_range[1], :]
            print(f'Batch of these windows have shape: {desired_windows.shape}')
            # 4. based on feature_idx and metric_idx, calculate the real index on dimension of x_data.shape[1]
            desired_windows = desired_windows.reshape(self.num_win, self.num_metrics, self.num_features)
            print(f'Reshaped batch have shape: {desired_windows.shape}')
            # 5. plot static plot
            for mIdx, mName in zip(metric_idx_lst, which_metric):
                for fIdx, fName in zip(feature_idx_lst, which_feature):
                    ax.plot(np.arange(win_range[0], win_range[1]), desired_windows[:,mIdx, fIdx], label=f'Act[{aName}]-F[{fName}]-M[{mName}]')
        # Put a legend to the right of the current axis
        ax.legend(bbox_to_anchor=(1.04, 0.5),loc="center left", borderaxespad=0)
        ax.set_xlabel('num_window')
        ax.set_title('Metrics of features for windows')
        plt.show()

    def plot_metric_features(self,
                             which_metric:list,
                             check_content:bool=False):
        """
        Make sure that:
            - which_metric only contains 1 metric
        """
        ### prepare basics
        mIdx = get_metric_index(which_metric)[0]
        self.num_win = self.x_data.shape[0]
        self.num_metrics = int(self.x_data.shape[1] / self.num_features)
        feature_index_dict = get_feature_index_dict(split=True)
        ### prepare data of the metric to plot
        shaped_x_data = self.x_data.copy()
        shaped_x_data = shaped_x_data.reshape(self.num_win, self.num_metrics, self.num_features)
        metric_data = shaped_x_data[:,mIdx,:]
        del shaped_x_data
        ### output content to check if anything wrong
        if check_content:
            print(f'Check activity:')
            print('=========================================')
            for aName,aIdx in self.aIdx_dict.items():
                print(f'act name: {aName}, act idx: {aIdx}')
            print()
            print(f'Check features:')
            print('=========================================')
            for catogary,fIdx_dict in feature_index_dict.items():
                print(f'catogary: {catogary}')
                print('----------------------------------------')
                for fName,fIdx in fIdx_dict.items():
                    print(f'feature name: {fName}, feature idx: {fIdx}')
        ### localization
        # show localization
        print(f'values: {self.values}')
        print(f'starts: {self.starts}')
        print(f'counts: {self.counts}')
        x_upperLim = np.max(self.counts)
        ### plotting
        ncol = len(feature_index_dict)
        nrow = len(self.aIdx_dict)
        fig, axes = plt.subplots(nrow, ncol, figsize=(40,30))
        for fNum,(catogary,fIdx_dict) in enumerate(feature_index_dict.items()):
            # labels = list(fName for fName,_ in fIdx_dict.items())
            ## determine limits for axis
            current_features_idx = list(f_idx for _,f_idx in fIdx_dict.items())
            y_upperLim = np.max(metric_data[:, current_features_idx])
            y_lowerLim = np.min(metric_data[:, current_features_idx])
            for aNum,(aName,aIdx) in enumerate(self.aIdx_dict.items()):
                ## where is beginning index of this act in self.y_data, based on 'values':
                label_idx = np.where(self.values == aIdx)[0][0]
                ## select te segment of this activity to plot
                start_idx = self.starts[label_idx]
                end_idx = start_idx+self.counts[label_idx]
                desired_windows = metric_data[start_idx:end_idx, :]
                ## plot
                for fName,fIdx in fIdx_dict.items():
                    axes[aNum,fNum].plot(np.arange(0,self.counts[label_idx]), desired_windows[:,fIdx])
                # axes[0,fNum].legend(loc='upper left',
                #                     bbox_to_anchor=(1.04, 0.5),
                #                     labels=labels)
                if aNum == 0:
                    axes[aNum,fNum].set_title(f'feature: {catogary}',fontsize=20)
                if fNum == 0:
                    axes[aNum,fNum].set_ylabel(f'{aName}',fontsize=20)
                axes[aNum,fNum].set_ylim([y_lowerLim,y_upperLim])
                axes[aNum,fNum].set_xlim([0,x_upperLim])
        fig.suptitle(f'MetaFeature: {which_metric[0]}',fontsize=40)
        fig.tight_layout()
        plt.show()

class DynamicData():
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
        self.win_frame_index = self.test_data.win_frame_index
        print(f'x_train shape: {self.x_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print(f'x_test shape: {self.x_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')
        print()
