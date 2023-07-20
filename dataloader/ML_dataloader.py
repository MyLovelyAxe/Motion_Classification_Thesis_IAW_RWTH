import numpy as np
import matplotlib.pyplot as plt
from util.features import dynamic_features

class StaticData():

    """
    Static activity means the movemnt whose features defining its class
    can be classified with only one singel frame,
    i.e. the movement without concept of time accumulation
    """

    def __init__(self,
                 train_len,
                 test_len,
                 trainset_path,
                 testset_path):
        self.train_len = train_len
        self.test_len = test_len
        self.trainset_path = trainset_path
        self.testset_path = testset_path
        self.create_trainset()
        self.create_testset()

    def create_trainset(self):

        with open(self.trainset_path[0], 'rb') as xf:
            self.x_data = np.load(xf)
        with open(self.trainset_path[1], 'rb') as yf:
            self.y_data = np.load(yf)
        # define length for trainset and randomly select train samples
        self.choices_train = np.random.randint(self.x_data.shape[0], size = self.train_len)
        self.x_train = self.x_data[self.choices_train]
        self.y_train = self.y_data[self.choices_train]
        print(f'x_train shape: {self.x_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print()

    def create_testset(self):

        # if no outside testset is designated, then extract testset from trainset
        if self.testset_path is None:
            # delete train samples for test samples
            self.remain_x_data = np.delete(self.x_data, self.choices_train, axis=0)
            self.remain_y_data = np.delete(self.y_data, self.choices_train, axis=0)
            # randomly select test samples
            self.choices_test = np.random.randint(self.remain_x_data.shape[0], size = self.test_len)
            self.x_test = self.remain_x_data[self.choices_test]
            self.y_test = self.remain_y_data[self.choices_test]
        # if outside testset is designated, then load outside testset
        else:
            with open(self.testset_path[0], 'rb') as test_xf:
                self.x_test = np.load(test_xf)
            with open(self.testset_path[1], 'rb') as test_yf:
                self.y_test = np.load(test_yf)
        print(f'x_test shape: {self.x_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')
        print()

class DynamicData():

    """
    Dynamic activities can not be easily classified with single frame,
    therefore, to create window which covers several sequential frames is necessary.
    In each window, there are statistic data of each features
    to describe the movement inside.
    """

    def __init__(self,
                 window_size,
                 train_len,
                 test_len,
                 trainset_path,
                 testset_path,
                 trial=False):
        self.wl = window_size
        self.train_len = train_len
        self.test_len = test_len
        self.trainset_path = trainset_path
        self.testset_path = testset_path
        self.trial = trial
        self.load_data()
        self.create_windows()
        self.calc_statistic_features()
        if not self.trial:
            self.create_trainset()
            self.create_testset()                                 

    def load_data(self):
        """
        x_data shape:
            original: [#frames,#features]
        y_data shape:
            original: [#frames,]
        """
        with open(self.trainset_path[0], 'rb') as xf:
            self.x_data_ori = np.load(xf)
        with open(self.trainset_path[1], 'rb') as yf:
            self.y_data_ori = np.load(yf)
        print(f'loaded original x_data shape: {self.x_data_ori.shape}')
        print(f'loaded original y_data shape: {self.y_data_ori.shape}')
        print()
        self.num_features = self.x_data_ori.shape[1]

    def create_windows(self):
        """
        x_data shape:
            original: [#frames,#features]
            with windows: [#win,window_size,#features]
        y_data shape:
            original: [#frames,]
            with windows: [#win,]
        """
        self.x_data_win_lst = []
        self.y_data_win_lst = []
        _,start_index,counts = np.unique(self.y_data_ori, return_counts=True, return_index=True)
        counts = counts[start_index.argsort()]
        start_index.sort()
        for start,count in zip(start_index,counts):
            for step in range(count-self.wl):
                window = self.x_data_ori[(start+step):(start+step+self.wl),:]
                label = self.y_data_ori[int(start+step+self.wl/2)]
                self.x_data_win_lst.append(np.expand_dims(window,axis=0))
                self.y_data_win_lst.append(np.expand_dims(label,axis=0))
        self.x_data_win = np.concatenate(self.x_data_win_lst,axis=0)
        self.y_data_win = np.concatenate(self.y_data_win_lst,axis=0)
        print(f'x_data with window has shape: {self.x_data_win.shape}')
        print(f'y_data with window has shape: {self.y_data_win.shape}')
        print()
        del self.x_data_win_lst,self.y_data_win_lst,self.x_data_ori,self.y_data_ori,start_index,counts

    def calc_statistic_features(self):
        """
        x_data shape:
            with windows: [#win,window_size,#features]
            with statistic features: [#win,#features*#metrics]
        y_data shape:
            with windows: [#win,]
            with statistic features: [#win,]
        """
        self.x_data = dynamic_features(self.x_data_win)
        self.y_data = self.y_data_win
        print(f'x_data with window features has shape: {self.x_data.shape}')
        print(f'y_data with window features has shape: {self.y_data.shape}')
        print()
        if not self.trial:
            del self.x_data_win,self.y_data_win

    def plot_window_features(self,
                             which_activity:list,
                             win_range:list,
                             feature_idx:list,
                             metric_idx:list):
        """
            which_activity: [act_label_1,act_label_2,...]
                a list containing the label of which activities to plot
            win_range: [start_idx,end_idx]
                a list containing the desired range to plot, i.e. from which window to which window
                ps: the range is the index inside each activity, i.e. relatice range, not absolute range
                e.g. which_activity=3, win_range=[0,200] means plot from 0th to 200th window of act_3
                     which_activity=6, win_range=[0,200] means plot from 0th to 200th window of act_6
            feature_idx: [f_idx_1,f_idx2,...]
                a list containing indices of desired features to plot
            metric_idx: [m_idx_1,m_idx_2,...]
                a list containing indices of desired metrics (e.g. mean, std) to plot
        """
        values,starts,counts = np.unique(self.y_data, return_counts=True, return_index=True)
        print(f'values: {values}')
        print(f'starts: {starts}')
        print(f'counts: {counts}')
        self.num_win = win_range[0] - win_range[1]
        self.num_metrics = int(self.x_data.shape[1] / self.num_features)
        print(f'{self.num_win} windows will be checked, in each window there are {self.num_features} features, each feature has {self.num_metrics} metrics')
        fig = plt.figure(figsize=(8,6))
        ax = plt.subplot(1,1,1)
        for act in which_activity:
            # 1. where is beginning index of this act in self.y_data, based on 'values':
            label_idx = np.where(values==act)[0][0]
            print(f'act: {act}, label_idx: {label_idx}')
            # 2. check if the given win_range exceeds the upper limit of this act's range
            print(f'upper limit of this act: {counts[label_idx]}')
            assert win_range[1]<=counts[label_idx], f'the desired window_range exceeds upper limit of act {act}'
            # 3. select te segment of windows to plot
            print(f'from {starts[label_idx]+win_range[0]}th window to {starts[label_idx]+win_range[1]}th window')
            desired_windows = self.x_data[starts[label_idx]+win_range[0]:starts[label_idx]+win_range[1],:]
            print(f'Batch of these windows have shape: {desired_windows.shape}')
            # 4. based on feature_idx and metric_idx, calculate the real index on dimension of x_data.shape[1]
            desired_windows = desired_windows.reshape(self.num_win,self.num_metrics,self.num_features)
            print(f'Reshaped batch have shape: {desired_windows.shape}')
            # 5. plot static plot
            for i in metric_idx:
                for j in feature_idx:
                    ax.plot(np.arange(win_range[0],win_range[1]),desired_windows[:,i,j],label=f'Act{act}-F{j}-M{i}')
        ax.legend()
        ax.set_xlabel('num_window')
        plt.show()

    def create_trainset(self):

        # define length for trainset and randomly select train samples
        self.choices_train = np.random.randint(self.x_data.shape[0], size = self.train_len)
        self.x_train = self.x_data[self.choices_train]
        self.y_train = self.y_data[self.choices_train]
        print(f'x_train shape: {self.x_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print()

    def create_testset(self):

        # if no outside testset is designated, then extract testset from trainset
        if self.testset_path is None:
            # delete train samples for test samples
            self.remain_x_data = np.delete(self.x_data, self.choices_train, axis=0)
            self.remain_y_data = np.delete(self.y_data, self.choices_train, axis=0)
            # randomly select test samples
            self.choices_test = np.random.randint(self.remain_x_data.shape[0], size = self.test_len)
            self.x_test = self.remain_x_data[self.choices_test]
            self.y_test = self.remain_y_data[self.choices_test]
        # if outside testset is designated, then load outside testset
        else:
            with open(self.testset_path[0], 'rb') as test_xf:
                self.x_test = np.load(test_xf)
            with open(self.testset_path[1], 'rb') as test_yf:
                self.y_test = np.load(test_yf)
        print(f'x_test shape: {self.x_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')
        print()

