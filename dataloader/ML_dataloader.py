import numpy as np

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
                 testset_path):
        self.wl = window_size
        self.train_len = train_len
        self.test_len = test_len
        self.trainset_path = trainset_path
        self.testset_path = testset_path
        self.load_data()
        self.create_windows()
        # self.calc_statistic_features()
        # self.create_trainset()
        # self.create_testset()

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
        self.num_frames, self.num_ori_features = self.x_data_ori.shape

    def create_windows(self):
        """
        x_data shape:
            original: [#frames,#features]
            with windows: [#win,window_size,#features]
        y_data shape:
            original: [#frames,]
            with windows: [#win,]
        """
        # self.x_data_win = np.zeros((self.num_frames,self.wl,self.num_ori_features))
        self.x_data_win_lst = []
        self.y_data_win_lst = []
        values, start_index, counts = np.unique(self.y_data_ori, return_counts=True, return_index=True)
        counts = counts[start_index.argsort()]
        start_index.sort()
        # end_index = start_index + list(count-1 for count in counts)
        for start,count in zip(start_index,counts):
            for step in range(count-self.wl):
                window = self.x_data_ori[(start+step):(start+step+self.wl),:]
                label = self.y_data_ori[int(start+step+self.wl/2)]
                self.x_data_win_lst.append(np.expand_dims(window,axis=0))
                self.y_data_win_lst.append(np.expand_dims(label,axis=0))
                # self.y_data_win_lst.append(label)
        self.x_data_win = np.concatenate(self.x_data_win_lst,axis=0)
        self.y_data_win = np.concatenate(self.y_data_win_lst,axis=0)
        print(f'x_data with window has shape: {self.x_data_win.shape}')
        print(f'y_data with window has shape: {self.y_data_win.shape}')
        del self.x_data_win_lst,self.y_data_win_lst

    def calc_statistic_features(self):
        """
        x_data shape:
            with windows: [#win,window_size,#features]
            with statistic features: [#win,#stat_features]
        y_data shape:
            with windows: [#win,]
            with statistic features: [#win,]
        """
        self.x_data = self.x_data_win
        self.y_data = self.y_data_win
        del self.x_data_win,self.y_data_win

    def create_trainset(self):

        # define length for trainset and randomly select train samples
        self.choices_train = np.random.randint(self.x_data.shape[0], size = self.train_len)
        self.x_train = self.x_data[self.choices_train]
        self.y_train = self.y_data[self.choices_train]


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
