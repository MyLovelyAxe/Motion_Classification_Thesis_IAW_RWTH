import numpy as np

class StaticData():

    def __init__(self,
                 train_len,
                 test_len,
                 trainset_path,
                 testset_path):
        ### load data
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
        ### define length for trainset and randomly select train samples
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

class DynammicData():
    pass
