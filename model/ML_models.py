import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class Classification():
    
    def __init__(self,
                 Train_Len,
                 Test_Len,
                 Trainset_Path,
                 Testset_Path=None):
        self.get_data(train_len=Train_Len,
                      test_len=Test_Len,
                      trainset_path=Trainset_Path,
                      testset_path=Testset_Path)
        self.T_pred = None
        self.P_pred = None

    def get_data(self,
                 train_len,
                 test_len,
                 trainset_path,
                 testset_path):
        ### load data
        x_data_path, y_data_path = trainset_path[0], trainset_path[1]
        with open(x_data_path, 'rb') as xf:
            x_data = np.load(xf)
        with open(y_data_path, 'rb') as yf:
            y_data = np.load(yf)
        ### define length for trainset and randomly select train samples
        self.Train_Len = train_len
        choices_train = np.random.randint(x_data.shape[0], size = self.Train_Len)
        self.x_train = x_data[choices_train]
        self.y_train = y_data[choices_train]
        print(f'x_train shape: {self.x_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        ### prepare test set
        # if no outside testset is designated, then extract testset from trainset
        if testset_path is None:
            self.Test_Len = test_len
            # delete train samples for test samples
            new_x_data = np.delete(x_data, choices_train, axis=0)
            new_y_data = np.delete(y_data, choices_train, axis=0)
            # randomly select test samples
            choices_test = np.random.randint(new_x_data.shape[0], size = self.Test_Len)
            self.x_test = new_x_data[choices_test]
            self.y_test = new_y_data[choices_test]
        # if outside testset is designated, then load outside testset
        else:
            x_test_path, y_test_path = testset_path[0], testset_path[1]
            with open(x_test_path, 'rb') as test_xf:
                self.x_test = np.load(test_xf)
            with open(y_test_path, 'rb') as test_yf:
                self.y_test = np.load(test_yf)
        print(f'x_test shape: {self.x_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')
        print()
    
    def show_result(self):
        print(f'predicted target: {self.T_pred}')
        if not self.P_pred is None:
            self.P_pred = (self.P_pred).astype(np.float32)
            # Generate x-axis values
            frame_rate=60 # Hz
            sample_numbers = np.arange(self.P_pred.shape[0])/frame_rate
            # Plotting
            for i in range(self.P_pred.shape[1]):
                plt.plot(sample_numbers, self.P_pred[:, i], label=f'Activity #{i+1}')
            plt.xlabel(f'Time [sec]')
            plt.ylabel(f'Prediction Probability [%]')
            plt.legend()
            plt.show()
            print(f'type of P_pred: {type(self.P_pred)}')
            print(f'probability of predicted target: {self.P_pred}')
        print(f'true target: {self.y_test}')
        print(f'Accuracy = {np.sum(self.T_pred == self.y_test) / len(self.T_pred)}')
        print(f'Result: {self.T_pred == self.y_test}')

class KNN(Classification):
    
    def __init__(self,
                 N_neighbor,
                 Train_Len,
                 Test_Len,
                 Trainset_Path,
                 Testset_Path=None):
        super().__init__(Train_Len,
                         Test_Len,
                         Trainset_Path,
                         Testset_Path)
        # super().__init__()
        self.neigh = KNeighborsClassifier(n_neighbors=N_neighbor)
    def train(self):
        self.neigh.fit(self.x_train, self.y_train)
    def test(self):
        self.P_pred = self.neigh.predict_proba(self.x_test)
        self.T_pred = self.neigh.predict(self.x_test)
        
class RandomForest(Classification):
    
    def __init__(self,
                 Max_Depth,
                 Random_State,
                 Train_Len,
                 Test_Len,
                 Trainset_Path,
                 Testset_Path=None):
        super().__init__(Train_Len,
                         Test_Len,
                         Trainset_Path,
                         Testset_Path)
        self.random_forest = RandomForestClassifier(max_depth=Max_Depth,random_state=Random_State)
    def train(self):
        self.random_forest.fit(self.x_train, self.y_train)
    def test(self):
        self.P_pred = self.random_forest.predict_proba(self.x_test)
        maximums = np.max(self.P_pred,axis=1)
        mask = np.where(maximums < 0.4, False, True)
        self.T_pred = self.random_forest.predict(self.x_test)
        self.T_pred = np.multiply(self.T_pred,mask)
        
class SVM(Classification):
    
    def __init__(self,
                 Train_Len,
                 Test_Len,
                 Trainset_Path,
                 Testset_Path=None):
        super().__init__(Train_Len,
                         Test_Len,
                         Trainset_Path,
                         Testset_Path)
        self.svm = svm.SVC()
    def train(self):
        self.svm.fit(self.x_train, self.y_train)
    def test(self):
        self.T_pred = self.svm.predict(self.x_test)
