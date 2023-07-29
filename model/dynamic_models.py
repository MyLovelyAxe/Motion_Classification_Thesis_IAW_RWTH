import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from dataloader.ML_dataloader import DynamicData

class DynamicClassModel():
    
    def __init__(self,
                 Window_Size,
                 Split_Method_Paths,
                 Trainset_Path,
                 Testset_Path=None,
                 Split_Ratio=0.8):
        self.T_pred = None
        self.P_pred = None
        self.dynamic_data = DynamicData(window_size=Window_Size,
                                        split_method_paths=Split_Method_Paths,
                                        trainset_path=Trainset_Path,
                                        testset_path=Testset_Path,
                                        split_ratio=Split_Ratio
                                        )

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
        print(f'true target: {self.dynamic_data.y_test}')
        print(f'Accuracy = {np.sum(self.T_pred == self.dynamic_data.y_test) / len(self.T_pred)}')
        print(f'Result: {self.T_pred == self.dynamic_data.y_test}')

class KNN(DynamicClassModel):
    
    def __init__(self,
                 N_neighbor,
                 Window_Size,
                 Split_Method_Paths,
                 Trainset_Path,
                 Testset_Path=None,
                 Split_Ratio=0.8):
        super().__init__(Window_Size,
                         Split_Method_Paths,
                         Trainset_Path,
                         Testset_Path,
                         Split_Ratio)
        self.neigh = KNeighborsClassifier(n_neighbors=N_neighbor)
    def train(self):
        self.neigh.fit(self.dynamic_data.x_train, self.dynamic_data.y_train)
    def test(self):
        self.P_pred = self.neigh.predict_proba(self.dynamic_data.x_test)
        self.T_pred = self.neigh.predict(self.dynamic_data.x_test)
        
class RandomForest(DynamicClassModel):
    
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
        self.random_forest.fit(self.static_data.x_train, self.static_data.y_train)
    def test(self):
        self.P_pred = self.random_forest.predict_proba(self.static_data.x_test)
        maximums = np.max(self.P_pred,axis=1)
        mask = np.where(maximums < 0.4, False, True)
        self.T_pred = self.random_forest.predict(self.static_data.x_test)
        self.T_pred = np.multiply(self.T_pred,mask)
        
class SVM(DynamicClassModel):
    
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
        self.svm.fit(self.static_data.x_train, self.static_data.y_train)
    def test(self):
        self.T_pred = self.svm.predict(self.static_data.x_test)
