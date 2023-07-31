import os
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

    def show_result(self,modle_name,save=True):
        print(f'predicted target: {self.T_pred}')
        if not self.P_pred is None:
            # define output path
            output_path = 'result'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            self.P_pred = (self.P_pred).astype(np.float32)
            # Generate x-axis values
            frame_rate=60 # Hz
            sample_numbers = np.arange(self.P_pred.shape[0])/frame_rate
            # prepare act names
            aName_dict = {}
            for k,v in self.dynamic_data.aIdx_dict.items():
                aName_dict[v] = k
            # Plotting
            fig = plt.figure(figsize=(8,6))
            for idx,act_idx in enumerate(self.dynamic_data.values):
                plt.plot(sample_numbers, self.P_pred[:, idx], label=f'{aName_dict[act_idx]}')
            plt.title(f'Testing on model {modle_name}')
            plt.xlabel(f'Time [sec]')
            plt.ylabel(f'Prediction Probability')
            plt.legend(loc='upper left',bbox_to_anchor=(1.04, 1.0))
            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(output_path,f'Dynamic_{modle_name}.png'))
            else:
                plt.show()
            print(f'P_pred: type: {type(self.P_pred)}, shape: {self.P_pred.shape}')
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
        self.random_forest = RandomForestClassifier(max_depth=Max_Depth,random_state=Random_State)
    def train(self):
        self.random_forest.fit(self.dynamic_data.x_train, self.dynamic_data.y_train)
    def test(self):
        self.P_pred = self.random_forest.predict_proba(self.dynamic_data.x_test)
        self.T_pred = self.random_forest.predict(self.dynamic_data.x_test)
        
class SVM(DynamicClassModel):
    
    def __init__(self,
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
        self.svm = svm.SVC(probability=True)
    def train(self):
        self.svm.fit(self.dynamic_data.x_train, self.dynamic_data.y_train)
    def test(self):
        self.P_pred = self.svm.predict_proba(self.dynamic_data.x_test)
        self.T_pred = self.svm.predict(self.dynamic_data.x_test)
