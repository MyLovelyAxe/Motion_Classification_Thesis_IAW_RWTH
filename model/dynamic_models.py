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
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None,
                 Split_Ratio=0.8):
        self.T_pred = None
        self.P_pred = None
        self.dynamic_data = DynamicData(window_size=Window_Size,
                                        train_split_method_paths=Train_Split_Method_Paths,
                                        trainset_path=Trainset_Path,
                                        test_split_method_paths=Test_Split_Method_Paths,
                                        testset_path=Testset_Path,
                                        split_ratio=Split_Ratio
                                        )

    def show_result(self,args):

        print(f'predicted target: {self.T_pred}')
        if not self.P_pred is None:
            ### define output path
            output_path = 'result'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            self.P_pred = (self.P_pred).astype(np.float32)
            ### Generate x-axis values
            frame_rate=60 # Hz
            sample_numbers = np.arange(self.P_pred.shape[0])/frame_rate
            ### prepare act names
            aName_dict = {}
            for k,v in self.dynamic_data.train_data.aIdx_dict.items():
                aName_dict[v] = k
            ### Plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,13))
            use_ext_test = '(with external testset)' if args.outside_test else '(with internal testset)'
            title = f'Testing: Dynamic_{args.model}_wl{args.window_size} {use_ext_test}'
            fig.suptitle(title,fontsize=15)
            # prediction & truth plot
            for idx,act_idx in enumerate(self.dynamic_data.train_data.values):
                ax1.plot(sample_numbers, self.P_pred[:, idx], label=f'{aName_dict[act_idx]}')
                truth = np.where(self.dynamic_data.y_test==act_idx,1,0)
                ax2.plot(sample_numbers, truth, label=f'{aName_dict[act_idx]}')
            ax1.set_title(f'Prediction',fontsize=10)
            ax1.set_ylabel(f'Prediction Probability')
            ax2.set_title(f'Truth',fontsize=10)
            ax2.set_xlabel(f'Time [sec]')
            ax2.set_ylabel(f'Prediction Probability')
            plt.legend()
            ### save plot
            if args.save_res:
                plt.savefig(os.path.join(output_path,f'{title}.png'))
            else:
                plt.show()
            ### print out results
            print(f'P_pred: type: {type(self.P_pred)}, shape: {self.P_pred.shape}')
            print(f'probability of predicted target: {self.P_pred}')
        print(f'true target: {self.dynamic_data.y_test}')
        print(f'Accuracy = {np.sum(self.T_pred == self.dynamic_data.y_test) / len(self.T_pred)}')
        print(f'Result: {self.T_pred == self.dynamic_data.y_test}')

class KNN(DynamicClassModel):
    
    def __init__(self,
                 N_neighbor,
                 Window_Size,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None,
                 Split_Ratio=0.8):
        super().__init__(Window_Size,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
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
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None,
                 Split_Ratio=0.8):
        super().__init__(Window_Size,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
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
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None,
                 Split_Ratio=0.8):
        super().__init__(Window_Size,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
                         Testset_Path,
                         Split_Ratio)
        self.svm = svm.SVC(probability=True)
    def train(self):
        self.svm.fit(self.dynamic_data.x_train, self.dynamic_data.y_train)
    def test(self):
        self.P_pred = self.svm.predict_proba(self.dynamic_data.x_test)
        self.T_pred = self.svm.predict(self.dynamic_data.x_test)
