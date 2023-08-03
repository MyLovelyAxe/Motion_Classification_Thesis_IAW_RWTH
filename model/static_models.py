import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from dataloader.ML_dataloader import StaticData

class StaticClassModel():
    
    def __init__(self,
                 Train_Len,
                 Test_Len,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None):
        self.T_pred = None
        self.P_pred = None
        self.external_testset = False
        if Test_Split_Method_Paths and Testset_Path:
            self.external_testset = True
        self.static_data = StaticData(train_len=Train_Len,
                                      test_len=Test_Len,
                                      train_split_method_paths=Train_Split_Method_Paths,
                                      trainset_path=Trainset_Path,
                                      test_split_method_paths=Test_Split_Method_Paths,
                                      testset_path=Testset_Path)

    def show_result(self,args):

        print(f'predicted target: {self.T_pred}')
        acc = np.sum(self.T_pred == self.static_data.y_test) / len(self.T_pred)
        if not self.P_pred is None:
            print(f'self.P_pred shape: {self.P_pred.shape}')
            ### define output path
            output_path = 'result'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            self.P_pred = (self.P_pred).astype(np.float32)
            # Generate x-axis values
            frame_rate=60 # Hz
            sample_numbers = np.arange(self.P_pred.shape[0])/frame_rate

            values = np.unique(self.static_data.y_test)

            # Plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13,13))
            use_ext_test = '(with external testset)' if args.outside_test else '(with internal testset)'
            title = f'Testing: Static_{args.model} {use_ext_test} acc={acc}'
            fig.suptitle(title,fontsize=15)

            aName_dict = {}
            for k,v in self.static_data.train_aIdx_dict.items():
                aName_dict[v] = k
            print(f'aName_dict: {aName_dict}')
            # for idx in range(self.P_pred.shape[1]):
            for idx,act_idx in enumerate(values):
                ax1.plot(sample_numbers, self.P_pred[:, idx])
                truth = np.where(self.static_data.y_test==act_idx,1,0)
                ax2.plot(sample_numbers, truth, label=f'{aName_dict[act_idx]}')

            # # for i in range(self.P_pred.shape[1]):
            # if not args.outside_test:
            #     aName_dict = {}
            #     for k,v in self.static_data.train_aIdx_dict.items():
            #         aName_dict[v] = k
            #     print(f'aName_dict: {aName_dict}')
            #     # for idx in range(self.P_pred.shape[1]):
            #     for idx,act_idx in enumerate(values):
            #         ax1.plot(sample_numbers, self.P_pred[:, idx])
            #         truth = np.where(self.static_data.y_test==act_idx,1,0)
            #         ax2.plot(sample_numbers, truth, label=f'{aName_dict[act_idx]}')
            # else:
            #     # values= np.unique(self.static_data.y_test)
            #     values = np.roll(values,-1)
            #     aName_dict = {}
            #     for k,v in self.static_data.test_aIdx_dict.items():
            #         aName_dict[v] = k
            #     print(f'aName_dict: {aName_dict}')
            #     for i in range(self.P_pred.shape[1]):
            #         ax1.plot(sample_numbers, self.P_pred[:, i])
            #     for idx,act_idx in enumerate(values):
            #         truth = np.where(self.static_data.y_test==act_idx,1,0)
            #         ax2.plot(sample_numbers, truth, label=f'{aName_dict[act_idx]}')

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
            print(f'type of P_pred: {type(self.P_pred)}')
            print(f'probability of predicted target: {self.P_pred}')
        print(f'true target: {self.static_data.y_test}')
        print(f'Accuracy = {acc}')
        print(f'Result: {self.T_pred == self.static_data.y_test}')

class KNN(StaticClassModel):
    
    def __init__(self,
                 N_neighbor,
                 Train_Len,
                 Test_Len,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None):
        super().__init__(Train_Len,
                         Test_Len,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
                         Testset_Path)
        self.neigh = KNeighborsClassifier(n_neighbors=N_neighbor)
    def train(self):
        self.neigh.fit(self.static_data.x_train, self.static_data.y_train)
    def test(self):
        self.P_pred = self.neigh.predict_proba(self.static_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)
        # self.T_pred = self.neigh.predict(self.static_data.x_test)
        
class RandomForest(StaticClassModel):
    
    def __init__(self,
                 Max_Depth,
                 Random_State,
                 Train_Len,
                 Test_Len,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None):
        super().__init__(Train_Len,
                         Test_Len,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
                         Testset_Path)
        self.random_forest = RandomForestClassifier(max_depth=Max_Depth,random_state=Random_State)
    def train(self):
        self.random_forest.fit(self.static_data.x_train, self.static_data.y_train)
    def test(self):
        self.P_pred = self.random_forest.predict_proba(self.static_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)
        # maximums = np.max(self.P_pred,axis=1)
        # mask = np.where(maximums < 0.4, False, True)
        # self.T_pred = self.random_forest.predict(self.static_data.x_test)
        # self.T_pred = np.multiply(self.T_pred,mask)
        
class SVM(StaticClassModel):
    
    def __init__(self,
                 Train_Len,
                 Test_Len,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None):
        super().__init__(Train_Len,
                         Test_Len,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
                         Testset_Path)
        self.svm = svm.SVC(probability=True)
    def train(self):
        self.svm.fit(self.static_data.x_train, self.static_data.y_train)
    def test(self):
        self.P_pred = self.svm.predict_proba(self.static_data.x_test)
        # self.T_pred = self.svm.predict(self.static_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)
