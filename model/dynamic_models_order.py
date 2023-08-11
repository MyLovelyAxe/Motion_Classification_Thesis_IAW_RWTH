import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from dataloader.ML_dataloader_order import DynamicData

class DynamicClassModel():
    
    def __init__(self,
                 Window_Size,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None,
                 Split_Ratio=0.8,
                 Desired_Features=None):
        self.T_pred = None
        self.P_pred = None
        self.wl = Window_Size
        self.dynamic_data = DynamicData(window_size=Window_Size,
                                        train_split_method_paths=Train_Split_Method_Paths,
                                        trainset_path=Trainset_Path,
                                        test_split_method_paths=Test_Split_Method_Paths,
                                        testset_path=Testset_Path,
                                        split_ratio=Split_Ratio,
                                        desired_features=Desired_Features
                                        )

    def misclass_index(self):
        """
        record the indices of frames which are misclassified
        """
        self.mis_index = np.where(self.T_pred!=self.dynamic_data.y_test)[0]
        # true_labels = self.dynamic_data.test_data.y_data_ori[self.mis_index]
        pred_labels = self.T_pred[self.mis_index]
        mis_index_ori = self.dynamic_data.test_data.y_MisClsExm[self.mis_index]
        true_labels = self.dynamic_data.test_data.y_data_ori[mis_index_ori]
        Exm_start_frames = mis_index_ori - int(self.wl/2)
        Exm_end_frames = mis_index_ori + int(self.wl/2)
        self.Exm_indices = np.vstack((Exm_start_frames,Exm_end_frames)).T
        print(f'The misclassified windows has shape: {self.Exm_indices.shape}')
        print(f'Examine the windows with these indices in data_visualization.py:')
        print(f'idx of misclassified window | check on dataset with:[start_frame, end_frame] | truth | prediction')
        for mis_idx,exm_idxs,tru,pre in zip(self.mis_index,self.Exm_indices,true_labels,pred_labels):
            print(f'{mis_idx} | {exm_idxs} | {tru} | {pre}')

    def show_result(self,args):

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
            acc = np.sum(self.T_pred == self.dynamic_data.y_test) / len(self.T_pred)
            title = f'{args.exp_group}: {args.model}_wl{args.window_size} {use_ext_test} acc={round(acc, 3)}'
            fig.suptitle(title,fontsize=15)
            # reconstruct back to initial index
            if np.count_nonzero(self.dynamic_data.test_data.y_ori_idx_win) > 1:
                plot_truth = self.dynamic_data.y_test[(self.dynamic_data.test_data.y_ori_idx_win).argsort()]
                plot_pred = self.P_pred[self.dynamic_data.test_data.y_ori_idx_win.argsort()]
            else:
                plot_truth = self.dynamic_data.y_test
                plot_pred = self.P_pred
            # prediction & truth plot
            for idx,act_idx in enumerate(self.dynamic_data.train_data.values):
                ax1.plot(sample_numbers, plot_pred[:, idx], label=f'{aName_dict[act_idx]}')
                truth = np.where(plot_truth==act_idx,1,0)
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
        print(f'predicted target: {self.T_pred}')
        print(f'true target: {self.dynamic_data.y_test}')
        print(f'Accuracy = {acc}')
        print(f'Result: {self.T_pred == self.dynamic_data.y_test}')

class KNN(DynamicClassModel):
    
    def __init__(self,
                 N_neighbor,
                 Window_Size,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None,
                 Split_Ratio=0.8,
                 Desired_Features=None):
        super().__init__(Window_Size,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
                         Testset_Path,
                         Split_Ratio,
                         Desired_Features)
        self.neigh = KNeighborsClassifier(n_neighbors=N_neighbor)
    def train(self):
        self.neigh.fit(self.dynamic_data.x_train, self.dynamic_data.y_train)
    def test(self):
        self.P_pred = self.neigh.predict_proba(self.dynamic_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)
        # self.T_pred = self.neigh.predict(self.dynamic_data.x_test)
        
class RandomForest(DynamicClassModel):
    
    def __init__(self,
                 Max_Depth,
                 Random_State,
                 Window_Size,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None,
                 Split_Ratio=0.8,
                 Desired_Features=None):
        super().__init__(Window_Size,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
                         Testset_Path,
                         Split_Ratio,
                         Desired_Features)
        self.random_forest = RandomForestClassifier(max_depth=Max_Depth,random_state=Random_State)
    def train(self):
        self.random_forest.fit(self.dynamic_data.x_train, self.dynamic_data.y_train)
    def test(self):
        self.P_pred = self.random_forest.predict_proba(self.dynamic_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)
        # self.T_pred = self.random_forest.predict(self.dynamic_data.x_test)
        
class SVM(DynamicClassModel):
    
    def __init__(self,
                 Window_Size,
                 Train_Split_Method_Paths,
                 Trainset_Path,
                 Test_Split_Method_Paths=None,
                 Testset_Path=None,
                 Split_Ratio=0.8,
                 Desired_Features=None):
        super().__init__(Window_Size,
                         Train_Split_Method_Paths,
                         Trainset_Path,
                         Test_Split_Method_Paths,
                         Testset_Path,
                         Split_Ratio,
                         Desired_Features)
        self.svm = svm.SVC(probability=True)
    def train(self):
        self.svm.fit(self.dynamic_data.x_train, self.dynamic_data.y_train)
    def test(self):
        self.P_pred = self.svm.predict_proba(self.dynamic_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)
        # self.T_pred = self.svm.predict(self.dynamic_data.x_test)
