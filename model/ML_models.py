import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from dataloader.ML_dataloader import DynamicData

class DynamicClassModel():
    
    def __init__(self,args):

        self.T_pred = None
        self.P_pred = None
        self.wl = args.window_size
        self.dynamic_data = DynamicData(args)
        # create model
        if args.model == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=args.n_neighbor)
        elif args.model == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=args.max_depth,random_state=args.random_state)
        elif args.model == 'SVM':
            self.model = svm.SVC(probability=True)

    def train(self):
        self.model.fit(self.dynamic_data.x_train, self.dynamic_data.y_train)

    def test(self):
        self.P_pred = self.model.predict_proba(self.dynamic_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)

    def misclass_index(self):
        """
        record the indices of frames which are misclassified
        """
        self.mis_index = np.where(self.T_pred!=self.dynamic_data.y_test)[0]
        pred_labels = self.T_pred[self.mis_index]
        mis_index_ori = self.dynamic_data.y_MisClsExm[self.mis_index]
        true_labels = self.dynamic_data.y_data_ori[mis_index_ori]
        Exm_start_frames = mis_index_ori - int(self.wl/2)
        Exm_end_frames = mis_index_ori + int(self.wl/2)
        self.Exm_indices = np.vstack((Exm_start_frames,Exm_end_frames)).T

        # because the result plot is converted back to orignal order
        # but to examine windows has to be in generated dataset, i.e. the one which was already split and concatenated
        # so plot_MisIdx is only for intuitive presentation on result plot, i.e. original data
        # and self.Exm_indices is to check windows on split&concat data
        tmp_idx = self.dynamic_data.y_ori_idx_win.copy()
        tmp_idx[self.mis_index] = -1 # -1 is just a label, to label where is misclassified
        new = tmp_idx[self.dynamic_data.y_ori_idx_win.argsort()]
        plot_MisIdx = np.where(new==-1)[0]

        print(f'The misclassified windows has shape: {self.Exm_indices.shape}')
        print(f'Examine the windows with these indices in data_visualization.py:')
        print(f'idx of misclassified window | check on dataset with:[start_frame, end_frame] | truth | prediction')
        for mis_idx,exm_idxs,tru,pre in zip(plot_MisIdx,self.Exm_indices,true_labels,pred_labels):
            print(f'{mis_idx} | {exm_idxs} | {tru} | {pre}')

    def show_result(self,args):

        if not self.P_pred is None:

            ### define output path
            output_path = 'result'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            self.P_pred = (self.P_pred).astype(np.float32)

            ### Generate x-axis values
            self.frame_rate=60 # Hz
            self.sample_numbers = np.arange(self.P_pred.shape[0])/self.frame_rate
            
            ### prepare act names
            aName_dict = {}
            for k,v in self.dynamic_data.train_data.aIdx_dict.items():
                aName_dict[v] = k

            ### re-index result to match original order
            plot_truth = self.dynamic_data.y_test[(self.dynamic_data.y_ori_idx_win).argsort()]
            plot_pred = self.P_pred[self.dynamic_data.y_ori_idx_win.argsort()]

            ### Plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,13))
            use_ext_test = '(with external testset)' if args.outside_test else '(with internal testset)'
            acc = np.sum(self.T_pred == self.dynamic_data.y_test) / len(self.T_pred)
            title = f'{args.exp_group}: {args.model}_wl{args.window_size} {use_ext_test} acc={round(acc, 3)}'
            fig.suptitle(title,fontsize=15)

            # prediction & truth plot
            for idx,act_idx in enumerate(self.dynamic_data.train_data.values):
                ax1.plot(self.sample_numbers, plot_pred[:, idx], label=f'{aName_dict[act_idx]}')
                truth = np.where(plot_truth==act_idx,1,0)
                ax2.plot(self.sample_numbers, truth, label=f'{aName_dict[act_idx]}')
            ax1.set_title(f'Prediction',fontsize=10)
            ax1.set_ylabel(f'Prediction Probability')
            ax2.set_title(f'Truth',fontsize=10)
            ax2.set_xlabel(f'Time [sec]')
            ax2.set_ylabel(f'Prediction Probability')
            plt.legend()

            ### save plot
            if args.save_res:
                output_image = f"{args.start_time}-{args.exp_group}-{args.model}-wl{args.window_size}-{use_ext_test}-Acc{round(acc, 3)}.png"
                plt.savefig(os.path.join(output_path,output_image))
            else:
                plt.show()

            ### print out results
            print(f'P_pred: type: {type(self.P_pred)}, shape: {self.P_pred.shape}')
            print(f'probability of predicted target: {self.P_pred}')
        print(f'predicted target: {self.T_pred}')
        print(f'true target: {self.dynamic_data.y_test}')
        print(f'Accuracy = {acc}')
        print(f'Result: {self.T_pred == self.dynamic_data.y_test}')
