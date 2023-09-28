import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from dataloader.ML_dataloader import DynamicData
from util.features import get_act_index_dict
from util.utils import save_result

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

    def test(self, loaded_model=None):
        if loaded_model is None:
            self.P_pred = self.model.predict_proba(self.dynamic_data.x_test)
        else:
            self.P_pred = loaded_model.predict_proba(self.dynamic_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)

    def misclass_index(self):
        """
        record the indices of windows which are misclassified
        """
        self.misCls_win_index = np.where(self.T_pred!=self.dynamic_data.y_test)[0]
        pred_labels = self.T_pred[self.misCls_win_index]
        true_labels = self.dynamic_data.y_test[self.misCls_win_index]
        misCls_win_frame_index = self.dynamic_data.win_frame_index[self.misCls_win_index]
        print(f'idx of misclassified window | check on dataset with:[start_frame, end_frame] | truth | prediction')
        for mis_idx,exm_idxs,tru,pre in zip(self.misCls_win_index,misCls_win_frame_index,true_labels,pred_labels):
            print(f'{mis_idx} | {exm_idxs} | {tru} | {pre}')

    def result(self,args):

        ### save results
        actName_actLabel_dict = get_act_index_dict(self.dynamic_data.train_data.frame_split_method,NL=False)
        acc = np.sum(self.T_pred == self.dynamic_data.y_test) / len(self.T_pred)
        save_result(args=args,
                    model=self.model,
                    acc=acc,
                    plot_pred=self.P_pred,
                    values=self.dynamic_data.train_data.values,
                    plot_truth=self.dynamic_data.y_test,
                    actName_actLabel_dict=actName_actLabel_dict)

        # # e.g. args.load_model: 'save/10_Sep_16_33-Dynamic_Apostolos-RandomForest-wl100-MaxDepth6-RandomState0'
        # train_exp = (args.load_model).split('/')[1].split('-')[1]
        # test_exp = args.exp_group
        # output_image = f"{args.start_time}-Cross-Train_{train_exp}-Test_{test_exp}-{args.model}-wl{args.window_size}-Acc{round(acc, 3)}.png"

        ### print out results
        print(f'P_pred: type: {type(self.P_pred)}, shape: {self.P_pred.shape}')
        print(f'probability of predicted target: {self.P_pred}')
        print(f'predicted target: {self.T_pred}')
        print(f'true target: {self.dynamic_data.y_test}')
        print(f'Accuracy = {acc}')
        print(f'Result: {self.T_pred == self.dynamic_data.y_test}')
