import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from dataloader.ML_dataloader import TrainTestExp,LoadTestExp
from util.features import get_act_index_dict
from util.utils import save_result

class Exp():
    
    def __init__(self,args):

        self.T_pred = None
        self.P_pred = None
        self.wl = args.window_size
        # train and test a model, or test a loaded model
        if args.load_model is None:
            self.exp_data = TrainTestExp(args)
        else:
            self.exp_data = LoadTestExp(args)
        # create model
        if args.model == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=args.n_neighbor)
        elif args.model == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=args.max_depth,random_state=args.random_state)
        elif args.model == 'SVM':
            self.model = svm.SVC(probability=True)

    def train(self):
        self.model.fit(self.exp_data.x_train, self.exp_data.y_train)

    def test(self, loaded_model=None):
        if loaded_model is None:
            self.P_pred = self.model.predict_proba(self.exp_data.x_test)
        else:
            self.P_pred = loaded_model.predict_proba(self.exp_data.x_test)
        self.T_pred = np.argmax(self.P_pred,axis=1)

    def misclass_index(self):
        """
        record the indices of windows which are misclassified
        """
        self.misCls_win_index = np.where(self.T_pred!=self.exp_data.y_test)[0]
        pred_labels = self.T_pred[self.misCls_win_index]
        true_labels = self.exp_data.y_test[self.misCls_win_index]
        misCls_win_frame_index = self.exp_data.win_frame_index[self.misCls_win_index]
        print(f'idx of misclassified window | check on dataset with:[start_frame, end_frame] | truth | prediction')
        for mis_idx,exm_idxs,tru,pre in zip(self.misCls_win_index,misCls_win_frame_index,true_labels,pred_labels):
            print(f'{mis_idx} | {exm_idxs} | {tru} | {pre}')

    def result(self,args):
        """
        save trained model, config of args, performance plot
        """
        ### save results
        actLabel_actName_dict = get_act_index_dict(self.exp_data.frame_split_method,NL=False)
        acc = np.sum(self.T_pred == self.exp_data.y_test) / len(self.T_pred)
        save_result(args=args,
                    model=self.model,
                    acc=acc,
                    plot_pred=self.P_pred,
                    plot_truth=self.exp_data.y_test,
                    actLabel_actName_dict=actLabel_actName_dict)
        ### print out results
        print(f'P_pred: type: {type(self.P_pred)}, shape: {self.P_pred.shape}')
        print(f'probability of predicted target: {self.P_pred}')
        print(f'predicted target: {self.T_pred}')
        print(f'true target: {self.exp_data.y_test}')
        print(f'Accuracy = {acc}')
        print(f'Result: {self.T_pred == self.exp_data.y_test}')
