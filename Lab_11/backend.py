'''
Sinh viên thực hiện: Nguyễn Huỳnh Hải Đăng
MSSV: 20521159
'''

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier



from typing import *

class Dataset:
    def __init__(self,path_data,typedata="path"):
        self.data=None
        self.encoder={}
        self.features = []
        self.str_col=[]
        self.target_encoder = None
        if typedata=="df":
            self.origin_data = path_data.copy()
        else:
            self.origin_data = pd.read_csv(path_data).sample(frac=1).reset_index(drop=True)
        if len(self.origin_data.columns)==1:
            self.origin_data = pd.read_table(path_data,sep=";")

    def process_data(self, data:object):
        self.data = data.copy()
        self.features.append(list(self.data.columns))
        self.target = self.data.columns[-1]
        self.features_value = list(filter(lambda x: x!=self.target, self.data.columns))
        self.data[self.target] =  list(map(int,self.data[self.target]==2))
        return self.target, self.features_value

    def get_feature_target(self,features_list:List[str],target:str,pca=False, pca_n=1):
        features_list+=[target]
        target_col,feature_col = self.process_data(self.origin_data[features_list])
        print(f"feature_col:{feature_col}")
        print(f"target_col:{target_col}")
        return self.data[feature_col].values, self.data[[target_col]].values
    def getLenOriData(self):
        return len(self.origin_data[self.origin_data.columns[0]].values)


class Model_AI:
    def __init__(self,dataset,setting):
        self.data= dataset
        self.setting = setting
        self.AbsModel = LogisticRegression
        self.listModel = {"lr":(LogisticRegression,dict()),
                          "dt":(DecisionTreeClassifier,dict()), 
                          "svm":(SVC,dict(probability=True)), 
                          "xgb":(XGBClassifier,dict(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'))}
        self.history = {}
        if self.setting["LogLoss"]:
            self.history["LogLoss"]={"lr":{},"dt":{}, "svm":{}, "xgb":{}}
        if self.setting["F1"]:
            self.history["F1"]={"lr":{},"dt":{}, "svm":{}, "xgb":{}}

    def fit(self, a=""):
        X,Y = self.data.get_feature_target(self.setting["feature_list"],self.setting["target"])
        print(f"y={Y}")
        if self.setting["kfold"]:
            kf = KFold(n_splits=self.setting["K"], shuffle=True)
            fold_id=0
            for train_index, test_index in kf.split(X):
                scaler = MinMaxScaler()
                xtrain, xtest = X[train_index].copy(), X[test_index].copy()
                xtrain = scaler.fit_transform(xtrain)
                xtest = scaler.transform(xtest)

                ytrain, ytest = Y[train_index].copy(), Y[test_index].copy()

                for model_name in self.listModel.keys():
                    model_info = self.listModel[model_name]
                    Model = model_info[0](**model_info[1])
                    Model.fit(xtrain,ytrain)

                    yhat = Model.predict(xtest)
                    yhat_p = Model.predict_proba(xtest)

                    print(f"yhat1={yhat}")
                    print(f"yhat_p1={yhat_p}")
                    if self.setting["LogLoss"]:
                        self.history["LogLoss"][model_name].update({fold_id:log_loss(ytest,yhat_p)})
                    if self.setting["F1"]:
                        self.history["F1"][model_name].update({fold_id:f1_score(ytest,yhat, average='weighted')})
                fold_id+=1
        else:
            xtrain,xtest,ytrain,ytest = train_test_split(X,Y, test_size = 1-self.setting["rate"])
            for model_name in self.listModel.keys():
                model_info = self.listModel[model_name]
                Model = model_info[0](**model_info[1])
                Model.fit(xtrain,ytrain)
                yhat = Model.predict(xtest)
                yhat_p = Model.predict_proba(xtest)

                #print(f"yhat2={yhat}")
                #print(f"yhat_p2={yhat_p}")
                if self.setting["LogLoss"]:
                    #print("yhat_p={yhat_p}")
                    self.history["LogLoss"][model_name].update({0:log_loss(ytest,yhat_p)})
                if self.setting["F1"]:
                    self.history["F1"][model_name].update({0:f1_score(ytest,yhat, average='weighted')})
        self.model = Model
        #print(self.history)
    @property
    def get_value_metrics(self):
        return {"f1":{model_key:np.mean(list(self.history["F1"][model_key].values())) for model_key in self.history["F1"].keys()}, 
                "LogLoss":{model_key:np.mean(list(self.history["LogLoss"][model_key].values())) for model_key in self.history["LogLoss"].keys()}}

    def plot_history(self):
        fig, ax = plt.subplots(nrows=2)
        title = []
        history = self.history
        for idx,i in enumerate(history.keys()):
            for j,model_R in enumerate(history[i].keys()):
                h = history[i][model_R]
                print(h)
                try:
                    v = np.array([i for i in list(h.values())])
                    label = np.array([i for i in range(len(v))])
                    print(f"i:{i}")
                    ax[idx].bar(label+0.2*j, v, label=model_R, width=0.2)
                    #ax[i].scatter([i for i in range(len(F1))], F1, label=model_R)
                    #title.append(i)
                    #ax[idx].legend(loc='lower right')
                    ax[idx].set_title(i)
                    ax[idx].set_xticks([i+0.3 for i in range(len(v))])
                    ax[idx].set_xticklabels([str(i+1) for i in range(len(v))])
                except Exception as e:
                    print(f"Error in plot: {e}")
                    raise e
        ax[-1].legend(loc='lower right',bbox_to_anchor=(1.2,0.8))
            #try:
            #    LogLoss = [i for i in data["log_loss"].values]
            #    rects2=ax.plot(labels, LogLoss, label='log_loss',color="blue")
            #    ax.scatter(labels, LogLoss, label='log_loss',color="blue")
            #    title.append("log_loss")
            #except:
            #    pass
            # Add some text for labels, title and custom x-axis tick labels, etc.
            #ax.set_ylabel('F1 Score')
            #ax.set_yscale("log")
            #ax.set_title("F1 score with different component")
            #ax.set_xticks(x, labels)
            #ax.legend()
            #ax.bar_label(rects1, padding=3)
            #ax.bar_label(rects2, padding=3)
        fig.tight_layout()
        return fig

    def extract_vector(self,features):
        feature_vector = []
        for i in features.keys():
            if i not in self.str_col:
                feature_vector.append(features[i])
        for i in self.str_col:
            feature_vector.extend(self.encoder[i].transform([[features[i]]])[0])
        return feature_vector

    def predict(self, features):
        '''
            features = { Position:..., Level:...}
        '''
        features = self.extract_vector(features)
        return self.best_model.predict([features]).reshape(1)[0]

def search_PCA(dataset:Dataset,setting:dict, limit_component:int ) -> tuple:
    history ={"f1":{},"log_loss":{}}
    setting = setting.copy()

    for i in range(1,limit_component+1):
        setting["pca"] = True
        setting["pca_n"] = i
        model = Model_AI(dataset,setting)
        model.fit(a="Cal from search PCA")
        f1, log_loss = list(model.get_value_metrics.values())
        history["f1"].update({i:f1})
        history["log_loss"].update({i:log_loss})


    data = pd.DataFrame(history)
    labels = list(data.index)
    fig, ax = plt.subplots(ncols=2,nrows=1)
    title = []
    for i in history.keys():
        for model_R in history[i].keys():
            h = history[i][model_R]
            try:
                F1 = [i for i in h.values()]
                rects1= ax.bar([i for i in range(len(F1))], F1, label=model_R)
                ax[i].scatter([i for i in range(len(F1))], F1, label=model_R)
                title.append(i)
            except:
                pass
        #try:
        #    LogLoss = [i for i in data["log_loss"].values]
        #    rects2=ax.plot(labels, LogLoss, label='log_loss',color="blue")
        #    ax.scatter(labels, LogLoss, label='log_loss',color="blue")
        #    title.append("log_loss")
        #except:
        #    pass
        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.set_ylabel('F1 Score')
        #ax.set_yscale("log")
        #ax.set_title("F1 score with different component")
        #ax.set_xticks(x, labels)
        #ax.legend()
        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    return fig