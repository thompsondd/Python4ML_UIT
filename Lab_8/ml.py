import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class OnehotEncoder:
    def __init__(self):
        self.lenght=0
        self.data={}
    def fit(self,data):
        for id,key in enumerate(data):
            self.data[key]=id
        self.lenght = len(data)
    def transform(self,data):
        ba = []
        for i in data:
            a = np.zeros(self.lenght,dtype=np.int32)
            for j in i:
                a[self.data[j]]=1
            ba.append(a.tolist())
        return ba

class Dataset:
    def __init__(self,path_data):
        self.data=None
        self.encoder={}
        self.features = []
        self.str_col=[]
        self.origin_data = pd.read_csv(path_data)
        if len(self.origin_data.columns)==1:
            self.origin_data = pd.read_table(path_data,sep=";")

    def check_str_type(self, word):
        return str(type(word)).split()[1].split("'")[1] == "str"
    def process_data(self, data):
        self.data = data.copy()
        self.features.append(list(self.data.columns))
        self.target = self.data.columns[-1]
        for i in self.data.columns:
            if self.check_str_type(self.data.iloc[0][i]):
                LaEn = OnehotEncoder()
                temp = self.data[i].unique()
                LaEn.fit(temp)
                self.str_col.append(i)
                temp_data = np.array(LaEn.transform(self.data[i].values.reshape((-1,1)))).T
                for id, key in enumerate(temp):
                    self.data[key]=temp_data[id]
                self.encoder.update({i:LaEn})
                self.data.drop(i,axis=1,inplace=True)
        self.features_value = list(filter(lambda x: x!=self.target, self.data.columns))
        return self.target, self.features_value
    def get_feature_target(self,features_list,target,pca=False, pca_n=1):
        features_list+=[target]
        target_col,feature_col = self.process_data(self.origin_data[features_list])
        if pca:
            return self.get_pca_feature(pca_n,feature_col), self.data[[target_col]].values
        return self.data[feature_col].values, self.data[[target_col]].values
    def get_pca_feature(self, n_components_user_choose,feature_col):
        n_components = n_components_user_choose - len(self.str_col) + len(self.data.columns) - len(self.origin_data.columns)
        self.pca = PCA(n_components=n_components)
        self.pca_features = self.pca.fit_transform(self.data[[feature_col]].values)
        return self.pca_features
class Model_AI:
    def __init__(self,dataset,setting):
        self.data= dataset
        self.setting = setting
        self.history = {}
        if self.setting["LogLoss"]:
            self.history["LogLoss"]={}
        if self.setting["F1"]:
            self.history["F1"]={}

    def fit(self):
        if self.setting["pca"]:
            X,y = self.data.get_feature_target(self.setting["feature_list"],self.setting["target"],True,self.setting["pca_n"])
        else:
            X,y = self.data.get_feature_target(self.setting["feature_list"],self.setting["target"])
        if self.setting["kfold"]:
            kf = KFold(n_splits=self.setting["K"])
            fold_id=0
            for train_index, test_index in kf.split(X):
                xtrain, xtest = X[train_index], X[test_index]
                ytrain, ytest = y[train_index], y[test_index]

                Model = LogisticRegression()
                Model.fit(xtrain,ytrain)
                yhat = Model.predict(xtest)
                yhat_p = Model.predict_proba(xtest)
                #print(yhat)
                if self.setting["LogLoss"]:
                    self.history["LogLoss"].update({fold_id:log_loss(ytest,yhat_p)})
                if self.setting["F1"]:
                    self.history["F1"].update({fold_id:f1_score(ytest,yhat, average='weighted')})
                fold_id+=1
        else:
            xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size = 1-self.setting["rate"])
            Model = LogisticRegression()
            Model.fit(xtrain,ytrain)
            yhat = Model.predict(xtest)
            yhat_p = Model.predict_proba(xtest)
            if self.setting["LogLoss"]:
                self.history["LogLoss"].update({0:log_loss(ytest,yhat_p)})
            if self.setting["F1"]:
                self.history["F1"].update({0:f1_score(ytest,yhat, average='weighted')})
        self.model = Model
        print(self.history)

    def plot_history(self):
        data = pd.DataFrame(self.history)
        labels = list(data.index)
        fig, ax = plt.subplots()
        title = []
        if len(labels)>1:
            labels +=["Mean"]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        try:
            F1 = [i for i in data["F1"].values]
            if "Mean" in labels:
                F1 +=[np.mean(data["F1"].values)]
            rects1= ax.bar(x - width/2, F1, width, label='F1', color="green")
            title.append("F1")
        except:
            pass

        try:
            LogLoss = [i for i in data["LogLoss"].values]
            if "Mean" in labels:
                LogLoss += [np.mean(data["LogLoss"].values)]
            rects2=ax.bar(x + width/2, LogLoss, width, label='LogLoss',color="blue")
            title.append("LogLoss")
        except:
            pass
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Error')
        #ax.set_yscale("log")
        ax.set_title(f'Error of {title[0] if len(title)<1 else "and".join(title)}')
        ax.set_xticks(x, labels)
        ax.legend()

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

