import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

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
        print(f"self.features_value:{self.features_value}")
        return self.target, self.features_value

    def get_feature_target(self,features_list,target,pca=False, pca_n=1):
        features_list+=[target]
        target_col,feature_col = self.process_data(self.origin_data[features_list])
        if pca:
            return self.get_pca_feature(pca_n,feature_col), self.data[[target_col]].values
        return self.data[feature_col].values, self.data[[target_col]].values


class Model_AI:
    def __init__(self,dataset,setting):
        self.data= dataset
        self.setting = setting
        self.AbsModel = RandomForestClassifier
        self.history = {}
        if self.setting["LogLoss"]:
            self.history["LogLoss"]={}
        if self.setting["F1"]:
            self.history["F1"]={}

    def fit(self, a=""):
        X,y = self.data.get_feature_target(self.setting["feature_list"],self.setting["target"])
        if self.setting["kfold"]:
            kf = KFold(n_splits=self.setting["K"], shuffle=True)
            fold_id=0
            print(f"{a} - X={X}")
            print(f"{a} - y={y}")
            for train_index, test_index in kf.split(X):
                scaler = MinMaxScaler()
                xtrain, xtest = X[train_index], X[test_index]
                xtrain = scaler.fit_transform(xtrain)
                xtest = scaler.transform(xtest)

                ytrain, ytest = y[train_index], y[test_index]

                if self.setting["pca"]:
                    PCA_Model = PCA(self.setting["pca_n"])
                    xtrain = PCA_Model.fit_transform(xtrain)

                Model = self.AbsModel()
                Model.fit(xtrain,ytrain)
                if self.setting["pca"]:
                    xtest = PCA_Model.transform(xtest)
                yhat = Model.predict(xtest)
                yhat_p = Model.predict_proba(xtest)
                #print(f"yhat1={yhat}")
                #print(f"yhat_p1={yhat_p}")
                if self.setting["LogLoss"]:
                    self.history["LogLoss"].update({fold_id:log_loss(ytest,yhat_p)})
                if self.setting["F1"]:
                    self.history["F1"].update({fold_id:f1_score(ytest,yhat, average='weighted')})
                fold_id+=1
        else:
            xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size = 1-self.setting["rate"])
            Model = self.AbsModel()

            if self.setting["pca"]:
                PCA_Model = PCA(self.setting["pca_n"])
                xtrain = PCA_Model.fit_transform(xtrain)

            Model.fit(xtrain,ytrain)

            if self.setting["pca"]:
                xtest = PCA_Model.transform(xtest)

            yhat = Model.predict(xtest)
            yhat_p = Model.predict_proba(xtest)
            print(f"yhat2={yhat}")
            print(f"yhat_p2={yhat_p}")
            if self.setting["LogLoss"]:
                print("yhat_p={yhat_p}")
                self.history["LogLoss"].update({0:log_loss(ytest,yhat_p)})
            if self.setting["F1"]:
                self.history["F1"].update({0:f1_score(ytest,yhat, average='weighted')})
        self.model = Model
        #print(self.history)
    @property
    def get_value_metrics(self):
        return {"f1":np.mean(list(self.history["F1"].values())), "LogLoss":np.mean(list(self.history["LogLoss"].values()))}

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
        ax.set_ylabel('Value')
        #ax.set_yscale("log")
        ax.set_title(f'{title[0] if len(title)<1 else " and ".join(title)}')
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

def search_PCA(dataset:Dataset,setting:dict, limit_component:int ) -> tuple:
    history ={"f1":{},"log_loss":{}}
    best = {"c":1,"f1":0}
    setting = setting.copy()
    for i in range(1,limit_component+1):
        setting["pca"] = True
        setting["pca_n"] = i
        model = Model_AI(dataset,setting)
        model.fit(a="Cal from search PCA")
        f1, log_loss = list(model.get_value_metrics.values())
        history["f1"].update({i:f1})
        history["log_loss"].update({i:log_loss})

        if history["f1"][i]>best["f1"]:
            best["c"] = i
            best["f1"] = history["f1"][i]

    data = pd.DataFrame(history)
    labels = list(data.index)
    fig, ax = plt.subplots()
    title = []
    try:
        F1 = [i for i in data["f1"].values]
        rects1= ax.bar(labels, F1, label='f1', color="green")
        ax.scatter(labels, F1, label='f1', color="green")
        title.append("f1")
    except:
        pass
    try:
        LogLoss = [i for i in data["log_loss"].values]
        rects2=ax.plot(labels, LogLoss, label='log_loss',color="blue")
        ax.scatter(labels, LogLoss, label='log_loss',color="blue")
        title.append("log_loss")
    except:
        pass
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 Score')
    #ax.set_yscale("log")
    ax.set_title("F1 score with different component")
    #ax.set_xticks(x, labels)
    ax.legend()
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    return fig,best