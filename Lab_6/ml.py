import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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
        return self.target, list(filter(lambda x: x!=self.target, self.data.columns))
    def get_feature_target(self,features_list,target):
        features_list+=[target]
        target_col,feature_col = self.process_data(self.origin_data[features_list])
        return self.data[feature_col].values, self.data[[target_col]].values
class Model_AI:
    def __init__(self,dataset,setting):
        self.data= dataset
        self.setting = setting
        self.history = {}
        if self.setting["MAE"]:
            self.history["MAE"]={}
        if self.setting["MSE"]:
            self.history["MSE"]={}

    def fit(self):
        X,y = self.data.get_feature_target(self.setting["feature_list"],self.setting["target"])
        if self.setting["kfold"]:
            kf = KFold(n_splits=self.setting["K"])
            fold_id=0
            for train_index, test_index in kf.split(X):
                xtrain, xtest = X[train_index], X[test_index]
                ytrain, ytest = y[train_index], y[test_index]

                Linear_model = LinearRegression()
                Linear_model.fit(xtrain,ytrain)
                yhat = Linear_model.predict(xtest)
                
                if self.setting["MAE"]:
                    self.history["MAE"].update({fold_id:mean_absolute_error(ytest,yhat)})
                if self.setting["MSE"]:
                    self.history["MSE"].update({fold_id:mean_squared_error(ytest,yhat)})
                fold_id+=1
        else:
            xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size = 1-self.setting["rate"])
            Linear_model = LinearRegression()
            Linear_model.fit(xtrain,ytrain)
            yhat = Linear_model.predict(xtest)
            if self.setting["MAE"]:
                self.history["MAE"].update({0:mean_absolute_error(ytest,yhat)})
            if self.setting["MSE"]:
                self.history["MSE"].update({0:mean_squared_error(ytest,yhat)})
        self.model = Linear_model
        print(self.history)

    def plot_history(self):
        data = pd.DataFrame(self.history)
        labels = list(data.index)
        mse = [i for i in data["MSE"].values]
        mae = [i for i in data["MAE"].values]
        if len(labels)>1:
            labels +=["Mean"]
            mse +=[np.mean(data["MSE"].values)]
            mae += [np.mean(data["MAE"].values)]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1= ax.bar(x - width/2, mse, width, label='MSE', color="green")
        rects2=ax.bar(x + width/2, mae, width, label='MAE',color="blue")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Error')
        ax.set_yscale("log")
        ax.set_title('Error of MSE and MAE')
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

