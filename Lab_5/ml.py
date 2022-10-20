import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

class Model_AI:
    def __init__(self):
        self.data=None
        self.encoder={}
        self.best_model=None
        self.features = []
        self.str_col=[]
    def check_str_type(self, word):
        return str(type(word)).split()[1].split("'")[1] == "str"

    def process_data(self, data_path):
        self.data = pd.read_csv(data_path)
        self.origin_data = self.data.copy()
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
        return (self.target,list(filter(lambda x: x!=self.target, self.data.columns)))

    def fit(self, data_path):
        rank={}
        target, feature = self.process_data(data_path)
        X = self.data[feature].values
        y = self.data[[target]].values
        xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size = 0.3)
        Linear_model = LinearRegression()
        Linear_model.fit(xtrain,ytrain)
        rank.update({Linear_model.score(xtest,ytest):Linear_model})
        
        svr_model = SVR(kernel="poly", degree=5)
        svr_model.fit(xtrain,ytrain)
        rank.update({svr_model.score(xtest,ytest):svr_model})
        
        rfr_model = RandomForestRegressor(n_estimators=500)
        rfr_model.fit(xtrain,ytrain)
        rank.update({rfr_model.score(xtest,ytest):rfr_model})
        self.best_model = rank[max(list(rank.keys()))]

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

