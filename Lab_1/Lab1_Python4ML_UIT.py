import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Salary Data
print(f"{'-'*50}\nLoading data: Salary_Data.csv")
path_data = os.path.join(r".\data\Salary_Data.csv")
data = pd.read_csv(path_data)
x = data["YearsExperience"].values.reshape((-1,1))
y = data["Salary"].values
print("Load data successfully\n")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.4)

svr_model = SVR(kernel="poly", degree=4)
svr_model.fit(x_train,y_train)
y_hat = svr_model.predict(x_test)
print(f"Evaluate model SVR: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
y_hat = linear_model.predict(x_test)
print(f"Evaluate model LinearRegression: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

rfr_model = RandomForestRegressor(n_estimators=30)
rfr_model.fit(x_train,y_train)
y_hat = rfr_model.predict(x_test)
print(f"Evaluate model RandomForestRegressor: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

#-----------------------------------------------------------------------------------

print(f"{'-'*50}\nLoading data: Position_Salaries.csv")
path_data = os.path.join(r".\data\Position_Salaries.csv")
data = pd.read_csv(path_data)
x = data["Level"].values.reshape((-1,1))
y = data["Salary"].values
print("Load data successfully\n")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.4)

svr_model = SVR(kernel="poly", degree=6)
svr_model.fit(x_train,y_train)
y_hat = svr_model.predict(x_test)
print(f"Evaluate model SVR: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
y_hat = linear_model.predict(x_test)
print(f"Evaluate model LinearRegression: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

rfr_model = RandomForestRegressor(n_estimators=500)
rfr_model.fit(x_train,y_train)
y_hat = rfr_model.predict(x_test)
print(f"Evaluate model RandomForestRegressor: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

#-----------------------------------------------------------------------------------

print(f"{'-'*50}\nLoading data: 50_Startups.csv")
path_data = os.path.join(r".\data\50_Startups.csv")
encoder = LabelEncoder()
data = pd.read_csv(path_data)
data["Label_State"] = encoder.fit_transform(data["State"])
x = data[["R&D Spend","Administration","Marketing Spend","Label_State"]].values
y = data["Profit"].values
print("Load data successfully\n")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.4)

svr_model = SVR(kernel="poly", degree=5)
svr_model.fit(x_train,y_train)
y_hat = svr_model.predict(x_test)
print(f"Evaluate model SVR: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
y_hat = linear_model.predict(x_test)
print(f"Evaluate model LinearRegression: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

rfr_model = RandomForestRegressor(n_estimators=500)
rfr_model.fit(x_train,y_train)
y_hat = rfr_model.predict(x_test)
print(f"Evaluate model RandomForestRegressor: \n\tr2: {r2_score(y_test,y_hat)}\n\tmse: {mean_squared_error(y_test,y_hat)}\n")

'''
Thông qua thực nghiệm, mô hình RandomForestRegressor luôn đưa các dự đoán khá tốt trong cả 3 tập dữ liệu, ruy nhiên cần điều chỉnh tham số n_estimators có kết quả tốt nhất.
Đối với mô hình LinearRegression thì chỉ hoạt động tốt đối với tập dữ liệu Salary_Data.csv, còn đối với các tập dữ liệu còn lại thì mô hình hoạt động không tốt.
Đối với mô hình SVM cho regression thì chỉ hoạt động không tốt đối với tập dữ liệu Salary_Data.csv, còn đối với các bộ dữ liệu còn lại thì mô hình cho ra các kết quả dự đoán khá tốt. 
'''