import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


# Salary Data
print("Loading data")
path_data = os.path.join(r".\data\Salary_Data.csv")
data = pd.read_csv(path_data)
x = data[["YearsExperience"]]
y = data[["Salary"]]
print("Load data successfully")

print("\nSpliting data")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
print("Split data successfully")

def Train_and_plot_model(*data):
    dict_result={}
    x_train, x_test, y_train, y_test = data
    #for info in [("Linear",LinearRegression),("SVR",SVR),("RFR",RandomForestRegressor())]:
    print("\nCreating and training model")
    linear_model = LinearRegression()
    #x_train.reshape((-1,1))
    linear_model.fit(x_train,y_train)
    print("Create and train model successfully")

    print("\nPredicting test data")
    y_linear_hat = linear_model.predict(x_test)
    print("Predict test data successfully")

    print("\nEvaluating model")
    score = r2_score(y_test,y_hat)
    print(f"Evaluate model successfully: score={score}")          
    dict_result.update({"linear":{"x_test":x_test,"y_test":y_test,"y_hat":y_hat}})

    fig, ax = plt.subplots(1,3, figsize=(30,5))

    ax[0].scatter(x_test,y_test,color="r")
    ax[0].plot(x_test,y_hat,color="b")
    ax[0].set_title("Linear Regression - r2:{score}")
    ax[0].set_xlabel("YearsExperience")
    ax[0].set_ylabel("Salary")

    plt.show()