import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



# Salary Data
print("Loading data")
path_data = os.path.join(r".\data\Salary_Data.csv")
data = pd.read_csv(path_data)
x = data[["YearsExperience"]]
y = data[["Salary"]]
print("Load data successfully")

def train_plot_svr(*data):
    x,y = data
    history_svr = {}
    n=100
    s,e=1,9
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
        for degree in range(s,e):
            svr_model = SVR(kernel="poly", degree=degree)
            #x_train.reshape((-1,1))
            svr_model.fit(x_train,y_train)
            #print("Create and train linear model successfully")

            #print("\nPredicting test data")
            y_svr_hat = svr_model.predict(x_test)
            #print("Predict test data successfully")

            #print("\nEvaluating model")
            score = r2_score(y_test,y_svr_hat)
            #print(f"Evaluate model successfully: score={score}")
            if degree not in history_svr.keys():
                history_svr.update({degree:[]})
            history_svr[degree].append(score)
        print(f"{i*100/n}%")

    fig, ax = plt.subplots(1,2, figsize=(20,5), num="SVM For Regression")
    stats = np.array(list(map(lambda x: history_svr[x],history_svr.keys())))
    print(f"stats - {stats.shape}")
    means = np.mean(stats,axis=1)
    print(f"means - {means.shape}: {means}")
    std = np.std(stats,axis=1)
    print(f"std - {std.shape}: {std}")

    print(np.argmax(means)+1)

    svr_model = SVR(kernel="poly", degree=np.argmax(means)+1)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
    svr_model.fit(x_train,y_train)
    y_hat = svr_model.predict(np.sort(x_test, axis = 0))

    ax[0].plot(list(history_svr.keys()),means)
    ax[0].errorbar(list(history_svr.keys()), means, std, fmt='o', linewidth=2, capsize=6)
    ax[0].set_xlabel("Degree of poly")
    ax[0].set_ylabel("Mean of R2")

    ax[1].scatter(x_test,y_test,color="r")
    ax[1].plot(np.sort(x_test, axis = 0), y_hat )
    ax[1].set_xlabel("Years Experience")
    ax[1].set_ylabel("Salary")
    ax[1].set_title(f"r2: {r2_score(y_test,y_hat)}")
    #fig.canvas.set_window_title("SVM For Regression")
    plt.show()

def train_plot_linear(*data):
    x,y = data
    history = []
    n=100
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
        linear_model = LinearRegression()
        linear_model.fit(x_train,y_train)
        y_hat = linear_model.predict(x_test)
        score = r2_score(y_test,y_hat)
        history.append(score)
        print(f"{i*100/n}%")

    fig, ax = plt.subplots(1,2, figsize=(20,5),num="Linear Regression")
    stats = np.array(history)
    print(f"stats - {stats.shape}")
    means = np.mean(stats)
    print(f"means - {means.shape}: {means}")
    std = np.std(stats)
    print(f"std - {std.shape}: {std}")

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
    linear_model = LinearRegression()
    linear_model.fit(x_train,y_train)
    y_hat = linear_model.predict(x_test)
    score = r2_score(y_test,y_hat)

    ax[0].errorbar(1, means, std, fmt='o', linewidth=2, capsize=6)
    ax[0].set_ylabel("Mean of R2")

    ax[1].scatter(x_test,y_test,color="r")
    ax[1].plot(x_test, y_hat )
    ax[1].set_xlabel("Years Experience")
    ax[1].set_ylabel("Salary")
    ax[1].set_title(f"r2: {score}")
    #fig.canvas.set_window_title("Linear Regression")

    plt.show()

#train_plot_svr(x,y)

def train_plot_rfr(*data):
    x,y = data
    history = {}
    n=10
    s,e,st=5,81, 10
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
        for estimators in range(s,e,st):
            rfr_model = RandomForestRegressor(n_estimators=estimators)
            rfr_model.fit(x_train,y_train)
            y_rfr_hat = rfr_model.predict(x_test)
            score = r2_score(y_test,y_rfr_hat)

            if estimators not in history.keys():
                history.update({estimators:[]})
            history[estimators].append(score)
        print(f"{i*100/n}%")

    fig, ax = plt.subplots(1,2, figsize=(20,5), num="Randomforest Regression")
    stats = np.array(list(map(lambda x: history[x],history.keys())))
    print(f"stats - {stats.shape}")
    means = np.mean(stats,axis=1)
    print(f"means - {means.shape}: {means}")
    std = np.std(stats,axis=1)
    print(f"std - {std.shape}: {std}")

    print(list(history.keys())[np.argmax(means)])

    rfr_model = RandomForestRegressor(n_estimators=list(history.keys())[np.argmax(means)])
    rfr_model.fit(x_train,y_train)
    y_rfr_hat = rfr_model.predict(x_test)
    score = r2_score(y_test,y_rfr_hat)

    ax[0].plot(list(history.keys()),means)
    ax[0].errorbar(list(history.keys()), means, std, fmt='o', linewidth=2, capsize=6)
    ax[0].set_xlabel("number of estimators")
    ax[0].set_ylabel("Mean of R2")

    ax[1].scatter(x_test,y_test,color="r")
    ax[1].step(np.sort(x_test, axis = 0), y_rfr_hat )
    ax[1].set_xlabel("Years Experience")
    ax[1].set_ylabel("Salary")
    ax[1].set_title(f"r2: {score}")
    #fig.canvas.set_window_title("Randomforest Regression")
    plt.show()

train_plot_rfr(x,y)