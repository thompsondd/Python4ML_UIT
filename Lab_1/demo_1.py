import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

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

print("\nCreating and training model")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#x_train.reshape((-1,1))
model.fit(x_train,y_train)
print("Create and train model successfully")

print("\nPredicting test data")
y_hat = model.predict(x_test)
print("Predict test data successfully")

print("\nEvaluating model")
score = r2_score(y_test,y_hat)
print(f"Evaluate model successfully: score={score}")          

fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
plt.scatter(x_test,y_test,color="r")
plt.plot(x_test,y_hat,color="b")
plt.show()