import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading data")
data = pd.read_csv(r"C:\Users\ACER\Desktop\Public_danhnt_python4ML\Salary_Data.csv")
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
from sklearn.metrics import r2_score
score = r2_score(y_test,y_hat)
print(f"Evaluate model successfully: score={score}")          

plt.scatter(x_train,y_train,color="r")
plt.plot(x_test,y_hat,color="b")
plt.show()