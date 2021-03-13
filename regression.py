import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from sklearn import ensemble
# from sklearn import linear_model, datasets
# from sklearn.cross_validation import train_test_split
# import imageio as io
# import os
import random

## read data & choose data
melbourne_file_path = 'Data/20343.csv'
# melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = pd.read_csv(melbourne_file_path).loc[lambda x: x['Small Molecule Name'] == 'Abemaciclib'].loc[lambda x: x['Cell Name'] == 'HCC1806']
feature_array = melbourne_data["Small Mol Concentration (uM)"]
label_array = melbourne_data["Increased Fraction Dead"]

features = []
labels = []
for f in feature_array:
    features.append(f)
for l in label_array:
    labels.append(l)
features = np.array(features).reshape(-1, 1)
labels = np.array(labels)

# model1 = DecisionTreeRegressor(max_depth=3)
# model2 = neighbors.KNeighborsRegressor()
# model3 = linear_model.LinearRegression()
# model4 = model_SVR = svm.SVR()
# model5 = ensemble.RandomForestRegressor(n_estimators=20)
# model4 = linear_model.Ridge(alpha=.5)
model1 = DecisionTreeRegressor(max_depth=1)
model2 = DecisionTreeRegressor(max_depth=2)
model3 = DecisionTreeRegressor(max_depth=3)
model4 = DecisionTreeRegressor(max_depth=4)
model5 = DecisionTreeRegressor(max_depth=5)
model1.fit(features, labels)
model2.fit(features, labels)
model3.fit(features, labels)
model4.fit(features, labels)
model5.fit(features, labels)

X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]
y_1 = model1.predict(X_test)
y_2 = model2.predict(X_test)
y_3 = model3.predict(X_test)
y_4 = model4.predict(X_test)
y_5 = model5.predict(X_test)

# reg = linear_model.LinearRegression()
# reg = linear_model.Ridge(alpha=.5)
# reg.fit (features,labels)

# plt.plot(features, y_predict, 'g-', linewidth=1, label='预测值')
# plt.scatter(feature_array, y_predict, alpha=0.1, label='预测值')
# plt.scatter(feature_array, label_array, alpha=0.2, label='真实值')
# plt.xlabel('Concentration (uM)')
# plt.ylabel('Increased Fraction Dead')

plt.figure()
plt.scatter(features, labels, s=20, edgecolor="black",c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=1", linewidth=2)
plt.plot(X_test, y_2, color="yellow", label="knn2", linewidth=2)
plt.plot(X_test, y_3, color='red', label='liner regression3', linewidth=3)
plt.plot(X_test, y_4, color='green', label='svm5', linewidth=2)
plt.plot(X_test, y_5, color='brown', label='forest10', linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

