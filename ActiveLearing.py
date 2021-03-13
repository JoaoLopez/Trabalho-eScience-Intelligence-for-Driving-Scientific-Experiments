import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn import neighbors
from sklearn import ensemble
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
# from sklearn import linear_model, datasets
# from sklearn.cross_validation import train_test_split
# import imageio as io
# import os
import random
import copy

## read data & choose data
melbourne_file_path = 'Data/20343.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
feature_array = melbourne_data["Small Mol Concentration (uM)"]
label_array = melbourne_data["Increased Fraction Dead"]
# the size of data in a same model
size = 9
# iteration times
itera = size - 3
# itera = 4

## get every feature matrix and label vector
features = []
labels = []
tmp_i = 0
tmp_f = []
for f in feature_array:
    tmp_f.append(f)
    if tmp_i % size == size - 1:
        # arr_f = np.array(tmp_f)#.reshape(-1, 1)
        features.append(tmp_f)
        tmp_f = []
    tmp_i += 1
tmp_i = 0
tmp_l = []
for l in label_array:
    tmp_l.append(l)
    if tmp_i % size == size - 1:
        # arr_l = np.array(tmp_l)
        labels.append(tmp_l)
        tmp_l = []
    tmp_i += 1

test_feature = []
test_label = []
for i in range(len(features)):
    ii = i % size
    test_feature.append(features[i].pop(ii))
    test_label.append(labels[i].pop(ii))

far_y = []
near_y = []
for i in range(itera+1):
    tmp=[]
    far_y.append(copy.deepcopy(tmp))
    near_y.append(copy.deepcopy(tmp))

for i in range(len(features)):

    # far_model = linear_model.LinearRegression()
    # near_model = linear_model.LinearRegression()
    # model = DecisionTreeRegressor(max_depth=3)
    # model = neighbors.KNeighborsRegressor()
    # model = svm.SVR()
    # model = linear_model.Ridge(alpha=.5)
    far_model = ensemble.RandomForestRegressor(n_estimators=20)
    near_model = ensemble.RandomForestRegressor(n_estimators=20)

    r1, r2 = random.sample(range(size-2),2)
    f1 = features[i][r1]
    f2 = features[i][r2]
    features[i].pop(r1)
    features[i].pop(r2)
    l1 = labels[i][r1]
    l2 = labels[i][r2]
    labels[i].pop(r1)
    labels[i].pop(r2)

    tmp_f = []
    tmp_f.append(f1)
    tmp_f.append(f2)
    tmp_f = np.array(tmp_f).reshape(-1, 1)
    far_model.fit(tmp_f, [l1, l2])
    near_model.fit(tmp_f, [l1, l2])
    far_y[0].append(far_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
    near_y[0].append(near_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])

    far = []
    near = []
    for dis in features[i]:
        far.append(min(abs(dis - f1), abs(dis - f2)))
        near.append(min(abs(dis - f1), abs(dis - f2)))
    
    for j in range(0, itera):
        far_site = far.index(max(far)) 
        far[far_site] = 0
        far_model.fit(np.array(features[i][far_site]).reshape(-1, 1), [labels[i][far_site]])
        far_y[j+1].append(far_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        for k in range(len(features[i])):
            if k != far_site:
                far[k] = min(far[k], abs(features[i][far_site]-features[i][k]))

        near_site = near.index(min(near)) 
        near[near_site] = 100
        near_model.fit(np.array(features[i][near_site]).reshape(-1, 1), [labels[i][near_site]])
        near_y[j+1].append(near_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        for k in range(len(features[i])):
            if (near[k] < 100) and (k != near_site):
                near[k] = min(near[k], abs(features[i][near_site]-features[i][k]))


    features[i].append(f1)
    features[i].append(f2)
    labels[i].append(l1)
    labels[i].append(l2)

far_mse = []
near_mse = []
far_r2 = []
near_r2 = []
for i in range(itera+1):
    far_mse.append(mean_squared_error(test_label, far_y[i]))
    near_mse.append(mean_squared_error(test_label, near_y[i]))
    far_r2.append(r2_score(test_label, far_y[i]))
    near_r2.append(r2_score(test_label, near_y[i]))

iteration = [1,2,3,4,5,6]
plt.plot(iteration, far_mse[1:])
plt.plot(iteration, near_mse[1:])
# plt.plot(iteration,r2[1:])
plt.show()

