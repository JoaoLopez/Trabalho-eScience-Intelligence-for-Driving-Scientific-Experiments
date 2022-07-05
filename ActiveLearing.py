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

from intpy.intpy import initialize_intpy, deterministic

@initialize_intpy(__file__)
def main():
    ## read data & choose data
    melbourne_file_path = 'Data/20343.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    
#########ESSA PARTE PODE SER DETERMINÍSTICA
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
        temp = features[i]
        test_feature.append(temp.pop(ii))
        temp = labels[i]
        test_label.append(temp.pop(ii))

    far_y = []
    near_y = []
    for i in range(itera+1):
        tmp=[]
        far_y.append(copy.deepcopy(tmp))
        near_y.append(copy.deepcopy(tmp))
#########################################################################

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
        temp = features[i]
        temp.pop(r1)
        temp = features[i]
        temp.pop(r2)
        l1 = labels[i][r1]
        l2 = labels[i][r2]
        temp = labels[i]
        temp.pop(r1)
        temp = labels[i]
        temp.pop(r2)

        tmp_f = []
        tmp_f.append(f1)
        tmp_f.append(f2)
        temp = np.array(tmp_f)
        tmp_f = temp.reshape(-1, 1)
        far_model.fit(tmp_f, [l1, l2])
        near_model.fit(tmp_f, [l1, l2])
        temp = far_y[0]
        temp1 = np.array(test_feature[i])
        temp.append(far_model.predict(temp1.reshape(-1, 1))[0])
        temp = near_y[0]
        temp1 = np.array(test_feature[i])
        temp.append(near_model.predict(temp1.reshape(-1, 1))[0])

    ##################ESSA PARTE PODE SER DETERMINÍSTICA
        far = []
        near = []
        for dis in features[i]:
            far.append(min(abs(dis - f1), abs(dis - f2)))
            near.append(min(abs(dis - f1), abs(dis - f2)))
    ###################################################

        for j in range(0, itera):
            far_site = far.index(max(far)) 
            far[far_site] = 0
            temp = np.array(features[i][far_site])
            far_model.fit(temp.reshape(-1, 1), [labels[i][far_site]])
            temp = far_y[j+1]
            temp1 = np.array(test_feature[i])
            temp.append(far_model.predict(temp1.reshape(-1, 1))[0])
            
    ##################ESSA PARTE PODE SER DETERMINÍSTICA
            for k in range(len(features[i])):
                if k != far_site:
                    far[k] = min(far[k], abs(features[i][far_site]-features[i][k]))
    ################################################

            near_site = near.index(min(near)) 
            near[near_site] = 100
            temp = np.array(features[i][near_site])
            near_model.fit(temp.reshape(-1, 1), [labels[i][near_site]])
            temp = near_y[j+1]
            temp1 = np.array(test_feature[i])
            temp.append(near_model.predict(temp1.reshape(-1, 1))[0])
            
    ##################ESSA PARTE PODE SER DETERMINÍSTICA
            for k in range(len(features[i])):
                if (near[k] < 100) and (k != near_site):
                    near[k] = min(near[k], abs(features[i][near_site]-features[i][k]))
    #########################################

        temp = features[i]
        temp.append(f1)
        temp = features[i]
        temp.append(f2)
        temp = labels[i]
        temp.append(l1)
        temp = labels[i]
        temp.append(l2)

    ########################ESSA PARTE PODE SER DETERMINÍSTICA
    @deterministic
    def func1(itera, test_label, far_y, near_y):
        far_mse = []
        near_mse = []
        far_r2 = []
        near_r2 = []
        for i in range(itera+1):
            far_mse.append(mean_squared_error(test_label, far_y[i]))
            near_mse.append(mean_squared_error(test_label, near_y[i]))
            far_r2.append(r2_score(test_label, far_y[i]))
            near_r2.append(r2_score(test_label, near_y[i]))
        return far_mse, near_mse, far_r2, near_r2
    far_mse, near_mse, far_r2, near_r2 = func1(itera, test_label, far_y, near_y)
    #########################################

    iteration = [1,2,3,4,5,6]
    plt.plot(iteration, far_mse[1:], label="Choose a further point")
    plt.plot(iteration, near_mse[1:], label="Choose a closer point")
    # plt.plot(iteration,r2[1:])
    #############plt.show()
    plt.savefig("out/activeLearing.png")

main()