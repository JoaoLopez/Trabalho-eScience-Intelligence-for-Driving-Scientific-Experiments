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
import random
import copy

def ActiveLearningDiffPredict(file_path):
    ## read data & choose data
    data = pd.read_csv(file_path)
    feature_array = data["Small Mol Concentration (uM)"]
    label_array = data["Increased Fraction Dead"]
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

    # devide into test part and train part
    test_feature = []
    test_label = []
    for i in range(len(features)):
        ii = i % size
        test_feature.append(features[i].pop(ii))
        test_label.append(labels[i].pop(ii))

    # init y 
    # y1 = []
    # y2 = []
    # y3 = []
    # # y4 = []
    # y5 = []
    y6 = []
    for i in range(itera+1):
        tmp=[]
        # y1.append(copy.deepcopy(tmp))
        # y2.append(copy.deepcopy(tmp))
        # y3.append(copy.deepcopy(tmp))
        # # y4.append(copy.deepcopy(tmp))
        # y5.append(copy.deepcopy(tmp))
        y6.append(copy.deepcopy(tmp))
        

    for i in range(len(features)):

        # init models
        # mo1 = linear_model.LinearRegression()
        # mo2 = linear_model.Ridge(alpha=.5)
        # mo3 = DecisionTreeRegressor(max_depth=3)
        # mo4 = neighbors.KNeighborsRegressor()
        # mo5 = svm.SVR()
        mo6 = ensemble.RandomForestRegressor(n_estimators=20)

        # select 2 sample randomly
        r1, r2 = random.sample(range(size-2),2)
        f1 = features[i][r1]
        f2 = features[i][r2]
        features[i].pop(r1)
        features[i].pop(r2)
        l1 = labels[i][r1]
        l2 = labels[i][r2]
        labels[i].pop(r1)
        labels[i].pop(r2)

        # the first time of iteration
        tmp_f = []
        tmp_f.append(f1)
        tmp_f.append(f2)
        tmp_f = np.array(tmp_f).reshape(-1, 1)
        # mo1.fit(tmp_f, [l1, l2])
        # mo2.fit(tmp_f, [l1, l2])
        # mo3.fit(tmp_f, [l1, l2])
        # # mo4.fit(tmp_f, [l1, l2])
        # mo5.fit(tmp_f, [l1, l2])
        mo6.fit(tmp_f, [l1, l2])
        # y1[0].append(mo1.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        # y2[0].append(mo2.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        # y3[0].append(mo3.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        # # y4[0].append(mo4.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        # y5[0].append(mo5.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        y6[0].append(mo6.predict(np.array(test_feature[i]).reshape(-1, 1))[0])

        # to record the Distance
        far = []
        for dis in features[i]:
            far.append(min(abs(dis - f1), abs(dis - f2)))
        
        for j in range(0, itera):
            # find the furthest point
            far_site = far.index(max(far)) 
            far[far_site] = 0
            # mo1.fit(np.array(features[i][far_site]).reshape(-1, 1), [labels[i][far_site]])
            # mo2.fit(np.array(features[i][far_site]).reshape(-1, 1), [labels[i][far_site]])
            # mo3.fit(np.array(features[i][far_site]).reshape(-1, 1), [labels[i][far_site]])
            # # mo4.fit(np.array(features[i][far_site]).reshape(-1, 1), [labels[i][far_site]])
            # mo5.fit(np.array(features[i][far_site]).reshape(-1, 1), [labels[i][far_site]])
            mo6.fit(np.array(features[i][far_site]).reshape(-1, 1), [labels[i][far_site]])
            # y1[j+1].append(mo1.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
            # y2[j+1].append(mo2.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
            # y3[j+1].append(mo3.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
            # # y4[j+1].append(mo4.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
            # y5[j+1].append(mo5.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
            y6[j+1].append(mo6.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
            # update the most cloest distance of every untrained point to all the points that have been trained
            for k in range(len(features[i])):
                if k != far_site:
                    far[k] = min(far[k], abs(features[i][far_site]-features[i][k]))

        # put the two random points back
        features[i].append(f1)
        features[i].append(f2)
        labels[i].append(l1)
        labels[i].append(l2)

    ## caculate the MSE and R-square
    # mse1 = []
    # mse2 = []
    # mse3 = []
    # # mse4 = []
    # mse5 = []
    mse6 = []
    r2 = []
    for i in range(itera+1):
        # mse1.append(mean_squared_error(test_label, y1[i]))
        # mse2.append(mean_squared_error(test_label, y2[i]))
        # mse3.append(mean_squared_error(test_label, y3[i]))
        # # mse4.append(mean_squared_error(test_label, y4[i]))
        # mse5.append(mean_squared_error(test_label, y5[i]))
        mse6.append(mean_squared_error(test_label, y6[i]))
        r2.append(r2_score(test_label, y6[i]))

    ## draw
    iteration = list(range(1, itera+1))
    plt.figure()
    plt.title('Random Forest Regressor')
    plt.xlabel('Iteration Times')
    plt.ylabel('Mean Square Errors')
    # plt.plot(iteration, mse1[1:], label="LinearRegression")
    # plt.plot(iteration, mse2[1:], label="Ridge")
    # plt.plot(iteration, mse3[1:], label="Decision Tree Regressor")
    # # plt.plot(iteration, mse4[1:], label="K-Neighbors Regressor")
    # plt.plot(iteration, mse5[1:], label="SVM")
    plt.plot(iteration, mse6[1:], label="Mean Squared Error")
    plt.plot(iteration, r2[1:], label="R-square")
    plt.legend()
    plt.show()

