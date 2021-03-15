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

def ActiveLearningDiffQuery(file_path):
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

    # init y predicted by two different ways and a random way
    far_y = []
    near_y = []
    random_y = []
    for i in range(itera+1):
        tmp=[]
        far_y.append(copy.deepcopy(tmp))
        near_y.append(copy.deepcopy(tmp))
        random_y.append(copy.deepcopy(tmp))


    for i in range(len(features)):

        far_model = linear_model.LinearRegression()
        near_model = linear_model.LinearRegression()
        random_model = linear_model.LinearRegression()

        # put 2 random points into training
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
        random_model.fit(tmp_f, [l1, l2])
        far_y[0].append(far_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        near_y[0].append(near_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
        random_y[0].append(random_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])

        far = []
        near = []
        # storage the most cloest distance of every untrained point 
        # to all the points (now, they are f1 and f2) that have been trained
        for dis in features[i]:
            far.append(min(abs(dis - f1), abs(dis - f2)))
            near.append(min(abs(dis - f1), abs(dis - f2)))

        ran = list(range(0, itera))
        
        # 6 times iteraion for every group
        for j in range(0, itera):

            # find the furthest point
            far_site = far.index(max(far)) 
            # make the point unavailable
            far[far_site] = 0
            # use the point to train
            far_model.fit(np.array(features[i][far_site]).reshape(-1, 1), [labels[i][far_site]])
            far_y[j+1].append(far_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
            # update the most cloest distance of every untrained point to all the points that have been trained
            for k in range(len(features[i])):
                if k != far_site:
                    far[k] = min(far[k], abs(features[i][far_site]-features[i][k]))

            # find the closest point
            near_site = near.index(min(near)) 
            # make the point unavailable
            near[near_site] = 100
            # train
            near_model.fit(np.array(features[i][near_site]).reshape(-1, 1), [labels[i][near_site]])
            near_y[j+1].append(near_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])
            # update
            for k in range(len(features[i])):
                if (near[k] < 100) and (k != near_site):
                    near[k] = min(near[k], abs(features[i][near_site]-features[i][k]))

            # find a random point
            ran_site = random.choice(ran)
            # drop it
            ran.remove(ran_site)
            # train
            random_model.fit(np.array(features[i][ran_site]).reshape(-1, 1), [labels[i][ran_site]])
            random_y[j+1].append(random_model.predict(np.array(test_feature[i]).reshape(-1, 1))[0])



        # put the two random points back
        features[i].append(f1)
        features[i].append(f2)
        labels[i].append(l1)
        labels[i].append(l2)

    ## caculate the mean square error
    far_mse = []
    near_mse = []
    random_mse = []
    for i in range(itera+1):
        far_mse.append(mean_squared_error(test_label, far_y[i]))
        near_mse.append(mean_squared_error(test_label, near_y[i]))
        random_mse.append(mean_squared_error(test_label, random_y[i]))

    # draw
    iteration = list(range(1, itera+1))
    plt.figure()
    plt.title('Different Query Algorithms')
    plt.xlabel('Iteration Times')
    plt.ylabel('Mean Square Errors')
    plt.plot(iteration, far_mse[1:], label="Choose a furthest point")
    plt.plot(iteration, near_mse[1:], label="Choose a closest point")
    plt.plot(iteration, random_mse[1:], label="Choose a random point")
    plt.legend()
    plt.show()

