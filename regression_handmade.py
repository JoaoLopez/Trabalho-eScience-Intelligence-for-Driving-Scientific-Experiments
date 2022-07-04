import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from intpy.intpy import initialize_intpy, deterministic

class LinearRegression(object):
    def __init__(self):
        self.parameter = []
        self.a = 0.001
        self.iters = 1

    def regress_GD(self, feature, label):
################################ESSA PARTE É DETERMINÍSTICA
        # convert to array type
        X = np.array(feature)
        # add the constant(x**0)
        X = np.insert(X, 0, 1, axis=1)
        Y = np.array(label)
        pa = np.zeros(X.shape[1])

        # gradient descent
        for i in range(0, self.iters):
            for x, y in zip(X, Y):
                loss = y - np.dot(x, pa)
                pa = pa + np.multiply(self.a + loss, x)
#####################################################
        self.parameter = np.mat(pa).T
        return self.parameter

    def regress_matrix(self, feature, label):
################################ESSA PARTE É DETERMINÍSTICA
        # convert to a feature matrix
        X = np.mat(feature)
        # add the constant(x**0)
        X = np.insert(X, 0, 1, axis=1)
        # convert to a label vector
        Y = np.mat(label)
        XtX = X.T * X
        # in case XtX can not be inversed
        if np.linalg.det(XtX) == 0.0:
            print('Error: Maybe you should change the way')
            return
#######################################        
        # use the formula to calculate parameters
        self.parameter = XtX.I * X.T * Y
        return self.parameter
    
    ## choose the regressing way
    def regress(self, feature, label, type='matrix'):
        if type == 'matrix':
            return self.regress_matrix(feature, label)
        elif type == 'GD':
            return self.regress_GD(feature, label)

    def predict(self, feature):
###########################ESSA PARTE É DETERMINÍSTICA
        x = np.array(feature)
        x = np.insert(x, 0, 1, axis=1)
        pa = np.array(self.parameter)
        return np.dot(x, self.parameter)
##############################
    ## clear the data
    def clear(self):
        self.parameter = []

@initialize_intpy(__file__)
def main():
    ## read data & choose data
    melbourne_file_path = 'Data/20343.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    # melbourne_data = pd.read_csv(melbourne_file_path).loc[lambda x: x['Small Molecule Name'] == 'Abemaciclib'].loc[lambda x: x['Cell Name'] == 'HCC1806']
    feature_array = melbourne_data["Small Mol Concentration (uM)"]
    label_array = melbourne_data["Increased Fraction Dead"]
    # the size of data in a same model
    size = 9

    ## get every feature matrix and label vector
    lr = LinearRegression()

############ESSA PARTE PODE SER DETERMINÍSTICA
    @deterministic
    def f1(feature_array, label_array, size):
        features = []
        labels = []
        tmp_i = 0
        tmp_f = []
        for f in feature_array:
            tmp_f.append(f)
            if tmp_i % size == size - 1:
                mat_f = np.mat(tmp_f).T
                features.append(mat_f)
                tmp_f = []
            tmp_i += 1
        tmp_i = 0
        tmp_l = []
        for l in label_array:
            tmp_l.append(l)
            if tmp_i % size == size - 1:
                mat_l = np.mat(tmp_l).T
                labels.append(mat_l)
                tmp_l = []
            tmp_i += 1
        return features, labels
    features, labels = f1(feature_array, label_array, size)
###############################

    ## N-fold cross-validation
    # record the real value and the predictive value
    loss = []
    for j in range(len(features)):
        feature = features[j]
        label = labels[j]
        # divide into n sets
        n = feature.shape[0]
        for i in range(n):
            # test part
            test_feature = feature[0]
            test_label = label[0]
            # delete the test part then remain the training part
            feature = np.delete(feature, 0, 0)
            label = np.delete(label, 0, 0)

            # train & predict
            # lr.regress(feature, label)
            lr.regress(feature, label, 'GD')
            predict_label = lr.predict(test_feature)
            loss.append((np.array(test_label)[0][0],np.array(predict_label)[0][0]))

            # restore
            lr.clear()
            feature = np.append(feature, test_feature, 0)
            label = np.append(label, test_label, 0)

    # calculation for Receiver-Operating Characteristic curve
    roc_x = []
    roc_y = []
    roc_y_random = []
    for ii in range(1000):
        # Allowable error range
        i = 1.0 / 1000 * ii
        # hit rate count
        hit_cnt = 0
        hit_cnt_random = 0

        for test_y,predict_y in loss:
            random_y = 1.0 * random.randint(0,20000) / 10000 - 1
            if abs(test_y - predict_y) < i:
                # hit
                hit_cnt += 1
            if abs(test_y - random_y) < i:
                # hit
                hit_cnt_random += 1

        roc_x.append(i)
        roc_y.append(1.0 * hit_cnt / len(loss))
        roc_y_random.append(1.0 * hit_cnt_random / len(loss))

    # draw Receiver-Operating Characteristic curve
    plt.title('Receiver-Operating Characteristic curve')
    plt.plot(roc_x, roc_y)
    plt.plot(roc_x, roc_y_random)
    ########################plt.show()
    plt.savefig("out/regression_handmade.png")

    # plt.scatter(feature_array, label_array, alpha=0.2)
    # plt.xlabel('Concentration (uM)')
    # plt.ylabel('Increased Fraction Dead')
    # feature = np.insert(feature, 0, 1, axis=1)
    # x = feature.copy()
    # y = x * p
    # plt.plot(x,y)
    # plt.show()

main()