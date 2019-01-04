# -*- coding: utf-8 -*-
# base: https://gogul09.github.io/software/image-classification-python
# organize imports
# import mahotas
import cv2
import os
import h5py
import random
import numpy as np
import pandas as pd
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
from IPython.display import display

from matplotlib import pyplot
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

# no.of.trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.20

# seed for reproducing same results
seed = 9

def loads_data(dict_labels):
    features = np.load("cars.npz")

    # train_files = []
    # y_train = []
    # i=0

    # global_features = np.take(list(features.values()), 0)
    # global_features = list(features.values())[0][0]
    global_features = []
    labels = []

    for file in features:
        # print 'file: ', file
        global_features.append(features[file][0])
        label_in_file = dict_labels[file[:4]]
        labels.append(int(label_in_file))

    # get the overall feature vector size
    print "[STATUS] feature vector size {}".format(np.array(global_features).shape)
    # print 'global_features', global_features

    # get the overall training label size
    print "[STATUS] training Labels {}".format(np.array(labels).shape)

    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print "[STATUS] training labels encoded..."

    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print "[STATUS] feature vector normalized..."

    print "[STATUS] target labels: {}".format(target)
    print "[STATUS] target labels shape: {}".format(target.shape)

    # save the feature vector using HDF5
    h5f_data = h5py.File('output/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File('output/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    print "[STATUS] end of training.."

    train_test()

def train_test():
    
    # create all the machine learning models
    models = []
    models.append(('LR', LogisticRegression(random_state=9)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=9)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=9)))

    # variables to hold the results and names
    results = []
    names = []
    scoring = "accuracy"

    # import the feature vector and trained labels
    h5f_data = h5py.File('output/data.h5', 'r')
    h5f_label = h5py.File('output/labels.h5', 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    # verify the shape of the feature vector and labels
    print "[STATUS] features shape: {}".format(global_features.shape)
    print "[STATUS] labels shape: {}".format(global_labels.shape)

    print "[STATUS] training started..."


    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size,
                                                                                              random_state=seed)

    print "[STATUS] splitted train and test data..."
    print "Train data  : {}".format(trainDataGlobal.shape)
    print "Test data   : {}".format(testDataGlobal.shape)
    print "Train labels: {}".format(trainLabelsGlobal.shape)
    print "Test labels : {}".format(testLabelsGlobal.shape)

    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')

    # 10-fold cross validation
    for name, model in models:
        kfold = KFold(n=1522,n_folds=10, random_state=7)
        cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = pyplot.figure()
    fig.suptitle('Machine Learning algorithm comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()


if __name__ == '__main__':

                    #chave : valor
    dict_labels = { 
                    'onix' : '0001', 
                    'hb20' : '0002',
                    'sand' : '0003',
                    'vgol' : '0004',
                    'ford' : '0005'
                }
    
    loads_data(dict_labels)

    # #Splitting 
    # X_train, X_test, y_train, y_test = train_test_split(dataset,
    #                                                     dy_train,
    #                                                     test_size=0.2, 
    #                                                     random_state=33)

    # # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 
    # #                                                 test_size=0.5, 
    # #                                                 random_state=33)

    # display(X_train.shape)
    # display(X_test.shape)
    # display(y_train.shape)
    # display(y_test.shape)

    # print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(X_val), len(X_test)))

    # for idx in features:
    #     print idx, features[idx].shape, features[idx].max(), features[idx].min(), features[idx].mean()


    # random.seed(0)
    # np.random.seed(0)

    # # Separacao dos dados de treino
    # df_xTrain = df.iloc[:, 1:90].values
    # df_yTrain = df['salary'].values

    # X_train, X_test, y_train, y_test = train_test_split(df_xTrain, 
    #                                                     df_yTrain,
    #                                                     test_size=0.25,
    #                                                     stratify=df_yTrain,
    #                                                     shuffle='False',
    #                                                     random_state=None
    #                                                     )
    # #display(X_train.shape)
    # #display(X_test.shape)
    # #display(y_train.shape)
    # #display(y_test.shape)