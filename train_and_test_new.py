# -*- coding: utf-8 -*-
# base: https://gogul09.github.io/software/image-classification-python
# organize imports
# import mahotas
import cv2
import os
import random
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

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
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

# fixed-sizes for image
fixed_size = tuple((224, 224))

# path to training data
train_path = "data/train"

# no.of.trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.20

# seed for reproducing same results
seed = 9

# get the training labels
# train_labels = os.listdir(train_path)
train_labels = ['1','2','3','4','5']


# sort the training labels
train_labels.sort()
# print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# num of images per class
images_per_class = 80


def apresentaInfos(dataset, y_pred):
    tn, fp, fn, tp = confusion_matrix(dataset,y_pred).ravel()
    print("Falso Negativo (Errou):",fn)
    print("Falso Positivo (Errou):",fp)
    print("Verdadeiro Negativo (Acertou):",tn)
    print("Verdadeiro Positivo (Acertou):",tp) 

def apresentarMetricas(dataset, y_pred):
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    print(" F1_Score:",metrics.f1_score(y_test, y_pred))
    print("\n")

def find_paths(path):
    paths = []
    level_a = os.listdir(path)
    for level_name in level_a:
        for image_name in os.listdir(os.path.join(path, level_name)):
            paths += [os.path.join(path, level_name, image_name)]

    return paths

def carrega_dados_01(dict_labels):
    features = np.load("cars.npz")

    for file in features:
        # print 'file: ', file
        global_features.append(features[file][0])
        label_in_file = dict_labels[file[:4]]
        labels.append(int(label_in_file))


    random.seed(0)
    np.random.seed(0)

    X_train, X_test, y_train, y_test = train_test_split(global_features, 
                                                    labels,
                                                    test_size=test_size,
                                                    stratify=labels,
                                                    random_state=None
                                                    )

    # get the overall feature vector size
    print "[STATUS] global_features shape {}".format(np.array(global_features).shape)
    # print 'global_features', global_features

    # get the overall training label size
    print "[STATUS] labels shape  {}".format(np.array(labels).shape)

    #Treinando uma árvore de decisão com profundidade máxima = 3
    dt = tree.DecisionTreeClassifier(random_state=9)

    dt = dt.fit(X_train,y_train)
    print(dt)

    #Teste no conjunto de testes
    y_pred = dt.predict(X_test)

    #Calcule a acurácia da árvore
    print("Acurácia da Árvore: " + str(accuracy_score(y_test, y_pred)*100)+"%")


def carrega_dados_02(dict_labels):
    features = np.load("cars_train.npz")

    # create all the machine learning models
    models = []
    models.append(('1 - Logistic Regression (LR)', LogisticRegression(random_state=9)))
    models.append(('2 - Linear Discriminant Analysis (LDA)', LinearDiscriminantAnalysis()))
    models.append(('3 - K-Nearest Neighbors (KNN)', KNeighborsClassifier()))
    models.append(('4 - Decision Trees (CART)', DecisionTreeClassifier(random_state=9)))
    models.append(('5 - Random Forests (RF)', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
    models.append(('6 - Gaussian Naive Bayes (NB)', GaussianNB()))
    models.append(('7 - Support Vector Machine (SVM)', SVC(random_state=9)))

    # variables to hold the results and names
    results = []
    names = []
    scoring = "accuracy"

    base_path = 'data/train'
    paths = find_paths(base_path)
    #print paths

    for image_training in paths:
        # print image_training
        fname = image_training[16:]
        for file in features:
            # print 'fname: ', fname
            # print 'file: ', file
            if fname == file:
                global_features.append(features[file][0])
                label_in_file = dict_labels[file[:4]]
                labels.append(int(label_in_file))


    # verify the shape of the feature vector and labels
    # print "[STATUS] features shape: {}".format(np.array(global_features).shape)
    # print "[STATUS] labels shape: {}".format(np.array(labels).shape)

    # split the training and testing data
    (trainDataGlobal, testData, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                      np.array(labels),
                                                                      test_size=test_size,
                                                                      random_state=seed)

    # print "[STATUS] splitted train and test data..."
    # print "Train data  : {}".format(np.array(trainDataGlobal).shape)
    # print "Test data   : {}".format(np.array(testData).shape)
    # print "Train labels: {}".format(np.array(trainLabelsGlobal).shape)
    # print "Test labels : {}".format(np.array(testLabelsGlobal).shape)

    # print("Processando as acurácias dos 7 modelos...")
    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')

    # 10-fold cross validation
    for name, model in models:
        kfold = KFold(n=1214,n_folds=10, random_state=7)
        cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    testando_melhor_classificador(trainDataGlobal, trainLabelsGlobal)



def testando_melhor_classificador(trainDataGlobal, trainLabelsGlobal):
    features = np.load("cars_test.npz")
                    #chave : valor
    labels_dict = { 
              1 : 'Onix', 
              2 : 'HB20',
              3 : 'Sandero',
              4 : 'Gol',
              5 : 'Ford Ka'
             }
             
    print("Como a melhor acurácia foi utilizando o modelo de Regressão Logística,")
    print("foi com esse modelo que foi utilizado para testar com as imagens de teste....")
    # Logistic Regression (LR)
    LR = LogisticRegression(random_state=9)
    # Linear Discriminant Analysis (LDA)
    LDA = LinearDiscriminantAnalysis()
    # # K-Nearest Neighbors (KNN)
    KNN = KNeighborsClassifier()
    # # Decision Trees (CART)
    CART = DecisionTreeClassifier(random_state=9)
    # # Random Forests (RF)
    RF = RandomForestClassifier(n_estimators=num_trees, random_state=9)
    # # 'Gaussian Naive Bayes (NB)
    NB = GaussianNB()
    # # Support Vector Machine (SVM)
    SVM = SVC(random_state=9)

    # Trocar aqui pela abreviatura do modelo que deseja utilizar
    clf = LR 
    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)


    # path to test data
    test_path = "data/test"

    paths = find_paths(test_path)

    # print 'paths[1]: ', paths[288]
    num_samples = 35 # são 381 imagens para teste
    # test_indexes = random.sample(range(num_samples), 20)
    # print 'test_indexes', test_indexes
    for file in paths:
    # for i in test_indexes:
        # print 'i: ', i
        # file = paths[i]
        # print 'file: ', file
        # read the image
        image = cv2.imread(file)

        # # resize the image
        # image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fname = file[15:]
        # print 'train_labels: ', train_labels
        # print 'fname', fname
        global_feature = features[fname][0]

        # # predict label of test image
        prediction = clf.predict(global_feature.reshape(1,-1))[0]

        cv2.putText(image, labels_dict[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        # display the output image
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()
    y_test = clf.predict()

    print("Acurácia: " + str(accuracy_score(y_test, y_pred)*100)+"%")

if __name__ == '__main__':

                    #chave : valor
    dict_labels = { 
                    'onix' : '0001', 
                    'hb20' : '0002',
                    'sand' : '0003',
                    'vgol' : '0004',
                    'ford' : '0005'
                }

    
    carrega_dados_02(dict_labels)

    