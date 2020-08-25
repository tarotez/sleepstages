from __future__ import print_function
from freqAnalysisTools import band
import sys
import pickle
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import random
# from sklearn import linear_model, svm, ensemble, neural_network
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#---------------
# set up parameters

stageLabels=["W", "R", "S", "2", "3", "4", "M"]
labelNum = len(stageLabels)

#if len(sys.argv)>1:
#    classifierType = sys.argv[1]
# else:
#    classifierType = 'logreg'
classifierType = 'logreg'

print('classifier is ' + classifierType)

# for file reading
path = '../data/pickled/'
### path = '../data/test_pickled/'

if classifierType == 'logreg':
    print('using logistic regression as a classifier.')
    # classifierParams = [1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 1, 1e4, 1e8, 1e12, 1e16, 1e20]
    classifierParams = [1]
elif classifierType == 'svm':
    print('using SVM as a classifier.')
    classifierParams = [1e-3, 1, 1e3]
elif classifierType == 'rf' or classifierType == 'randomforest':
    classifierType = 'rf'
    print('using random forest as a classifier.')
    classifierParams = [1]
elif classifierType == 'nn':
    print('using a neural network as a classifier.')
    classifierParams = [1]
else:
    print('Sorry, this classifier not supported.')
    sys.exit()

paramNum = len(classifierParams)

#----------------
# read data from pickle

fileName = open(path + 'features' + '.pkl', 'rb')
features =  pickle.load(fileName)
(featureHistWithoutContext_L, featureHistWithContext_L, featureThetaDelta_L, stageSeqWithoutContext_L, stageSeqShortenedToIncludeContext_L, fileIDs, preContextSize, postContextSize) = features

#-----------------
# select a feature

# featureList = featureThetaDelta_L
### featureList = featureHistWithoutContext_L
featureList = featureHistWithContext_L
### stageSeq_L = stageSeqWithoutContext_L
stageSeq_L = stageSeqShortenedToIncludeContext_L
# print('fileIDS = ')
# print(fileIDs)
fileNum = len(fileIDs)

#----------------
# classify using logistic regression

meanPrecVec = np.zeros(paramNum)
cnf_tensor = np.zeros((labelNum,labelNum,paramNum,fileNum))
sensitivity = np.zeros((paramNum,fileNum,labelNum))
specificity = np.zeros((paramNum,fileNum,labelNum))
accuracy = np.zeros((paramNum,fileNum,labelNum))
y_test_list = []
y_pred_list = []

for paramID in range(paramNum):

    precisionVec = np.zeros(fileNum)

    print('classifier parameter = ' + str(classifierParams[paramID]))

    if classifierType == 'logreg':
        classifier = linear_model.LogisticRegression(C=classifierParams[paramID])
    elif classifierType == 'svm':
        # SVM was slower and has lower precision than logistic regression (with rbf and default parameter)
        classifier = svm.SVC(kernel='rbf', C=classifierParams[paramID])
    elif classifierType == 'rf':
        classifier = ensemble.RandomForestClassifier()
    elif classifierType == 'nn':
        classifier = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    featureDim = featureList[0].shape[0]
    x_train_trans = np.empty((featureDim, 0), float)
    y_train_L = []
    for trainSetID in range(fileNum):
        # print('  trainSetID = ' + str(trainSetID))
        feature4train = featureList[trainSetID]
        # print('feature4train.shape = ' + str(feature4train.shape))
        # print('feature4train = ')
        # print(feature4train)

        fLen = featureList[trainSetID].shape[1]
        sLen = len(stageSeq_L[trainSetID])
        # Below deals with the case not all of the time windows were labeled manually.
        # In such a case, stageSeq is shorter than featureList
        if fLen != sLen:
            feature4train = feature4train[:,:sLen]
            fLenShort = feature4train.shape[1]
            # print('for testSetID = ' + str(testSetID) + ', trainSetID = ' + str(trainSetID) + ' is used.')
            # print(' original length of feature = ' + str(fLen))
            # print(' revised length of feature = ' + str(fLenShort))
            # print(' len(stageSeq_L[trainSetID]) = ' + str(sLen))
            # print('')

        x_train_trans = np.append(x_train_trans, feature4train, axis=1)
        # print('  feature4train.shape = ' + str(feature4train.shape) + ', x_train_trans.shape = ' + str(x_train_trans.shape))
        y_train_L = y_train_L + stageSeq_L[trainSetID]
        print('trainSetID = ' + str(trainSetID) + ', fileName = ' + fileIDs[trainSetID])
        print('  feature4train.shape = ' + str(feature4train.shape))
        print('  len(stageSeq_L[tS]) = ' + str(len(stageSeq_L[trainSetID])))

    # print('x_train_trans.shape = ' + str(x_train_trans.shape))
    x_train = np.transpose(x_train_trans)
    y_train = np.array(y_train_L)
    print('  x_train.shape = ' + str(x_train.shape))
    print('  y_train.shape = ' + str(y_train.shape))

    classifier.fit(x_train, y_train)

    file = open(path + 'classifier.paramID.' + str(paramID) + '.pkl', 'wb')
    pickle.dump(classifier, file)


