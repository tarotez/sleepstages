from __future__ import print_function
from freqAnalysisTools import band
import sys
import pickle
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import random
from sklearn import linear_model, svm, ensemble, neural_network
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#---------------
# set up parameters

stageLabels=["W", "R", "S", "2", "3", "4", "M"]
labelNum = len(stageLabels)

if len(sys.argv)>1:
    classifierType = sys.argv[1]
else:
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
    classifierParams = [500]
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
        classifier = ensemble.RandomForestClassifier(n_estimators=classifierParams[paramID])
    elif classifierType == 'nn':
        classifier = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    for testSetID in range(fileNum):
        print('testSetID = ' + str(testSetID))

        featureDim = featureList[0].shape[0]
        x_train_trans = np.empty((featureDim, 0), float)
        y_train_L = []
        for trainSetID in range(fileNum):
            # print('  trainSetID = ' + str(trainSetID))
            if trainSetID != testSetID:
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
                # print('trainSetID = ' + str(trainSetID) + ', fileName = ' + fileIDs[trainSetID])
                # print('  feature4train.shape = ' + str(feature4train.shape))
                # print('  len(stageSeq_L[tS]) = ' + str(len(stageSeq_L[trainSetID])))

        # print('x_train_trans.shape = ' + str(x_train_trans.shape))
        x_train = np.transpose(x_train_trans)
        y_train = np.array(y_train_L)
        print('  x_train.shape = ' + str(x_train.shape))
        print('  y_train.shape = ' + str(y_train.shape))

        feature4test = featureList[testSetID]
        fLen4test = feature4test.shape[1]
        sLen4test = len(stageSeq_L[testSetID])
        if fLen4test != sLen4test:
            feature4test = feature4test[:,:sLen4test]
        x_test = np.transpose(feature4test)
        y_test = np.array(stageSeq_L[testSetID])
        y_test_list.append(y_test)

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        y_pred_list.append(y_pred)

        comparisonVec = (y_test == y_pred)
        precision = np.sum(comparisonVec) / comparisonVec.shape[0]
        precisionVec[testSetID] = precision
        # print('testSetID = ' + str(testSetID) + ' (' + fileIDs[testSetID] + '), precision = ' + str(precision))
        print('for ' + fileIDs[testSetID] + ':')
        # print(' ')
        cnf_mat = confusion_matrix(y_test, y_pred, labels=stageLabels)
        cnf_tensor[:,:,paramID,testSetID] = cnf_mat

        #--------------
        # compute sensitivity and accuracy

        # sensitivity = TP/P = TP/(TP + FN)
        # specificity = TN/N = TN/(TN + FP)
        # accuracy = (TP + TN)/(TP + FP + FN + TN)
        for labelID in range(3):
            targetLabel = stageLabels[labelID]
            TP = sum((y_test == targetLabel) & (y_pred == targetLabel))
            FP = sum((y_test != targetLabel) & (y_pred == targetLabel))
            FN = sum((y_test == targetLabel) & (y_pred != targetLabel))
            TN = sum((y_test != targetLabel) & (y_pred != targetLabel))
            sensitivity[paramID,testSetID,labelID] = TP / (TP + FN)
            specificity[paramID,testSetID,labelID] = TN / (TN + FP)
            accuracy[paramID,testSetID,labelID] = (TP + TN) / (TP + FP + FN + TN)
            # print('   for ' + stageLabels[labelID] + ', sensitivity = ' + str(sensitivity[paramID,testSetID,labelID]) + ', specificity = ' + str(specificity[paramID,testSetID,labelID]) + ', accuracy = ' + str(accuracy[paramID,testSetID,labelID]))
            # print('   for ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivity[paramID,testSetID,labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[paramID,testSetID,labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[paramID,testSetID,labelID]))
            print('   stage = ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivity[paramID,testSetID,labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[paramID,testSetID,labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[paramID,testSetID,labelID]))
        print(' ')

        meanPrec = np.mean(precisionVec)
        meanPrecVec[paramID] = meanPrec
        # print('   mean(precision) = ' + str(meanPrec))

        file = open(path + 'cnf_tensor' + '.pkl', 'wb')
        result = (stageLabels, cnf_tensor, sensitivity, specificity, accuracy, preContextSize, postContextSize)
        pickle.dump(result, file)

        file = open(path + 'pred_results.pkl', 'wb')
        result = (stageLabels, classifierParams, fileIDs, y_test_list, y_pred_list, preContextSize, postContextSize)
        pickle.dump(result, file)

        file = open(path + 'classifier.' + classifierType + '.param.' + str(classifierParams[paramID]) + '.testSetID.' + str(testSetID) + '.pkl', 'wb')
        pickle.dump(classifier, file)



