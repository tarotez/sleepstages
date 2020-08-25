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

stageLabels=["W", "R", "S", "2", "3", "4", "M"]

# for file reading
path = '../data/pickled/'

fileNum = len(fileIDs)

sensitivity = np.zeros((paramNum,fileNum,labelNum))
specificity = np.zeros((paramNum,fileNum,labelNum))
accuracy = np.zeros((paramNum,fileNum,labelNum))

for paramID in range(paramNum):

    print('classifier parameter = ' + str(classifierParams[paramID]))

    for testSetID in range(fileNum):

        for labelID in range(3):
            targetLabel = stageLabels[labelID]
            truePositiveCell = 
            targetRow = 
            targetColumn = 
            trueNegativeBlock = 
            TP = truePositiveCell
            FP = 
            FN = 
            TN = sum(sum(trueNegativeBlock))
            sensitivity[paramID,testSetID,labelID] = TP / (TP + FN)
            specificity[paramID,testSetID,labelID] = TN / (TN + FP)
            accuracy[paramID,testSetID,labelID] = (TP + TN) / (TP + FP + FN + TN)
            print('   for ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivity[paramID,testSetID,labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[paramID,testSetID,labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[paramID,testSetID,labelID]))

