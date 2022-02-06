from __future__ import print_function
from freqAnalysisTools import band
from os import listdir
from os.path import isfile, join, splitext
import sys
import pickle
import math
import numpy as np
from sklearn import linear_model, svm, ensemble, neural_network
from parameterSetup import ParameterSetup
from evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat

params = ParameterSetup()

classifierType = params.classifierType
classifierParams = params.classifierParams

paramNum = len(classifierParams)

# parameters for signal processing
windowSizeInSec = params.windowSizeInSec   # size of window in time for estimating the state
samplingFreq = params.samplingFreq   # sampling frequency of data

# parameters for using history
preContextSize = params.preContextSize

# parameters for making a histogram
wholeBand = params.wholeBand
binWidth4freqHisto = params.binWidth4freqHisto    # bin width in the frequency domain for visualizing spectrum as a histogram

# for reading data
classifierDir = params.classifierDir
# classifierName = params.classifierName
# samplePointNum = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
# time_step = 1 / samplingFreq
# binNum4spectrum = round(wholeBand.getBandWidth() / binWidth4freqHisto)
# print('samplePointNum = ' + str(samplePointNum))
# past_eeg = np.empty((samplePointNum, 0), dtype = np.float)
# past_feature = np.empty((binNum4spectrum, 0), dtype = np.float)
# print('in __init__, past_eeg.shape = ' + str(past_eeg.shape))

print('classifier type = ' + str(classifierType))

for paramID in range(paramNum):

    print('classifier parameter = ' + str(classifierParams[paramID]))

    files =  listdir(classifierDir)
    for excludedFileFullName in files:

        if excludedFileFullName.startswith('train_data.'):
            excludedFileName, excludedFile_extension = splitext(excludedFileFullName)
            elems = excludedFileName.split('.')
            excludedFileID = elems[3]
            print('excludedFileID = ' + excludedFileID)
            if params.useEMG:
                label4EMG = params.label4withEMG
            else:
                label4EMG = params.label4withoutEMG
            train_data_file = open(classifierDir + '/train_data.' + label4EMG + '.excludedFileID.' + excludedFileID + '.pkl', 'rb')
            train_data = pickle.load(train_data_file)
            (x_train, y_train) = train_data

            file = open(classifierDir + '/' + params.classifierPrefix + '.' + label4EMG + '.' + classifierType + '.param.' + str(classifierParams[paramID]) + '.excludedFileID.' + excludedFileID + '.pkl', 'rb')
            classifier = pickle.load(file)
            y_pred = classifier.predict(x_train)
            (stageLabels, sensitivity, specificity, accuracy) = y2sensitivity(y_train, y_pred)
            (stageLabels4confusionMat, confusionMat) = y2confusionMat(y_train, y_pred)
            printConfusionMat(stageLabels4confusionMat, confusionMat)

            y_matching = (y_train == y_pred)
            correctNum = sum(y_matching)
            # print('y_train = ' + str(y_train[:50]))
            # print('y_pred = ' + str(y_pred[:50]))
            # print('correctNum = ' + str(correctNum))
            y_length = y_pred.shape[0]
            precision = correctNum / y_length
            for labelID in range(len(stageLabels)):
                print('  stageLabel = ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivity[labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[labelID]))
            print('  precision = ' + "{0:.5f}".format(precision) + ' (= ' + str(correctNum) + '/' + str(y_length) +')')
            print('')
