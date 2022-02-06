from ctypes import byref

import numpy as np
import socket
import datetime
import threading
import tqdm
import time
import math
import pickle
from parameterSetup import ParameterSetup

params = ParameterSetup()
pickledDir = params.pickledDir
classifierType = params.classifierType
classifierParams = params.classifierParams
samplingFreq = params.samplingFreq
windowSizeInSec = params.windowSizeInSec
wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
eegFilePrefix = 'eegAndStage'

fileID = 'DBL-NO-D0846'
# fileID = 'DBL-NO-D0858'
# fileID = 'DBL-NO-D0859'
dataFileHandler = open(pickledDir + '/' + eegFilePrefix + '.' + fileID + '.pkl', 'rb')
(eeg, emg, stageSeq) = pickle.load(dataFileHandler)
y_test = np.array(stageSeq)

# fileName = '../data/waves/2017-10-05-16-23-02.061887.csv'
fileName = '../data/waves/wave_res.csv'
f = open(fileName)
labels = []
for line in f:
    elems = line.split(',')
    singleLabel = elems[1]
    if singleLabel in params.labelCorrectionDict.values():
        labels.append(params.reverseLabel(singleLabel))
y_pred = np.array(labels)

y_test = y_test[:y_pred.shape[0]]

for targetLabel in ['W', 'R', 'S']:
    TP = sum((y_test == targetLabel) & (y_pred == targetLabel))
    FP = sum((y_test != targetLabel) & (y_pred == targetLabel))
    FN = sum((y_test == targetLabel) & (y_pred != targetLabel))
    TN = sum((y_test != targetLabel) & (y_pred != targetLabel))
    sensitivity = 1.0 * TP / (TP + FN)
    accuracy = 1.0 * (TP + TN) / (TP + FP + FN + TN)
    specificity = 1.0 * TN / (TN + FP)
    print('for targetLabel = ' + targetLabel + ', sen = ' + str(sensitivity) + ', acc = ' + str(accuracy) + ', spec = ' + str(specificity))
