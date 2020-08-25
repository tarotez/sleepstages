import sys
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from os.path import splitext
import pickle
import numpy as np
from parameterSetup import ParameterSetup
from evaluationCriteria import labels2ID, getEpisodeLengths
from sequentialPrediction import classifySequentially
from writePredictionResults import writePredictionResults
from fileManagement import readTrainFileIDsUsedForTraining, getFilesNotUsedInTrain
from fileManagement import getAllEEGFiles, fileIDsFromEEGFiles

params = ParameterSetup()
lstm_length = params.torch_lstm_length
paramID, markovOrderL = 0, [0]
print('lstm_length =', lstm_length)

fileIDs = fileIDsFromEEGFiles(getAllEEGFiles(params))
# fileIDs = ['D1798', 'D1803', 'D1811', 'D1818', 'D1831', 'D1799', 'D1804', 'D1814',
#           'D1819', 'D1833', 'D1800', 'D1805', 'D1815', 'D1820', 'D1801', 'D1806',
#           'D1816', 'D1826', 'D1802', 'D1810', 'D1817', 'D1827']
classifierIDs = ['Y4HOFA', '5XTKMY']
methodNames = ['UTSN', 'UTSN-L']
allFileIDandClassifierIDsL = []
y_predLL = []

for classifierID, methodName in zip(classifierIDs, methodNames):
    train_fileTripletL = readTrainFileIDsUsedForTraining(params, classifierID)
    train_fileIDs = [train_fileID for _, _, train_fileID in train_fileTripletL]
    # print('# train_fileIDs =', train_fileIDs)
    test_fileTripletL = getFilesNotUsedInTrain(params, train_fileIDs)
    all_fileTripletL = test_fileTripletL
    allFileIDandClassifierIDs = [(fileID, classifierID) for fileID in fileIDs]
    print('# allFileIDandClassifierIDs =', allFileIDandClassifierIDs)

    y_testL, y_predL = [], []
    for fileCnt, fileIDandClassifierID in enumerate(allFileIDandClassifierIDs):
        epochNumByStage_testL, epochNumByStage_predL, avg_ep_testL, avg_ep_predL = [], [], [], []
        print('fileIDandClassifierID = ' + str(fileIDandClassifierID))
        fileID = fileIDandClassifierID[0]
        predictionTargetDataFilePath = params.pickledDir + '/' + params.eegFilePrefix + '.' + fileID + '.pkl'
        print('predictionTargetDataFilePath =', predictionTargetDataFilePath)
        dataFileHandler = open(predictionTargetDataFilePath, 'rb')
        (eeg, ch2, stageSeq, timeStamps) = pickle.load(dataFileHandler)
        totalEpochNum = len(stageSeq[lstm_length-1:])
        params.markovOrderForPrediction = 0
        (y_test, y_pred) = classifySequentially(params, paramID, params.pickledDir, fileIDandClassifierID)
        y_testL.append(y_test)
        y_predL.append(y_pred)

    allFileIDandClassifierIDsL.append(allFileIDandClassifierIDs)
    y_predLL.append(y_predL)

with open('../data/pickled/y_test_and_y_pred_for_graphs.pkl','wb') as f:
    pickle.dump((allFileIDandClassifierIDsL, y_testL, y_predLL), f)
