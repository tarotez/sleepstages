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
from fileManagement import readTrainFileIDsUsedForTraining, getFileIDsFromRemainingBlocks

# crossValidationID = 'ARS0Q'
# crossValidationID = '6S3BL'
args = sys.argv
crossValidationID = args[1]

params = ParameterSetup()
lstm_length = params.torch_lstm_length
paramID, markovOrder = 0, 0

# cross-validate, i.e. test using files not used for training
outputDir = '../data/pickled'

with open(outputDir + '/crossvalidation_metadata.' + crossValidationID + '.pkl', 'rb') as f:
    splitID, classifierIDsByMethod = pickle.load(f)
    print('splitID =', splitID)

fileIDsByBlocks = []
with open(outputDir + '/blocks_of_records.' + splitID + '.csv') as f:
    for line in f:
        fileIDsByBlocks.append(line.rstrip().split(','))
        # print(fieIDs)

testFileIDandClassifierIDs_byMethod, y_test_byMethod, y_pred_byMethod = [], [], []
# loopCnt = 0
for methodID, classifierIDsByBlockID in enumerate(classifierIDsByMethod):
    testFileIDandClassifierIDs_byBlock, y_test_byBlock, y_pred_byBlock = [], [], []
    for blockID, (classifierID, test_fileIDs) in enumerate(zip(classifierIDsByBlockID, fileIDsByBlocks)):
        print('')
        print('blockID =', blockID)
        print('classifierID =', classifierID)
        train_fileIDs = getFileIDsFromRemainingBlocks(fileIDsByBlocks, blockID)
        print('%%% len(train_fileIDs) =', len(train_fileIDs))
        print('%%% len(test_fileIDs) =', len(test_fileIDs))
        # pairing to predict and evaluate a file (test_fileID) using a classifier (classifierID)
        testFileIDandClassifierIDs = [(test_fileID, classifierID) for test_fileID in test_fileIDs]
        # print('# testFileIDandClassifierIDs =', testFileIDandClassifierIDs)
        y_test_byFile, y_pred_byFile = [], []
        for testFileCnt, testFileIDandClassifierID in enumerate(testFileIDandClassifierIDs):
            epochNumByStage_testL, epochNumByStage_predL, avg_ep_testL, avg_ep_predL = [], [], [], []
            print('testFileIDandClassifierID = ' + str(testFileIDandClassifierID))
            testFileID = testFileIDandClassifierID[0]
            # print('testFileIDandClassifierID[0] =', testFileIDandClassifierID[0])
            predictionTargetDataFilePath = params.pickledDir + '/' + params.eegFilePrefix + '.' + testFileID + '.pkl'
            print('predictionTargetDataFilePath =', predictionTargetDataFilePath)
            dataFileHandler = open(predictionTargetDataFilePath, 'rb')
            (eeg, ch2, stageSeq, timeStamps) = pickle.load(dataFileHandler)
            totalEpochNum = len(stageSeq[lstm_length-1:])
            params_for_classifier = ParameterSetup(paramFileName='params.'+classifierID+'.json')
            params_for_classifier.markovOrderForPrediction = markovOrder
            (y_test, y_pred) = classifySequentially(params_for_classifier, paramID, params.pickledDir, testFileIDandClassifierID)
            y_test_byFile.append(y_test)
            y_pred_byFile.append(y_pred)
            # loopCnt += 1
            # if loopCnt > 2:
            #    break

        testFileIDandClassifierIDs_byBlock.append(testFileIDandClassifierIDs)
        y_test_byBlock.append(y_test_byFile)
        y_pred_byBlock.append(y_pred_byFile)
        # if loopCnt > 2:
        #    break

    testFileIDandClassifierIDs_byMethod.append(testFileIDandClassifierIDs_byBlock)
    y_test_byMethod.append(y_test_byBlock)
    y_pred_byMethod.append(y_pred_byBlock)
    with open('../data/pickled/y_test_and_y_pred_for_graphs.' + crossValidationID + '.pkl','wb') as f:
        pickle.dump((testFileIDandClassifierIDs_byMethod, y_test_byMethod, y_pred_byMethod), f)
