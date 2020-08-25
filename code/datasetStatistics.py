import sys
from functools import reduce
from os.path import splitext
import pickle
import numpy as np
from parameterSetup import ParameterSetup
from evaluationCriteria import labels2ID, getEpisodeLengths
from sequentialPrediction import classifySequentially
from writePredictionResults import writePredictionResults
from fileManagement import readTrainFileIDsUsedForTraining, getFilesNotUsedInTrain

def avg_ep(ep_lengthsL):
    return [np.array(ep_lengths).mean() for ep_lengths in ep_lengthsL]

def getEpochNumByStage(ep_lengthsL):
    return [reduce(lambda a, x: a + x, ep_lengths) if len(ep_lengths) > 0 else 0 for ep_lengths in ep_lengthsL]

def getRatio(epochNumsL):
    sum = np.array(epochNumsL).sum()
    return [round(epochNum/sum, 4) for epochNum in epochNumsL]

args = sys.argv
classifierIDs = args[1:]
params = ParameterSetup()
pickledDir = params.pickledDir
paramDir = params.pickledDir
eegFilePrefix = params.eegFilePrefix
lstm_length = params.torch_lstm_length
paramID = 0
markovOrderL = [0]

print('lstm_length =', lstm_length)

for classifierID in classifierIDs:

    train_fileTripletL = readTrainFileIDsUsedForTraining(params, classifierID)
    train_fileIDs = [train_fileID for _, _, train_fileID in train_fileTripletL]
    # print('# train_fileIDs =', train_fileIDs)
    test_fileTripletL = getFilesNotUsedInTrain(params, train_fileIDs)

    ### all_fileTripletL = train_fileTripletL + test_fileTripletL
    all_fileTripletL = test_fileTripletL
    ### all_fileTripletL = [test_fileTripletL[0]]

    allFileIDandClassifierIDs = [(fileID, classifierID) for _, _, fileID in all_fileTripletL]
    print('# allFileIDandClassifierIDs =', allFileIDandClassifierIDs)

    epochNums = []
    epochNumByStage_testL = []
    epochNumByStage_predL = []
    avg_ep_testL = []
    avg_ep_predL = []
    for fileIDandClassifierID in allFileIDandClassifierIDs:
        print('fileIDandClassifierID = ' + str(fileIDandClassifierID))
        fileID = fileIDandClassifierID[0]
        predictionTargetDataFilePath = pickledDir + '/' + eegFilePrefix + '.' + fileID + '.pkl'
        print('predictionTargetDataFilePath =', predictionTargetDataFilePath)
        dataFileHandler = open(predictionTargetDataFilePath, 'rb')
        (eeg, ch2, stageSeq, timeStamps) = pickle.load(dataFileHandler)
        epochNums.append(len(stageSeq[lstm_length-1:]))

        params.markovOrderForPrediction = 0
        (y_test, y_pred) = classifySequentially(params, paramID, paramDir, fileIDandClassifierID)

        stageIDseq_test = labels2ID(params, y_test[lstm_length-1:])
        stageIDseq_pred = labels2ID(params, y_pred[lstm_length-1:])
        print('len(stageIDseq_test) =', len(stageIDseq_test))
        print('len(stageIDseq_pred) =', len(stageIDseq_pred))

        episodeLengths_test = getEpisodeLengths(params, y_test)
        episodeLengths_pred = getEpisodeLengths(params, y_pred)

        epochNumByStage_test = getEpochNumByStage(episodeLengths_test)
        epochNumByStage_pred = getEpochNumByStage(episodeLengths_pred)
        epochNumByStage_testL.append(epochNumByStage_test)
        epochNumByStage_predL.append(epochNumByStage_pred)
        print('epochNumByStage_test =', epochNumByStage_test)
        print('epochNumByStage_pred =', epochNumByStage_pred)
        print('total epochNum for test =', np.array(epochNumByStage_test).sum())
        print('total epochNum for pred =', np.array(epochNumByStage_pred).sum())
        print('ratioOfStages_test = ', getRatio(epochNumByStage_test))
        print('ratioOfStages_pred = ', getRatio(epochNumByStage_pred))

        avg_ep_test = avg_ep(episodeLengths_test)
        avg_ep_pred = avg_ep(episodeLengths_pred)
        avg_ep_testL.append(avg_ep_test)
        avg_ep_predL.append(avg_ep_pred)
        print('avgEpochLengths_test =', avg_ep_test)
        print('avgEpochLengths_pred =', avg_ep_pred)

    print('epochNums =', epochNums)
    print('params.correctedLabel2stageID =', params.correctedLabel2depthID)

    totalEpochNum = reduce(lambda a, x: a + x, epochNums)
    print('totalEpochNum = ', totalEpochNum)
    avgEpochNum = totalEpochNum / len(epochNums)
    print('avgEpochNum =', round(avgEpochNum))



    #-----
    # sort episodes
    print('episodeLengths_test =', episodeLengths_test)
    print('episodeLengths_pred =', episodeLengths_pred)
    print('')

    avg_ep_testA = np.array(avg_ep_testL)
    avg_ep_predA = np.array(avg_ep_predL)
    avg_ep_across_mice_test = np.mean(avg_ep_testA, axis=0)
    avg_ep_across_mice_pred = np.mean(avg_ep_predA, axis=0)

    print('avg_ep_across_mice_test =', avg_ep_across_mice_test)
    print('avg_ep_across_mice_pred =', avg_ep_across_mice_pred)
    print('')


####
