from __future__ import print_function
import sys
from functools import reduce
from os import listdir
from os.path import splitext, exists
import pickle
import numpy as np
from parameterSetup import ParameterSetup
from stageLabelAndOneHot import restrictStages
from fileManagement import readTrainFileIDsUsedForTraining

def extract_stageSeq(stageFilePath):
    stageFileHandler = open(stageFilePath, 'rb')
    (eeg, emg, stageSeq, timeStamps) = pickle.load(stageFileHandler)
    return np.array(stageSeq)

def computeTransitionTensor(params, paramID, classifierID):
    markovOrder = params.markovOrderForTraining
    featureDir = params.featureDir
    print('classifierID = ' + classifierID)
    if exists(featureDir + '/transitionTensor.' + classifierID + '.pkl'):
        print('transitionTensor.' + classifierID + '.pkl exists so skipping.')
        return

    train_eegAndStagePathL = [params.eegDir + '/' + train_eegAndStageFile for train_eegAndStageFile, _, _ in readTrainFileIDsUsedForTraining(params, classifierID)]
    y_train = restrictStages(reduce(lambda a, x: a + list(extract_stageSeq(x)), train_eegAndStagePathL, []), params.maximumStageNum)
    ### stageLabel2stageID = seq2stageDict(params, y_train)

    print('  y_train.shape = ' + str(y_train.shape))
    print('  y_train = ' + str(y_train))

    labelNum = params.maximumStageNum
    transitionTensorDims = tuple([labelNum for i in range(markovOrder+1)])
    transitionTensor = np.zeros(transitionTensorDims)

    labelsL = []
    for i in range(markovOrder+1):
        startIdx = markovOrder - i
        endIdx = - i
        if endIdx == 0:
            labelsL.append(y_train[startIdx:])
        else:
            labelsL.append(y_train[startIdx:endIdx])

    for sampleID in range(y_train.shape[0] - markovOrder):
        idxL = []
        for i in range(markovOrder+1):
            idx = params.stageLabel2stageID[labelsL[i][sampleID]]
            # print('idx = ' + str(idx))
            idxL.append(idx)
        # print('->  sampleID = ' + str(sampleID) + ', np.array(idxL) = ' + str(np.array(idxL)))
        # print(' ')
        transitionTensor[tuple(idxL)] += 1

    '''
    timeCnt = 0
    for label_present, label_past1, label_past2 in zip(y_train[2:], y_train[1:-1], y_train[:-2]):
        # print("[" + str(timeCnt) + "]" + str(params.stageLabel2stageID[label1]) + " -> " + str(params.stageLabel2stageID[label2]))
        transitionTensor[params.stageLabel2stageID[label_present], params.stageLabel2stageID[label_past1], params.stageLabel2stageID[label_past2]] += 1
        timeCnt += 1
    '''
    print('transitionTensor:')
    print('----')
    for mat in transitionTensor:
        for vec in mat:
            print(str(vec))
        print('----')
    # print("transition matrix:")
    # for l1 in range(labelNum):
        # for l2 in range(labelNum):
            # print(str(l1) + "=>" + str(l2) + " : " + str(transitionTensor[l1,l2]))
    # print(" ")

    transitionTensorFileName = featureDir + '/transitionTensor.' + classifierID + '.pkl'
    transitionTensorFileHandler = open(transitionTensorFileName, 'wb')
    pickle.dump(transitionTensor, transitionTensorFileHandler)

#---------------------
# main function

args = sys.argv
classifierIDs = args[1:]
params = ParameterSetup()
paramID = 0
for classifierID in classifierIDs:
    computeTransitionTensor(params, paramID, classifierID)
