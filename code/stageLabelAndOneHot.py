from __future__ import print_function
from os.path import splitext
import numpy as np

def seq2labelFreqs(stageSeq_arr):
    labels = np.unique(stageSeq_arr)
    labelNum = labels.shape[0]
    labelFreqs = np.zeros((labelNum))
    for label, labelID in zip(labels, range(labelNum)):
        matches = np.argwhere(stageSeq_arr==label)
        # print('label ' + label + ' appears ' + str(matches.size) + ' times.')
        if matches.size > 0:
            labelFreqs[labelID] = matches.shape[0]
    return labelFreqs

'''
def seq2stageDict(params, stageSeq):
    stageSeq_arr = np.array(stageSeq)
    labels = np.unique(stageSeq_arr)
    labelNum = labels.shape[0]
    labelFreqs = seq2labelFreqs(stageSeq_arr)
    print('labelFreqs = ' + str(labelFreqs))
    sortedLabelIDs = np.argsort(labelFreqs)
    print('sortedLabelIDs = ' + str(sortedLabelIDs))
    print('params.maximumStageNum = ' + str(params.maximumStageNum))
    labelIDsToBeUsed = sortedLabelIDs[(labelNum - params.maximumStageNum):]
    print('labels to be used: ' + str(labels[labelIDsToBeUsed]))
    stageDict = {labels[labelID] : labelID for labelID in labelIDsToBeUsed}
    return stageDict
'''

def restrictStages(params, orig_stageSeq, maximumStageNum):
    orig_stageSeq_arr = np.array(orig_stageSeq)
    labels = np.unique(orig_stageSeq_arr)
    labelFreqs = seq2labelFreqs(orig_stageSeq_arr)
    mostFrequentLabel = labels[np.argmax(labelFreqs)]
    # print('mostFrequentLabel = ' + str(mostFrequentLabel))
    for label in labels:
        if not label in params.stageLabels4evaluation:
            toBeReplacedSampleIDs = np.argwhere(orig_stageSeq_arr==label).reshape(-1)
            # print('toBeReplacedSampleIDs.shape = ' + str(toBeReplacedSampleIDs.shape))
            orig_stageSeq_arr[toBeReplacedSampleIDs] = mostFrequentLabel
    return orig_stageSeq_arr

def oneHot2stageLabel(oneHot, stageLabels4evaluation, stageLabel2stageID):
    # print('in oneHot2stageLabel, oneHot.shape = ' + str(oneHot.shape))
    # keyList = [keys for keys in params.stageLabel2stageID.keys()]
    # print('keyList = ' + str(keyList))
    stageID = np.argmax(oneHot)
    for key in stageLabels4evaluation:
        if stageLabel2stageID[key] == stageID:
            return key
    return '-'

def stageLabel2oneHot(stageLabel, maximumStageNum, stageLabel2stageID):
    oneHot = np.zeros((maximumStageNum), dtype=np.int)
    # print('stageLabel = ' + stageLabel)
    if stageLabel == 'None':
        print('stageLabel = ' + stageLabel)
    # stageLabel = stageLabel.replace('*','')
    # stageLabel = stageLabel.replace('2','S')
    # stageLabel = stageLabel.replace('None','P')
    stageID = stageLabel2stageID[stageLabel]
    # oneHot[stageID,0] = 1
    oneHot[stageID] = 1
    return oneHot

def constructPastStagesOneHots(stageSeq, wID, pastStageLookUpNum, stageNum):
    pastStagesOneHots = np.zeros((pastStageLookUpNum * stageNum, 1), dtype=np.float)
    for offset in range(1, pastStageLookUpNum + 1):
        oneHot = stageLabel2oneHot(stageSeq[wID - offset])
        # print('   offset = ' + str(offset))
        # print('   oneHot.shape = ' + str(oneHot.shape))
        # print('   pastStagesOneHots.shape = ' + str(pastStagesOneHots.shape))
        # print('   index = ' + str((offset - 1) * stageNum) + ':' + str(offset * stageNum))
        # pastStagesOneHots[((offset - 1) * stageNum):(offset * stageNum),0] = oneHot[:,0]
        pastStagesOneHots[((offset - 1) * stageNum):(offset * stageNum),0] = oneHot
    return pastStagesOneHots
