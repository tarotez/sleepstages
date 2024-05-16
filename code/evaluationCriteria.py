from __future__ import print_function
import numpy as np
from parameterSetup import ParameterSetup

def labels2ID(params, y_seq):
    return [params.correctedLabel2depthID[params.labelCorrectionDict[y]] for y in y_seq]

def getEpisodeLengths(params, y_seq):
    episodeLengths = [[] for _ in range(len(params.correctedLabel2depthID))]
    stageID_seq = labels2ID(params, y_seq)
    # print('stageID_seq = ', end='')
    # [print(i, end='') for i in stageID_seq]
    # print('')
    episodeStageID = stageID_seq[0]
    episodeLength = 1
    for newStageID in stageID_seq[1:]:
        if newStageID != episodeStageID:
            episodeLengths[episodeStageID].append(episodeLength)
            episodeStageID = newStageID
            episodeLength = 1
        else:
            episodeLength += 1
    episodeLengths[episodeStageID].append(episodeLength)   # for the last segment
    return episodeLengths

def mathewsCorrelationCoefficient(stageLabels4confusionMat, confusionMat, targetStage):
    labelNum = len(stageLabels4confusionMat)
    targetID = stageLabels4confusionMat.index(targetStage)
    others = [i for i in range(labelNum)]
    del others[targetID]
    tp = np.sum(confusionMat[targetID,targetID])
    tn = np.sum(confusionMat[others,others])
    fp = np.sum(confusionMat[others,targetID])
    fn = np.sum(confusionMat[targetID,others])
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
    return mcc

def multiClassMCC(c):  # c is the confusion mat
    r = range(c.shape[0])
    numer = np.array([[[ ((c[k,k] * c[l,m]) - (c[k,l] * c[m,k])) for k in r] for l in r] for m in r]).sum()
    denom1 = sum([ sum(c[k,:]) * sum([sum(c[h,:]) if h != k else 0 for h in r]) for k in r])
    denom2 = sum([ sum(c[:,k]) * sum([sum(c[:,h]) if h != k else 0 for h in r]) for k in r])
    return numer / np.sqrt(1.0 * denom1 * denom2)

def y2confusionMat(y_test, y_pred, stageLabels):
    labelNum = len(stageLabels)
    confusionMat = np.zeros((labelNum,labelNum), dtype=int)
    for labelID4test in range(labelNum):
        label4test = stageLabels[labelID4test]
        for labelID4pred in range(labelNum):
            label4pred = stageLabels[labelID4pred]
            match_onehot = (y_test == label4test) & (y_pred == label4pred)
            confusionMat[labelID4test, labelID4pred] = np.sum(match_onehot)
    return (stageLabels, confusionMat)

def printConfusionMat(stageLabels, confusionMat):
    labelNum = len(stageLabels)
    for labelID4test in range(labelNum):
        label4test = stageLabels[labelID4test]
        for labelID4pred in range(labelNum):
            label4pred = stageLabels[labelID4pred]
            print('  ' + label4test + '->' + label4pred + ': ' + str(confusionMat[labelID4test, labelID4pred]) + '\t', end='')
        print('')

def y2sensitivity(y_test, y_pred):
    params = ParameterSetup()
    stageLabels = params.stageLabels4evaluation
    labelNum = len(stageLabels)
    sensitivity = np.zeros((labelNum))
    specificity = np.zeros((labelNum))
    accuracy = np.zeros((labelNum))
    precision = np.zeros((labelNum))
    f1score = np.zeros((labelNum))
    for labelID in range(labelNum):
        targetLabel = stageLabels[labelID]
        # print('------------------------------')
        # print('type(y_test) =', type(y_test))
        # print('type(y_pred) =', type(y_pred))
        # print('y_test[:30] =', y_test[:30])
        # print('y_pred[:30] =', y_pred[:30])
        # print('targetLabel =', targetLabel)
        # print('------------------------------')
        TP = sum((y_test == targetLabel) & (y_pred == targetLabel))
        FP = sum((y_test != targetLabel) & (y_pred == targetLabel))
        FN = sum((y_test == targetLabel) & (y_pred != targetLabel))
        TN = sum((y_test != targetLabel) & (y_pred != targetLabel))
        sensitivity[labelID] = TP / (TP + FN) if TP + FN > 0 else 0
        specificity[labelID] = TN / (TN + FP) if TN + FP > 0 else 0
        accuracy[labelID] = (TP + TN) / (TP + FP + FN + TN)
        precision[labelID] = TP / (TP + FP) if TP + FP > 0 else 0
        f1score[labelID] = 2 * (precision[labelID] * sensitivity[labelID]) / (precision[labelID] + sensitivity[labelID]) if precision[labelID] + sensitivity[labelID] > 0 else 0
        # print('   for ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivity[labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[labelID]))
    return (stageLabels, sensitivity, specificity, accuracy, precision, f1score)
