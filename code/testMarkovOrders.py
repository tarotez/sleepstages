import numpy as np
import pickle
from parameterSetup import ParameterSetup
from fileManagement import getFileIDs
from evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat
from sequentialPrediction import classifySequentially
from featureExtractor import featureExtraction
from classifierTrainer import trainClassifier
from writeResults import writePredictions

orderMin = 0
orderMax = 8
# orderMax = 1
paramID = 0

params = ParameterSetup()
pickledDir = params.pickledDir
paramDir = params.pickledDir
classifierType = params.classifierType
classifierParams = params.classifierParams
eegFilePrefix = 'eegAndStage'

fileIDs = getFileIDs(pickledDir, eegFilePrefix)

print('classifier type = ' + str(classifierType))
print('useEMG = ' + str(params.useEMG))
print('emgTimeFrameNum = ' + str(params.emgTimeFrameNum))
print('binWidth4freqHisto = ' + str(params.binWidth4freqHisto))

stageLabels = params.stageLabels4evaluation
labelNum = len(stageLabels)
fileNum = len(fileIDs)

orderNum = orderMax - orderMin + 1
sensitivityMat = np.zeros((orderNum,labelNum), dtype=float)
specificityMat = np.zeros((orderNum,labelNum), dtype=float)
accuracyMat = np.zeros((orderNum,labelNum), dtype=float)
precisions = np.zeros((orderNum), dtype=float)

for markovOrder in range(orderMin, orderMax + 1):
    print('markovOrder = ' + str(markovOrder))
    params.pastStageLookUpNum = markovOrder

    for fileID in fileIDs:
        print('  extracting features for fileID = ' + str(fileID))
        featureExtraction(params, fileID)

    for fileID in fileIDs:
        print('  training and testing for fileID = ' + str(fileID))
        trainClassifier(params, paramID, fileID)
        (y_test, y_pred) = classifySequentially(params, paramID, paramDir, fileID)
        (stageLabels, sensitivity, specificity, accuracy) = y2sensitivity(y_test, y_pred)
        # (stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred)
        # printConfusionMat(stageLabels4confusionMat, confusionMat)

        y_matching = (y_test == y_pred)
        correctNum = sum(y_matching)
        # print('y_test = ' + str(y_test[:50]))
        # print('y_pred = ' + str(y_pred[:50]))
        y_length = y_pred.shape[0]
        precision = correctNum / y_length
        for labelID in range(labelNum):
            print('    stageLabel = ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivity[labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[labelID]))
            sensitivityMat[markovOrder,labelID] = sensitivity[labelID]
            specificityMat[markovOrder,labelID] = specificity[labelID]
            accuracyMat[markovOrder,labelID] = accuracy[labelID]
        print('    precision = ' + "{0:.5f}".format(precision) + ' (= ' + str(correctNum) + '/' + str(y_length) +')')
        precisions[markovOrder] = precision
        print('')
        resultFileDescription = 'markov.' + str(markovOrder)
        writePredictions(fileID, params, y_test, y_pred, resultFileDescription)

    saveData = (sensitivityMat, specificityMat, accuracyMat, precisions)
    outpath = open(pickledDir + '/res.testMarkovOrders.pkl', 'wb')
    pickle.dump(saveData, outpath)

print('in the final summary:')
print('classifier type = ' + str(classifierType))
print('useEMG = ' + str(params.useEMG))
print('emgTimeFrameNum = ' + str(params.emgTimeFrameNum))
print('binWidth4freqHisto = ' + str(params.binWidth4freqHisto))
print('precisions by different orders:')
for markovOrder in range(orderMin, orderMax + 1):
    print(str(markovOrder) + ', ' + "{0:.5f}".format(precisions[markovOrder]))
