import sys
from os.path import splitext
import numpy as np
from parameterSetup import ParameterSetup
from fileManagement import getFileIDsCorrespondingToClassifiers
from evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat, mathewsCorrelationCoefficient, multiClassMCC
from sequentialPrediction import classifySequentially
from writePredictionResults import writePredictionResults
from computeTestError import printMetadata, printStatistics
import matplotlib.pyplot as plt
from functools import reduce

#---------------
# main
args = sys.argv
if len(args) > 2:
    excludedFileID = args[1]
    classifierID = args[2]
    fileIDpair = [excludedFileID, classifierID]
    fileIDpairs = []
    fileIDpairs.append(fileIDpair)
    if len(args) > 3:
        jsonFilePath = args[3]
        elems = jsonFilePath.split('/')
        paramDir = '/'.join(elems[:-1])
        paramFileName = elems[-1]
        paramFileID, extension = splitext(paramFileName)
        outputDir = paramDir + '/' + paramFileID
        params = ParameterSetup(paramDir, paramFileName, outputDir)
    else:
        params = ParameterSetup()
elif len(args) == 2:
    excludedFileID = args[1]
    fileIDpairs = [[excludedFileID, '']]
elif len(args) == 1:
    fileIDpairs = getFileIDsCorrespondingToClassifiers(pickledDir, classifierFilePrefix)

if len(args) <= 3:
    params = ParameterSetup()

print('# fileIDpairs =', fileIDpairs)

pickledDir = params.pickledDir
classifierType = params.classifierType
classifierParams = params.classifierParams

extractorType = params.extractorType
deepParamsDir = params.deepParamsDir

featureFilePrefix = params.featureFilePrefix
classifierFilePrefix = params.classifierFilePrefix
resultFileDescription = ''

stageLabels = params.stageLabels4evaluation
labelNum = len(stageLabels)
fileNum = len(fileIDpairs)
totalConfusionMat = np.zeros((labelNum, labelNum))
markovOrders = np.arange(params.markovOrderForTraining)

paramNum = len(classifierParams)
for paramID in range(paramNum):
    print('classifier parameter = ' + str(classifierParams[paramID]))
    for fileIDpair in fileIDpairs:
        precs = []
        mcMCCs = []
        mccs = []
        for markovOrder in markovOrders:
            params.markovOrderForPrediction = markovOrder
            printMetadata(params)

            # print('fileIDpair = ' + str(fileIDpair))
            (y_test, y_pred) = classifySequentially(params, paramID, fileIDpair)

            (stageLabels, sensitivity, specificity, accuracy) = y2sensitivity(y_test, y_pred)
            (stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred)
            printConfusionMat(stageLabels4confusionMat, confusionMat)
            totalConfusionMat = totalConfusionMat + confusionMat

            y_matching = (y_test == y_pred)
            correctNum = sum(y_matching)
            # print('y_test = ' + str(y_test[:50]))
            # print('y_pred = ' + str(y_pred[:50]))
            y_length = y_pred.shape[0]
            precision = correctNum / y_length
            # print('labelNum = ' + str(labelNum))
            for targetStage in ['W','S','R']:
                mcc = mathewsCorrelationCoefficient(stageLabels4confusionMat, confusionMat, targetStage)
                mccs.append((markovOrder, targetStage, mcc))

                # print('  mcc for ' + targetStage + ' = ' + "{0:.5f}".format(mcc))
            mcMCC = multiClassMCC(stageLabels4confusionMat, confusionMat)
            mcMCCs.append(mcMCC)
            # print('  multi class mcc = ' + "{0:.5f}".format(mcc))
            precs.append(precision)

            # print('')
            writePredictionResults(fileIDpair, params, y_test, y_pred, resultFileDescription)

        print('fileIDpair =', fileIDpair)
        print('precs =', precs)
        print('mcMCC =', mcMCCs)
        print('mccs =', mccs)

        plt.plot(np.array(precs))
        plt.plot(np.array(mcMCCs))

        stage2ID = {'W':0, 'S':1, 'R':2}
        empt = [[] for _ in range(len(stage2ID))]
        def categorizeMCC(LL, x):
            return [elemL + [x[2]] if stage2ID[x[1]] == stageID else elemL for stageID, elemL in enumerate(LL)]
        mccsMatL = reduce(categorizeMCC, mccs, empt)
        plt.plot(np.array(mccsMatL).transpose())
        plt.legend(stage2ID.keys())
