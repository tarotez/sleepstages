import numpy as np
import pickle
from parameterSetup import ParameterSetup
from evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat, mathewsCorrelationCoefficient, multiClassMCC
from sequentialPrediction import classifySequentially
from writePredictionResults import writePredictionResults
from fileManagement import readTrainFileIDsUsedForTraining, getFilesNotUsedInTrain

def printMetadata(params):
    print('extractor type = ' + str(params.extractorType))
    print('classifier type = ' + str(params.classifierType))
    print('classifierParams = ' + str(params.classifierParams))
    print('pastStageLookUpNum = ' + str(params.pastStageLookUpNum))
    print('timeWindowStrideInSec = ' + str(params.timeWindowStrideInSec))
    print('useEMG = ' + str(params.useEMG))
    print('numOfConsecutiveWsThatProhibitsR = ' + str(params.numOfConsecutiveWsThatProhibitsR))
    print('markovOrderForPrediction = ' + str(params.markovOrderForPrediction))
    # print('ch2TimeFrameNum = ' + str(params.ch2TimeFrameNum))
    # print('binWidth4freqHisto = ' + str(params.binWidth4freqHisto))


def meanStatisticsForCV(sensitivityT, specificityT, accuracyT, precisionT, f1scoreT, mccT, mcMCCT, mcAccuracyT, confusionMatT, stageLabels):
    methodNum, blockNum, testFileNum, labelNum = sensitivityT.shape
    # print('in meanStatistics, labelNum =', labelNum)
    # print('  testFileNum =', testFileNum)
    sensitivityMeans = np.zeros((methodNum,labelNum), dtype=float)
    specificityMeans = np.zeros((methodNum,labelNum), dtype=float)
    accuracyMeans = np.zeros((methodNum,labelNum), dtype=float)
    precisionMeans = np.zeros((methodNum,labelNum), dtype=float)
    f1scoreMeans = np.zeros((methodNum,labelNum), dtype=float)
    mccMeans = np.zeros((methodNum,labelNum), dtype=float)
    mcMCCMeans = np.zeros((methodNum), dtype=float)
    mcAccuracyMeans = np.zeros((methodNum), dtype=float)
    for methodID in range(methodNum):
        for labelID in range(labelNum):
            # print('  labelID =', labelID, end='')
            # print('  stageLabel = ' + stageLabels[labelID], end='')
            sensitivityMeans[methodID,labelID] = sensitivityT[methodID,:,:,labelID].mean()
            # print('  sensitivity = ' + "{0:.3f}".format(sensitivityMeans[methodID,labelID]), end='')
            specificityMeans[methodID,labelID] = specificityT[methodID,:,:,labelID].mean()
            # print('  specificity = ' + "{0:.3f}".format(specificityMeans[methodID,labelID]), end='')
            accuracyMeans[methodID,labelID] = accuracyT[methodID,:,:,labelID].mean()
            # print('  accuracy = ' + "{0:.3f}".format(accuracyMeans[methodID,labelID]), end='')
            precisionMeans[methodID,labelID] = precisionT[methodID,:,:,labelID].mean()
            # print('  precision = ' + "{0:.3f}".format(precisionMeans[methodID,labelID]), end='')
            f1scoreMeans[methodID,labelID] = f1scoreT[methodID,:,:,labelID].mean()
            # print('  f1score = ' + "{0:.3f}".format(f1scoreMeans[methodID,labelID]), end='')
            mccMeans[methodID,labelID] = mccT[methodID,:,:,labelID].mean()
            # print('  mcc = ' + "{0:.3f}".format(mccMeans[methodID,labelID]))
        mcMCCMeans[methodID] = mcMCCT[methodID,:,:].mean()
        # print('  mcMCC = ' + "{0:.3f}".format(mcMCCMeans[methodID]))
        mcAccuracyMeans[methodID] = mcAccuracyT[methodID,:,:].mean()
        # print('  mcAccuracy = ' + "{0:.3f}".format(mcAccuracyMeans[methodID]))
    return sensitivityMeans, specificityMeans, accuracyMeans, precisionMeans, f1scoreMeans, mccMeans, mcMCCMeans, mcAccuracyMeans

def meanStatistics(sensitivityL, specificityL, accuracyL, precisionL, f1scoreL, mccL, mcMCCL, mcAccuracyL, stageLabels, labelNum, fileNum):
    # print('in meanStatistics, labelNum =', labelNum)
    # print('  fileNum =', fileNum)
    sensitivityMeans = np.zeros((labelNum), dtype=float)
    specificityMeans = np.zeros((labelNum), dtype=float)
    accuracyMeans = np.zeros((labelNum), dtype=float)
    precisionMeans = np.zeros((labelNum), dtype=float)
    f1scoreMeans = np.zeros((labelNum), dtype=float)
    mccMeans = np.zeros((labelNum), dtype=float)
    for labelID in range(labelNum):
        # print('  labelID =', labelID, end='')
        # print('  stageLabel = ' + stageLabels[labelID], end='')
        sensitivityMeans[labelID] = np.array(sensitivityL[labelID]).mean()
        # print('  sensitivity = ' + "{0:.3f}".format(sensitivityMeans[labelID]), end='')
        specificityMeans[labelID] = np.array(specificityL[labelID]).mean()
        # print('  specificity = ' + "{0:.3f}".format(specificityMeans[labelID]), end='')
        accuracyMeans[labelID] = np.array(accuracyL[labelID]).mean()
        # print('  accuracy = ' + "{0:.3f}".format(accuracyMeans[labelID]), end='')
        precisionMeans[labelID] = np.array(precisionL[labelID]).mean()
        # print('  precision = ' + "{0:.3f}".format(precisionMeans[labelID]), end='')
        f1scoreMeans[labelID] = np.array(f1scoreL[labelID]).mean()
        # print('  f1score = ' + "{0:.3f}".format(f1scoreMeans[labelID]), end='')
        mccMeans[labelID] = np.array(mccL[labelID]).mean()
        # print('  mcc = ' + "{0:.3f}".format(mccMeans[labelID]))
    mcMCCMean = np.array(mcMCCL).mean()
    # print('  mcMCC = ' + "{0:.3f}".format(mcMCCMean))
    mcAccuracyMean = np.array(mcAccuracyL).mean()
    # print('  mcAccuracy = ' + "{0:.3f}".format(mcAccuracyMean))
    return sensitivityMeans, specificityMeans, accuracyMeans, precisionMeans, f1scoreMeans, mccMeans, mcMCCMean, mcAccuracyMean

def saveStatistics(pickledDir, classifierID, testFileIDandClassifierIDs, sensitivityL, specificityL, accuracyL, precisionL, f1scoreL, mccL, mcMCCL, mcAccuracyL, confusionMatL, stageLabels, fileNum, labelNum, datasetType):
    if datasetType == 'test':
        f = open(pickledDir + '/test_result.' + classifierID + '.test.pkl', 'wb')
    else:
        f = open(pickledDir + '/test_result.' + classifierID + '.pkl', 'wb')
    pickle.dump((testFileIDandClassifierIDs, sensitivityL, specificityL, accuracyL, precisionL, f1scoreL, mccL, mcMCCL, mcAccuracyL, confusionMatL, stageLabels, fileNum, labelNum), f)
    f.close()

def test_by_classifierID(params, datasetType, classifierID):
    paramDir = params.pickledDir
    testFileDir = params.pickledDir
    stageLabels = params.stageLabels4evaluation
    labelNum = len(stageLabels)
    resultFileDescription = ''
    paramID = 0
    markovOrder = 0
    fileTripletL = readTrainFileIDsUsedForTraining(params, classifierID)
    train_fileIDs = [fileID for _, _, fileID in fileTripletL]
    # print('# train_fileIDs =', train_fileIDs)
    params_test = params
    if datasetType == 'test':
        params_test.pickledDir = testFileDir
    test_fileTripletL = getFilesNotUsedInTrain(params_test, train_fileIDs)
    testFileIDandClassifierIDs = [(test_fileID, classifierID) for _, _, test_fileID in test_fileTripletL]
    fileNum = len(testFileIDandClassifierIDs)
    print('# testFileIDandClassifierIDs =', testFileIDandClassifierIDs)
    # totalConfusionMat = np.zeros((labelNum, labelNum))

    # for paramID in range(len(classifierParams)):
    #     print('classifier parameter = ' + str(classifierParams[paramID]))
    sensitivityL = [[] for _ in range(labelNum)]
    specificityL = [[] for _ in range(labelNum)]
    accuracyL = [[] for _ in range(labelNum)]
    precisionL = [[] for _ in range(labelNum)]
    f1scoreL = [[] for _ in range(labelNum)]
    mccL = [[] for _ in range(labelNum)]
    mcMCCL = []
    mcAccuracyL = []
    confusionMatL = []

    for testFileIDandClassifierID in testFileIDandClassifierIDs:

        print('testFileIDandClassifierID = ' + str(testFileIDandClassifierID))
        params_for_classifier = ParameterSetup(paramFileName='params.'+classifierID+'.json')
        params_for_classifier.markovOrderForPrediction = markovOrder
        (y_test, y_pred) = classifySequentially(params_for_classifier, paramID, paramDir, testFileIDandClassifierID)

        print('y_test =', y_test)
        print('type(y_test) =', type(y_test))
        y_test = np.array(['W' if elem == 'RW' else elem for elem in y_test])
        print('after replace: y_test =', y_test)
        print('after replace: type(y_test) =', type(y_test))

        # ignore ?'s in the beginning produced by
        # i = 0
        # while y_pred[i] == '?':
        #    i++
        if params.classifierType == 'deep':
            i = params.torch_lstm_length - 1   # remove from all clalssifiers because LSTM cannot predict first 9 elements.
        else:
            i = 0
        print('for classifier ', testFileIDandClassifierID, ', first ', i, ' elements are removed.', sep='')

        y_test, y_pred = y_test[i:], y_pred[i:]

        (stageLabels, sensitivity, specificity, accuracy, precision, f1score) = y2sensitivity(y_test, y_pred)
        (stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred, params.stageLabels4evaluation)
        printConfusionMat(stageLabels4confusionMat, confusionMat)
        # totalConfusionMat = totalConfusionMat + confusionMat

        # print('y_test = ' + str(y_test[:50]))
        # print('y_pred = ' + str(y_pred[:50]))
        y_length = y_pred.shape[0]
        print('stageLabels =', stageLabels)
        print('labelNum = ' + str(labelNum))
        for labelID in range(labelNum):
            targetLabel = stageLabels[labelID]
            sensitivityL[labelID].append(sensitivity[labelID])
            specificityL[labelID].append(specificity[labelID])
            accuracyL[labelID].append(accuracy[labelID])
            precisionL[labelID].append(precision[labelID])
            f1scoreL[labelID].append(f1score[labelID])
            mcc = mathewsCorrelationCoefficient(stageLabels4confusionMat, confusionMat, targetLabel)
            mccL[labelID].append(mcc)
            print('  targetLabel = ' + targetLabel + ', sensitivity = ' + "{0:.3f}".format(sensitivity[labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[labelID])+ ', precision = ' + "{0:.3f}".format(precision[labelID]))
            print('     mcc for ' + targetLabel + ' = ' + "{0:.5f}".format(mcc))
        mcMCCL.append(multiClassMCC(confusionMat))
        print('  multi-class mcc = ' + "{0:.5f}".format(mcMCCL[-1]))
        mcAccuracyL.append(sum(y_test == y_pred) / len(y_test))
        print('  multi-class accuracy = ' + "{0:.5f}".format(mcAccuracyL[-1]))
        confusionMatL.append(confusionMat)
        print('')
        writePredictionResults(testFileIDandClassifierID, params, y_test, y_pred, resultFileDescription)

    if datasetType == 'test':
        f = open(pickledDir + '/test_result.' + classifierID + '.test.pkl', 'wb')
    else:
        f = open(pickledDir + '/test_result.' + classifierID + '.pkl', 'wb')
    pickle.dump((testFileIDandClassifierIDs, sensitivityL, specificityL, accuracyL, precisionL, f1scoreL, mccL, mcMCCL, mcAccuracyL, confusionMatL, stageLabels, fileNum, labelNum), f)
    f.close()

    #-----
    # show the summary (average) of the result
    print('Summary for classifierID ' + classifierID + ':')
    printMetadata(params)
    saveStatistics(params.pickledDir, classifierID, testFileIDandClassifierIDs, sensitivityL, specificityL, accuracyL, precisionL, f1scoreL, mccL, mcMCCL, mcAccuracyL, confusionMatL, stageLabels, fileNum, labelNum, datasetType)
    # print('ch2TimeFrameNum = ' + str(params.ch2TimeFrameNum))
    # print('binWidth4freqHisto = ' + str(params.binWidth4freqHisto))
    sensitivityMeans, specificityMeans, accuracyMeans, precisionMean, f1scoreMean, mccMeans, mcMCCMean, mcAccuracyMean = meanStatistics(sensitivityL, specificityL, accuracyL, precisionL, f1scoreL, mccL, mcMCCL, mcAccuracyL, stageLabels, labelNum, fileNum)
    # sensitivity_by_classifier_L.append(sensitivityMeans)
    # specificity_by_classifier_L.append(specificityMeans)
    # accuracy_by_classifier_L.append(accuracyMeans)
    # precision_by_classifier_L.append(precisionMean)
    # measures_by_classifier_L.append([sensitivityMeans, specificityMeans, accuracyMeans, precisionMean, f1scoreMean, mccMeans, mcMCCMean, mcAccuracyMean])
    return [sensitivityMeans, specificityMeans, accuracyMeans, precisionMean, f1scoreMean, mccMeans, mcMCCMean, mcAccuracyMean]
