import numpy as np
import pickle
from writePredictionResults import writePredictionResults
from evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat, mathewsCorrelationCoefficient, multiClassMCC
from tester import printMetadata

def getStatistics(params, testFileIDandClassifierIDs_byMethod, y_test_byMethod, y_pred_byMethod, crossValidationID):

    paramDir = params.pickledDir
    testFileDir = params.pickledDir
    stageLabels = params.stageLabels4evaluation
    labelNum = len(stageLabels)
    resultFileDescription = ''
    paramID = 0
    markovOrder = 0
    # fileTripletL = readTrainFileIDsUsedForTraining(params, classifierID)
    # train_fileIDs = [fileID for _, _, fileID in fileTripletL]
    # print('# train_fileIDs =', train_fileIDs)
    params_test = params
    # if datasetType == 'test':
    params_test.pickledDir = testFileDir

    # totalConfusionMat = np.zeros((labelNum, labelNum))
    # for paramID in range(len(classifierParams)):
    #     print('classifier parameter = ' + str(classifierParams[paramID]))
    methodNum = len(testFileIDandClassifierIDs_byMethod)
    blockNum = len(testFileIDandClassifierIDs_byMethod[0])
    testFileNum = len(testFileIDandClassifierIDs_byMethod[0][0])
    sensitivityT = np.zeros((methodNum, blockNum, testFileNum, labelNum))
    specificityT = np.zeros((methodNum, blockNum, testFileNum, labelNum))
    accuracyT = np.zeros((methodNum, blockNum, testFileNum, labelNum))
    precisionT = np.zeros((methodNum, blockNum, testFileNum, labelNum))
    f1scoreT = np.zeros((methodNum, blockNum, testFileNum, labelNum))
    mccT = np.zeros((methodNum, blockNum, testFileNum, labelNum))
    mcMCCT = np.zeros((methodNum, blockNum, testFileNum))
    mcAccuracyT = np.zeros((methodNum, blockNum, testFileNum))
    confusionMatT = [[[[] for _ in range(testFileNum)] for _ in range(blockNum)] for _ in range(methodNum)]

    for methodID, (y_test_byBlock, y_pred_byBlock) in enumerate(zip(y_test_byMethod, y_pred_byMethod)):
        for blockID, (y_test_byFile, y_pred_byFile) in enumerate(zip(y_test_byBlock, y_pred_byBlock)):
            testFileIDandClassifierIDs = [(test_fileID, classifierID) for test_fileID, classifierID in testFileIDandClassifierIDs_byMethod[methodID][blockID]]
            print('# testFileIDandClassifierIDs[:3] =', testFileIDandClassifierIDs[:3])
            for testFileCnt, ((testFileID, classifierID), y_test, y_pred) in enumerate(zip(testFileIDandClassifierIDs, y_test_byFile, y_pred_byFile)):
                print('testFileID = ', testFileID, ': classifierID =', classifierID)
                # test_fileTripletL = getFilesNotUsedInTrain(params_test, train_fileIDs)
                print('y_test =', y_test)
                print('type(y_test) =', type(y_test))
                y_test = np.array(['W' if elem == 'RW' else elem for elem in y_test])
                print('after replace: y_test =', y_test)
                print('after replace: type(y_test) =', type(y_test))
                # ignore ?'s in the beginning produced by
                # i = 0
                # while y_pred[i] == '?':
                #    i++
                # remove from all clalssifiers because LSTM cannot predict first 9 elements.
                i = params.torch_lstm_length - 1 if params.classifierType == 'deep' else 0

                y_test, y_pred = y_test[i:], y_pred[i:]
                (stageLabels, sensitivity, specificity, accuracy, precision, f1score) = y2sensitivity(y_test, y_pred)
                (stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred, params.stageLabels4evaluation)
                printConfusionMat(stageLabels4confusionMat, confusionMat)
                # print('y_test = ' + str(y_test[:50]))
                # print('y_pred = ' + str(y_pred[:50]))
                y_length = y_pred.shape[0]
                print('stageLabels =', stageLabels)
                print('labelNum = ' + str(labelNum))
                for labelID in range(labelNum):
                    targetLabel = stageLabels[labelID]
                    sensitivityT[methodID, blockID, testFileCnt, labelID] = sensitivity[labelID]
                    specificityT[methodID, blockID, testFileCnt, labelID] = specificity[labelID]
                    accuracyT[methodID, blockID, testFileCnt, labelID] = accuracy[labelID]
                    precisionT[methodID, blockID, testFileCnt, labelID] = precision[labelID]
                    f1scoreT[methodID, blockID, testFileCnt, labelID] = f1score[labelID]
                    mccT[methodID, blockID, testFileCnt, labelID] = mathewsCorrelationCoefficient(stageLabels4confusionMat, confusionMat, targetLabel)
                    print('  targetLabel = ' + targetLabel + ', sensitivity = ' + "{0:.3f}".format(sensitivity[labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[labelID])+ ', precision = ' + "{0:.3f}".format(precision[labelID]))
                    # print('     mcc for ' + targetLabel + ' = ' + "{0:.5f}".format(mccT[methodID, blockID, testFileCnt]))
                mcMCCT[methodID, blockID, testFileCnt] = multiClassMCC(confusionMat)
                print('  multi-class mcc = ' + "{0:.5f}".format(mcMCCT[methodID, blockID, testFileCnt]))
                mcAccuracyT[methodID, blockID, testFileCnt] = sum(y_test == y_pred) / len(y_test)
                print('  multi-class accuracy = ' + "{0:.5f}".format(mcAccuracyT[methodID, blockID, testFileCnt]))
                confusionMatT[methodID][blockID][testFileCnt] = confusionMat
                print('')
                # writePredictionResults(testFileIDandClassifierIDs, params, y_test, y_pred, resultFileDescription)
                print('writing to ' + params.pickledDir + '/test_result.' + classifierID + '.pkl')
    res = (sensitivityT, specificityT, accuracyT, precisionT, f1scoreT, mccT, mcMCCT, mcAccuracyT, confusionMatT, stageLabels)
    with open(params.pickledDir + '/test_result.' + crossValidationID + '.pkl', 'wb') as f:
        pickle.dump(res, f)
    return res
    #-----
    # show the summary (average) of the result
    # print('Summary for classifierID ' + classifierID + ':')
    # printMetadata(params)
    # saveStatistics(params.pickledDir, classifierID, testFileIDandClassifierIDs, sensitivityL, specificityL, accuracyL, precisionL, f1scoreL, mccL, mcMCCL, mcAccuracyL, confusionMatL, stageLabels, fileNum, labelNum, datasetType)
    # print('ch2TimeFrameNum = ' + str(params.ch2TimeFrameNum))
    # print('binWidth4freqHisto = ' + str(params.binWidth4freqHisto))
    # sensitivityMeans, specificityMeans, accuracyMeans, precisionMean, f1scoreMean, mccMeans, mcMCCMean, mcAccuracyMean = meanStatisticsForCV(sensitivityT, specificityT, accuracyT, precisionT, f1scoreT, mccT, mcMCCT, mcAccuracyT, stageLabels)
    # return [sensitivityMeans, specificityMeans, accuracyMeans, precisionMean, f1scoreMean, mccMeans, mcMCCMean, mcAccuracyMean]

print('*********')
print('crossValidation_computeStatistics.py only defines function getStatistics(). ')
print('Not to be used by itself')
print('*********')

'''
args = sys.argv
crossValidationID = args[1]
params = ParameterSetup()
with open(params.pickledDir + '/y_test_and_y_pred_for_graphs.' + crossValidationID + '.pkl','rb') as f:
    testFileIDandClassifierIDs_byMethod, y_test_byMethod, y_pred_byMethod = pickle.load(f)
    res = getStatistics(params, testFileIDandClassifierIDs_byMethod, y_test_byMethod, y_pred_byMethod, crossValidationID)
    print(res)
'''
