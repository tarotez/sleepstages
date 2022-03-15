import pickle
import numpy as np
from os.path import dirname, abspath
from parameterSetup import ParameterSetup
from stagePredictor import StagePredictor
from statistics import standardizer
from algorithmFactory import AlgorithmFactory
from deepClassifier import DeepClassifier

def classifySequentially(params, paramID, paramDir, fileIDpair):

    # print('classifySequentially by ' + str(fileID))
    pickledDir = params.pickledDir
    extractorType = params.extractorType
    classifierType = params.classifierType
    classifierParams = params.classifierParams
    samplingFreq = params.samplingFreq
    windowSizeInSec = params.windowSizeInSec
    wsizeInSamplePointNum = windowSizeInSec * samplingFreq   # window size in sample points. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
    timeWindowStrideInSec = params.timeWindowStrideInSec
    timeWindowStrideInSamplePointNum = timeWindowStrideInSec * samplingFreq
    markovOrderForPrediction = params.markovOrderForPrediction
    strideNumInTimeWindow = np.ceil(windowSizeInSec / timeWindowStrideInSec)
    classifierType = params.classifierType
    eegFilePrefix = params.eegFilePrefix
    classifierFilePrefix = params.classifierFilePrefix
    # replacesWWWRtoWWWW = params.replacesWWWRtoWWWW
    numOfConsecutiveWsThatProhibitsR = params.numOfConsecutiveWsThatProhibitsR
    predict_by_batch = params.predict_by_batch
    testFileID = fileIDpair[0]
    classifierID = fileIDpair[1]
    print('$#$#$# in sequentialPrediction, classifierID =', classifierID)

    predictionTargetDataFilePath = pickledDir + '/' + eegFilePrefix + '.' + testFileID + '.pkl'
    print('predictionTargetDataFilePath =', predictionTargetDataFilePath)
    dataFileHandler = open(predictionTargetDataFilePath, 'rb')
    (eeg, ch2, stageSeq, timeStamps) = pickle.load(dataFileHandler)
    dataFileHandler.close()
    print('eeg.shape = ' + str(eeg.shape))
    print('len(stageSeq) = ' + str(len(stageSeq)))

    if params.useEMG:
        label4EMG = params.label4withEMG
    else:
        label4EMG = params.label4withoutEMG

    params.classifierName = classifierFilePrefix + '.' + classifierType +  '.' + label4EMG + '.excludedFileID.' + testFileID + '.classifierID.' + classifierID
    factory = AlgorithmFactory(extractorType)
    extractor = factory.generateExtractor()

    # if params.classifierType == 'deep':
    #    classifier = DeepClassifier(classifierID=testFileID)
    # else:
    classifierDir = params.classifierDir
    pwd = dirname(abspath(__file__))
    print('# classifierID =', classifierID)
    # classLabels = list(params.labelCorrectionDict.keys())[:params.maximumStageNum]
    classLabels = params.sampleClassLabels[:params.maximumStageNum]
    paramFileName = 'params.' + classifierID + '.json'
    params_for_network_structure = ParameterSetup(paramDir=paramDir, paramFileName=paramFileName)
    if params.classifierType == 'deep':
        classifier = DeepClassifier(classLabels, classifierID=classifierID, paramsForDirectorySetup=params, paramsForNetworkStructure=params_for_network_structure)
        # if classifierID == '':
        #    model_path = pwd + '/' + paramDir + '/model.pkl'
        #else:
        #    model_path = pwd + '/' + params.deepParamsDir + '/' + testFileID + '/' + classifierID + '/model.pkl'
        model_path = pwd + '/' + paramDir + '/weights.' + classifierID + '.pkl'
        print('model_path = ', model_path)
        classifier.load_weights(model_path)
    else:
        classifierFileName = params.classifierDir + '/' + params.classifierPrefix + '.' + classifierID + '.pkl'
        classifierFileHandler = open(classifierFileName, 'rb')
        classifier = pickle.load(classifierFileHandler)
        classifierFileHandler.close()

    stagePredictor = StagePredictor(params_for_network_structure, extractor, classifier, classifierDir, classifierID, markovOrderForPrediction)
    ### stagePredictor = StagePredictor(params, extractor, classifier, classifierDir, classifierID)
    # stagePredictor = StagePredictor(params, extractor, classifier)

    sLen = len(stageSeq)
    samplePointNum = min(eeg.shape[0], sLen * wsizeInSamplePointNum)
    print('%$%$%$ sLen =', sLen)
    print('eeg.shape[0] =', eeg.shape[0])
    print('sLen * wsizeInSamplePointNum =', sLen * wsizeInSamplePointNum)
    print('%$%$%$ samplePointNum =', samplePointNum)
    print('%$%$%$ timeWindowStrideInSamplePointNum =', timeWindowStrideInSamplePointNum)
    # reset eeg and emg statistical values
    # eeg_mean = 0
    # eeg_variance = 0
    # emg_mean = 0
    # emg_variance= 0
    # oldSampleNum = 0

    y_test_L = []
    y_pred_L = []
    wID = 0
    # startSamplePoint = 0
    # stagePredictions_L = []
    replacedR = False
    # while startSamplePoint + wsizeInSamplePointNum <= samplePointNum:
    records_L = []
    timeStampSegments_L = []
    ### all_past_eeg, all_past_ch2 = np.array([]), np.array([])
    standardizer_eeg = standardizer(samplePointNum)
    standardizer_ch2 = standardizer(samplePointNum)
    # standardized_all_past_eeg = np.array([])
    for startSamplePoint in range(0, samplePointNum, timeWindowStrideInSamplePointNum):
        endSamplePoint = startSamplePoint + wsizeInSamplePointNum
        if endSamplePoint > samplePointNum:
            break
        timeStampSegment = timeStamps[startSamplePoint:endSamplePoint]
        eegSegment = eeg[startSamplePoint:endSamplePoint]
        ### standardized_eegSegment, all_past_eeg = standardize(eegSegment, all_past_eeg)
        standardized_eegSegment = standardizer_eeg.standardize(eegSegment)

        if params.useEMG:
            ch2Segment = ch2[startSamplePoint:endSamplePoint]
            ### standardized_ch2Segment, all_past_ch2 = standardize(ch2Segment, all_past_ch2)
            standardized_ch2Segment = standardizer_ch2.standardize(ch2Segment)
            one_record = np.r_[standardized_eegSegment, standardized_ch2Segment]
        else:
            one_record = standardized_eegSegment
        # standardized_all_past_eeg = np.r_[standardized_all_past_eeg, standardized_eegSegment]
        # print('standardized_all_past_eeg.mean() =', standardized_all_past_eeg.mean(), ', standardized_all_past_eeg.std() =', standardized_all_past_eeg.std())
        '''
        eegSegment_orig_mean = np.mean(eegSegment)
        eegSegment_orig_std = np.std(eegSegment)
        eeg_old_mean = eeg_mean
        eeg_mean = recompMean(eegSegment, eeg_mean, oldSampleNum)
        eeg_variance = recompVariance(eegSegment, eeg_variance, eeg_old_mean, eeg_mean, oldSampleNum)
        standardized_eegSegment = (eegSegment - eeg_mean) / np.sqrt(eeg_variance)
        if params.usech2:
            ch2Segment = ch2[startSamplePoint:endSamplePoint]
            ch2_old_mean = ch2_mean
            ch2_mean = recompMean(ch2Segment, ch2_mean, oldSampleNum)
            ch2_variance = recompVariance(ch2Segment, ch2_variance, ch2_old_mean, ch2_mean, oldSampleNum)
            standardized_ch2Segment = (ch2Segment - ch2_mean) / np.sqrt(ch2_variance)
            one_record = np.r_[standardized_eegSegment, standardized_ch2Segment]
        else:
            one_record = standardized_eegSegment
        oldSampleNum += eegSegment.shape[0]
        '''

        # local_mu = np.mean(standardized_eegSegment)
        # local_sigma = np.std(standardized_eegSegment)
        # print('local_mu =', local_mu)
        # print(' ')
        # print('in sequentialPredictionDecisionTree.classifySequentially():')
        ### print('one_record.shape = ' + str(one_record.shape))
        ### print('one_record[:] = ' + str(one_record[:]))
        # print('one_record[:10] = ' + str(one_record[:10]))
        # print('one_record[:,1] = ' + str(one_record[:,1]))
        # stageEstimate is one of ['w', 'n', 'r']
        # stagePrediction = params.reverseLabel(stagePredictor.predict(one_record, local_mu=eegSegment_orig_mean, local_sigma=eegSegment_orig_std))
        # print('np.mean(one_record[:,0]) = ' + str(np.mean(one_record[:])))
        # print('np.std(one_record[:,0]) = ' + str(np.std(one_record[:])))
        # print('before stagePredictor.predict')
        # stagePrediction = params.reverseLabel(stagePredictor.predict(one_record, timeStampSegment, local_mu=local_mu, local_sigma=local_sigma, wID=wID))
        if predict_by_batch:
            records_L.append(one_record)
            timeStampSegments_L.append(timeStampSegment)
        else:
            orig_prediction = stagePredictor.predict(one_record, timeStampSegment, params.stageLabels4evaluation, params.stageLabel2stageID)
            # print(orig_prediction, end='')
            stagePrediction = params.reverseLabel(orig_prediction)
            # print('after stagePredictor.predict')

            # replaces R to W when W appears consecutively
            '''
            Ws = 'W' * numOfConsecutiveWsThatProhibitsR
            for wCnt in range(1,numOfConsecutiveWsThatProhibitsR+1):
                if len(y_pred_L) >= wCnt:
                    if y_pred_L[len(y_pred_L)-wCnt] != 'W':
                        break
                if stagePrediction == 'R':
                    print(Ws + '->R changed to ' + Ws + '->W at wID = ' + str(wID) + ', startSamplePoint = ' + str(startSamplePoint))
                    stagePrediction = 'W'
                    replacedR = True
                    '''
            #----
            # if the prediction is P, then use the previous one
            if stagePrediction == 'P':
                # print('stagePrediction == P for wID = ' + str(wID))
                if len(y_pred_L) > 0:
                    stagePrediction = y_pred_L[len(y_pred_L)-1]
                else:
                    stagePrediction = 'M'

            #-----
            # append to the lists of results
            y_pred_L.append(stagePrediction)
            if wID >= len(stageSeq):
                print('len(stageSeq) =', len(stageSeq), ', wID =', wID)
                print('startSamplePoint =', startSamplePoint, ', samplePointNum =', samplePointNum, ', timeWindowStrideInSamplePointNum =', timeWindowStrideInSamplePointNum)

            trueLabel = stageSeq[wID]
            if replacedR:
                print('  -> for wID = ' + str(wID) + ', trueLabel = ' + trueLabel)
                replacedR = False
            y_test_L.append(trueLabel)
            wID += 1

            #-----
            # vote for choosing the label for 10 second window
            '''
            stagePredictions_L.append(stagePrediction)
            if len(stagePredictions_L) == strideNumInTimeWindow:
                finalStagePrediction = voteForStagePrediction(stagePredictions_L[-(markovOrderForPrediction+1):])
                #-----
                # append to the lists of results
                y_pred_L.append(finalStagePrediction)
                stagePredictions_L = []
                # print('stagePredictions_L[' + stagePrediction + '] = ' + )
                # startSamplePoint = endSamplePoint
                trueLabel = stageSeq[wID]
                if replacedR:
                    print('  -> for wID = ' + str(wID) + ', trueLabel = ' + trueLabel)
                    replacedR = False
                y_test_L.append(trueLabel)
                ### print('wID = ' + str(wID) + ', trueLabel = ' + trueLabel + ', stagePrediction = ' + stagePrediction)
                wID += 1
            '''

    if predict_by_batch:
        # print('predicting by batch:')
        y_pred = np.array([params.reverseLabel(y_pred_orig) for y_pred_orig in stagePredictor.batch_predict(np.array(records_L), np.array(timeStampSegments_L), local_mu=local_mu, local_sigma=local_sigma, stageLabels4evaluation=params.stageLabels4evaluation, stageLabel2stageID=params.stageLabel2stageID)])
        y_test = np.array(stageSeq)
    else:
        y_test = np.array(y_test_L)
        y_pred = np.array(y_pred_L)

    return (y_test, y_pred)

'''
# do voting to get the final prediction
def voteForStagePrediction(stagePredictions_L):
    stagesCnt = {}
    # print('in vote:')
    for stagePrediction in stagePredictions_L:
        # print('  stagePrediction = ' + stagePrediction)
        if stagesCnt.__contains__(stagePrediction):
            stagesCnt[stagePrediction] = stagesCnt[stagePrediction] + 1
        else:
            stagesCnt[stagePrediction] = 1
    finalStagePrediction = ''
    for stage in stagesCnt.keys():
        if finalStagePrediction == '':
            finalStagePrediction = stage
        if stagesCnt[stage] > stagesCnt[finalStagePrediction]:
            finalStagePrediction = stage
    # print('    -> finalStagePrediction = ' + finalStagePrediction)
    return finalStagePrediction
'''
