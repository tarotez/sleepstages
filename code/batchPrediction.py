import pickle
import numpy as np
from os import listdir
from os.path import dirname, abspath
from stagePredictor import StagePredictor
from statistics import recompMean, recompVariance
from algorithmFactory import AlgorithmFactory
from parameterSetup import ParameterSetup
from sequentialPrediction import voteForStagePrediction
from dataReader import DataReader

def batchClassifySequentially(eegFilePath, stagePredictor):

    dataReader = DataReader()
    eeg, emg, timeStamps = dataReader.readEEG(eegFilePath)
    print('eeg.shape = ' + str(eeg.shape))
    samplePointNum = eeg.shape[0]

    # reset eeg and emg statistical values
    eeg_mean = 0
    eeg_variance = 0
    emg_mean = 0
    emg_variance= 0
    oldSampleNum = 0

    y_pred_L = []
    wID = 0
    # startSamplePoint = 0
    stagePredictions_L = []
    replacedR = False
    # while startSamplePoint + wsizeInSamplePointNum <= samplePointNum:
    records_L = []
    timeStampSegments_L = []
    for startSamplePoint in range(0, samplePointNum, timeWindowStrideInSamplePointNum):
        endSamplePoint = startSamplePoint + wsizeInSamplePointNum
        if endSamplePoint > samplePointNum:
            break
        eegSegment = eeg[startSamplePoint:endSamplePoint]
        eegSegment_orig_mean = np.mean(eegSegment)
        eegSegment_orig_std = np.std(eegSegment)
        eeg_old_mean = eeg_mean
        eeg_mean = recompMean(eegSegment, eeg_mean, oldSampleNum)
        eeg_variance = recompVariance(eegSegment, eeg_variance, eeg_old_mean, eeg_mean, oldSampleNum)
        standardized_eegSegment = (eegSegment - eeg_mean) / np.sqrt(eeg_variance)
        if params.useEMG:
            emgSegment = emg[startSamplePoint:endSamplePoint]
            emg_old_mean = emg_mean
            emg_mean = recompMean(emgSegment, emg_mean, oldSampleNum)
            emg_variance = recompVariance(emgSegment, emg_variance, emg_old_mean, emg_mean, oldSampleNum)
            standardized_emgSegment = (emgSegment - emg_mean) / np.sqrt(emg_variance)
            one_record = np.r_[standardized_eegSegment, standardized_emgSegment]
        else:
            one_record = standardized_eegSegment
        oldSampleNum += eegSegment.shape[0]
        local_mu = np.mean(standardized_eegSegment)
        local_sigma = np.std(standardized_eegSegment)
        timeStampSegment = timeStamps[startSamplePoint:endSamplePoint]
        # if predict_by_batch:
        records_L.append(one_record)
        timeStampSegments_L.append(timeStampSegment)
        '''
        else:
            stagePrediction = params.reverseLabel(stagePredictor.predict(one_record, timeStampSegment, local_mu=local_mu, local_sigma=local_sigma))
            # print('after stagePredictor.predict')

            # replaces R to W when W appears consecutively
            Ws = 'W' * numOfConsecutiveWsThatProhibitsR
            for wCnt in range(1,numOfConsecutiveWsThatProhibitsR+1):
                if len(y_pred_L) >= wCnt:
                    if y_pred_L[len(y_pred_L)-wCnt] != 'W':
                        break
                if stagePrediction == 'R':
                    print(Ws + '->R changed to ' + Ws + '->W at wID = ' + str(wID) + ', startSamplePoint = ' + str(startSamplePoint))
                    stagePrediction = 'W'
                    replacedR = True

            #----
            # if the prediction is P, then use the previous one
            if stagePrediction == 'P':
                # print('stagePrediction == P for wID = ' + str(wID))
                if len(y_pred_L) > 0:
                    stagePrediction = y_pred_L[len(y_pred_L)-1]
                else:
                    stagePrediction = 'M'

            #-----
            # vote for choosing the label for 10 second window
            stagePredictions_L.append(stagePrediction)
            if len(stagePredictions_L) == strideNumInTimeWindow:
                finalStagePrediction = voteForStagePrediction(stagePredictions_L[-(lookBackTimeWindowNum+1):])
                #-----
                # append to the lists of results
                y_pred_L.append(finalStagePrediction)

    if predict_by_batch:
        '''
    ### y_pred = np.array([params.reverseLabel(y_pred_orig) for y_pred_orig in stagePredictor.batch_predict(np.array(records_L), np.array(timeStampSegments_L), local_mu=local_mu, local_sigma=local_sigma)])
    y_pred = np.array([params.capitalize_for_writing_prediction_to_file[y_pred_orig] for y_pred_orig in stagePredictor.batch_predict(params, np.array(records_L), np.array(timeStampSegments_L), local_mu=local_mu, local_sigma=local_sigma)])
    '''
    else:
        y_pred = np.array(y_pred_L)
        '''
    return y_pred

#--------------------
# main

params = ParameterSetup()
# print('params.finalClassifierDir = ' + str(params.finalClassifierDir))
extractorType = params.extractorType
classifierType = params.classifierType
classifierParams = params.classifierParams
samplingFreq = params.samplingFreq
windowSizeInSec = params.windowSizeInSec
wsizeInSamplePointNum = windowSizeInSec * samplingFreq   # window size in sample points. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
timeWindowStrideInSec = params.timeWindowStrideInSec
timeWindowStrideInSamplePointNum = timeWindowStrideInSec * samplingFreq
lookBackTimeWindowNum = params.lookBackTimeWindowNum
strideNumInTimeWindow = np.ceil(windowSizeInSec / timeWindowStrideInSec)
classifierFilePrefix = params.classifierFilePrefix
# replacesWWWRtoWWWW = params.replacesWWWRtoWWWW
numOfConsecutiveWsThatProhibitsR = params.numOfConsecutiveWsThatProhibitsR

if params.useEMG:
    label4EMG = params.label4withEMG
else:
    label4EMG = params.label4withoutEMG

params.classifierName = classifierFilePrefix + '.' + classifierType +  '.' + label4EMG
factory = AlgorithmFactory(extractorType)
extractor = factory.generateExtractor()

# if params.classifierType == 'deep':
#    classifier = DeepClassifier(classifierID=excludedFileID)
# else:
pwd = dirname(abspath(__file__))
classifierFileName = pwd + '/' + params.finalClassifierDir + '/' + params.classifierName + '.pkl'
classifierFileHandler = open(classifierFileName, 'rb')
print('classifierFileName = ' + classifierFileName)
classifier =  pickle.load(classifierFileHandler)

# stagePredictor = StagePredictor(params, extractor, classifier, excludedFileID=excludedFileID)
stagePredictor = StagePredictor(params, extractor, classifier)

fileNamesInPostDir = listdir(params.postDir)
for fileNameInPostDir in fileNamesInPostDir:
    if not fileNameInPostDir.startswith('.'):
        elems = fileNameInPostDir.split('.')
        fileNameForPredDir = '.'.join(elems[:-1]) + '_pred.' + elems[-1]
        predictionFilePath = params.predDir + '/' + fileNameForPredDir
        predictionFileHandler = open(predictionFilePath, 'w')
        print('started classifying ' + fileNameInPostDir)
        eegFilePath = params.postDir + '/' + fileNameInPostDir
        y_pred = batchClassifySequentially(eegFilePath, stagePredictor)
        print('finished classifying ' + fileNameInPostDir)
        for pred in y_pred:
            predictionFileHandler.write(pred + '\n')
        predictionFileHandler.close()
