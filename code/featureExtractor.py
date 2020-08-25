from __future__ import print_function
import pickle
import numpy as np
from emgProcessing import emg2feature

class FeatureExtractor():

    def __init__(self):
        self.extractorType = ''

    def getFeatures(self, eegSegment, timeStampSegment, time_step):
        print('passing getFeatures because FeatureExtractor is an interface.')

    def featureExtraction(self, params, fileID):
        #---------------
        # set up parameters
        # get params shared by programs
        useEMG = params.useEMG
        emgTimeFrameNum = params.emgTimeFrameNum
        targetBand = params.wholeBand
        eegDir = params.eegDir
        featureDir = params.featureDir
        samplingFreq = params.samplingFreq
        windowSizeInSec = params.windowSizeInSec
        binWidth4freqHisto = params.binWidth4freqHisto
        pastStageLookUpNum = params.pastStageLookUpNum
        stageNum = len(params.stageLabel2stageID)
        eegFilePrefix = params.eegFilePrefix
        featureFilePrefix = params.featureFilePrefix
        # print('stageNum = ' + str(stageNum))

        # compute parameters
        time_step = 1 / samplingFreq
        wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
        binNum4spectrum = round(targetBand.getBandWidth() / binWidth4freqHisto)
        binArray4spectrum = np.linspace(targetBand.bottom, targetBand.top, binNum4spectrum + 1)

        #--------
        # load data
        # print('fileID = ' + str(fileID))
        dataFileHandler = open(eegDir + '/' + eegFilePrefix + '.' + fileID + '.pkl', 'rb')
        (eeg, emg, stageSeq, timeStamps) = pickle.load(dataFileHandler)
        # print('eeg.shape = ' + str(eeg.shape))
        # print('len(stageSeq) = ' + str(len(stageSeq)))

        # normalize eeg and emg
        global_mu = np.mean(eeg)
        global_sigma = np.std(eeg)
        ### emg = (emg - np.mean(emg)) / np.std(emg)
        # print('in featureExtractionClassical(), eeg.shape = ' + str(eeg.shape))

        #----------
        # extract time windows from EEG, apply FFT, and bin them
        ### emgfeatureTensor = np.zeros((emgTimeFrameNum,0), dtype=np.float)
        startSamplePoint = 0
        while startSamplePoint + wsizeInTimePoints <= eeg.shape[0]:
            endSamplePoint = startSamplePoint + wsizeInTimePoints
            eegSegment = eeg[startSamplePoint:endSamplePoint]
            ### emgSegment = emg[startSamplePoint:endSamplePoint]
            timeStampSegment = timeStamps[startSamplePoint:endSamplePoint]
            # local_mu = np.mean(eegSegment)
            # local_sigma = np.std(eegSegment)
            eegSegmentStandardized = (eegSegment - global_mu) / global_sigma
            # local_mu = np.mean(eegSegmentStandardized)
            # local_sigma = np.std(eegSegmentStandardized)
            # print('local_mu = ' + str(local_mu) + ', local_sigma = ' + str(local_sigma))

            features = self.getFeatures(eegSegmentStandardized, timeStampSegment, time_step)
            featuresAdditional = np.array([features])
            if startSamplePoint == 0:
                featureTensor = featuresAdditional
            else:
                featureTensor = np.r_[featureTensor, featuresAdditional]
            # print('featureTensor.shape = ' + str(featureTensor.shape) + ', features.shape = ' + str(features.shape))
            startSamplePoint = endSamplePoint
            '''
            # emgfeatureTensor = np.c_[emgfeatureTensor, np.mean(np.abs(emgSegment))]
            emgFeature = emg2feature(emgSegment, emgTimeFrameNum)
            # print('emgFeature = ' + str(emgFeature))
            emgfeatureTensor = np.c_[emgfeatureTensor, emgFeature]
            '''

        if params.useEMG:
            label4EMG = params.label4withEMG
        else:
            label4EMG = params.label4withoutEMG
        #-----
        # get the lengths of sequences
        eLen = featureTensor.shape[1]
        sLen = len(stageSeq)
        # print('  eLen = ' + str(eLen) + ', sLen = ' + str(sLen))
        if eLen != sLen:
            featureTensor = featureTensor[:sLen]

        print('featureTensor.shape = ' + str(featureTensor.shape))

        '''
        pastStagesOneHots = np.zeros((pastStageLookUpNum * stageNum, 0), dtype=np.float)
        for wID in range(0, sLen):
            pastStagesOneHots = np.c_[pastStagesOneHots, constructPastStagesOneHots(stageSeq, wID, pastStageLookUpNum, stageNum)]

        # Below deals with the case that not all of the time windows have been labeled.
        # In such a case, stageSeq is shorter than featureArray
        if eLen != sLen:
            featureTensor = featureTensor[:,:sLen]
            emgfeatureTensor = emgfeatureTensor[:,:sLen]
        # print('featureTensor.shape = ' + str(featureTensor.shape) + ', emgfeatureTensor.shape = ' + str(emgfeatureTensor.shape) + ', pastStagesOneHots.shape = ' + str(pastStagesOneHots.shape))
        if useEMG:
            features_part = np.r_[featureTensor, emgfeatureTensor]
        else:
            features_part = featureTensor
        if pastStageLookUpNum > 0:
            features = np.r_[features_part, pastStagesOneHots]
        else:
            features = features_part
        '''
        featurePath = featureDir + '/' + featureFilePrefix + '.' + self.extractorType + '.' + label4EMG + '.' + fileID + '.pkl'
        print('extracting features for featurePath = ' + featurePath)
        featureFileHandler = open(featurePath, 'wb')
        pickle.dump(featureTensor, featureFileHandler)
