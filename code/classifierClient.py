import math
import numpy as np
import string
import datetime
from functools import reduce
from os.path import dirname, abspath
from statistics import standardize, centralize, subtractLinearFit
from connect_laser_device import connect_laser_device
from parameterSetup import ParameterSetup
from stagePredictor import StagePredictor
import timeFormatting
import time
import pickle
from filters import butter_lowpass_filter
from algorithmFactory import AlgorithmFactory
# from ksstatistics import StatisticalTester
# from fileManagement import readStandardMice, readdMat, readdTensor
from deepClassifier import DeepClassifier
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
class ClassifierClient:
    def __init__(self, recordWaves, extractorType, classifierType, classifierID, inputFileID='', offsetWindowID=0):
        self.recordWaves = recordWaves
        self.inputFileID = inputFileID

        self.params = ParameterSetup()
        self.samplingFreq = self.params.samplingFreq
        # strideInSec = 1
        # lookBackTimeWindowNum = 4
        self.updateFreqInHz = 10
        self.samplePointNum = self.params.windowSizeInSec * self.samplingFreq  # the number of sample points received at once
        self.updateGraph_samplePointNum = np.int(np.floor(self.samplingFreq / self.updateFreqInHz))
        print('self.samplingFreq =', self.samplingFreq)
        print('self.updateGraph_samplePointNum =', self.updateGraph_samplePointNum)
        # self.strideSamplePointNum = strideInSec * self.samplingFreq
        # self.lookBackSamplePointNum = lookBackTimeWindowNum * self.strideSamplePointNum
        self.hasGUI = True
        self.graphColors = ['b','g']
        ### self.graphColorsForKS = ['b','g','r','m','c','y','b']
        self.ylim_max = 2.0
        self.graph_ylim = [[-self.ylim_max, self.ylim_max], [-self.ylim_max, self.ylim_max]]

        self.lightPeriodStartTime = self.params.lightPeriodStartTime
        self.segmentID = offsetWindowID

        # makes a classifier class
        label4EMG = self.params.label4withoutEMG
        # self.params.classifierName = self.params.classifierPrefix + '.' + label4EMG
        # print('self.params.useEMG = ' + str(self.params.useEMG))

        self.minimumCh2Intensity = 0
        self.maximumCh2Intensity = 0

        self.past_eeg, self.past_ch2 = np.array([]), np.array([])
        self.previous_eeg, self.previous_ch2 = np.array([]), np.array([])

        # extractorType = self.params.extractorType
        # classifierType = self.params.classifierType

        classifierFilePrefix = self.params.classifierFilePrefix
        # markovOrderForPrediction = self.params.markovOrderForPrediction

        factory = AlgorithmFactory(extractorType)
        print('generating extractor: ')
        self.extractor = factory.generateExtractor()

        # self.params.classifierName = classifierFilePrefix + '.' + classifierType +  '.' + label4EMG
        # for reading data
        # finalClassifierDir = self.params.finalClassifierDir
        # pwd = dirname(abspath(__file__))
        # classifierFileName = pwd + '/' + finalClassifierDir + '/' + self.params.classifierName + '.pkl'
        # classifierFileHandler = open(classifierFileName, 'rb')
        ### classifierFileHandler = torch.load(open(classifierFileName, 'rb'), map_location='cpu')
        # print('classifierFileName = ' + classifierFileName)
        ### classifier = pickle.load(classifierFileHandler)
        self.classLabels = list(self.params.labelCorrectionDict.keys())[:self.params.maximumStageNum]

        self.setStagePredictor(classifierID)
        # K.clear_session()
        # print('finished to load a model')

        presentTime = timeFormatting.presentTimeEscaped()
        logFileName = 'classifier.' + presentTime + '.csv'
        self.logFile = open(self.params.logDir + '/' + logFileName, 'a')

        # connect to an output device
        self.connected2serialClient = False
        # print('in __init__ of classifierClient, self.connected2serialClient = False')
        self.serialClient, self.connected2serialClient = connect_laser_device()

        # prepare for computing Kolmogorov-Smirnov test
        self.eeg_till_now = np.zeros((0,))
        # standardMice_L, files_L = readStandardMice(self.params)
        # self.statisticalTester = StatisticalTester(standardMice_L)
        # self.dSeries_L = []
        # self.dMat = readdMat(self.params)
        # self.dTensor = readdTensor(self.params)
        # self.dTensorSegmentNum = self.dTensor.shape[1]
        # self.segmentMax4computingKS = self.dTensorSegmentNum * 4
        # self.segmentMax4computingKS = self.dTensorSegmentNum
        # print('Computes KS (Kolmogorov-Smirnov) till the input reaches segment', self.segmentMax4computingKS)

        # opens a file for recording waves and prediction results
        if self.inputFileID == '':
            outputFileID = timeFormatting.presentTimeEscaped()
        else:
            outputFileID = self.inputFileID
        waveFileName = outputFileID + '_wave.csv'
        # ksFileName = outputFileID + '_ks.csv'

        self.ch2_mode = "Video"
        self.ch2_thresh_value = self.params.ch2_thresh_default
        self.eeg_normalize = 1
        self.ch2_normalize = 1
        self.currentCh2Intensity = 0

        if self.recordWaves:
            self.waveOutputFile = open(self.params.waveOutputDir + '/' + waveFileName, 'a')
            self.waveOutputFile_standardized = open(self.params.waveOutputDir + '/standardized_' + waveFileName, 'a')

        self.predictionState = 0

        '''
        try:
            self.ksOutputFile = open(self.params.ksDir + '/' + ksFileName, 'a')
            outLine = 'segmentID, d, chi^2\n'
            self.writeKS2file(outLine)
            # self.windowStartTime = presentTime
        except EnvironmentError:
            pass
        '''

        # makes a file in params.predDir
        if self.inputFileID == '':
            self.predFileID = timeFormatting.presentTimeEscaped()
        else:
            self.predFileID = self.inputFileID
        print('writes prediction results to ' + self.params.predDir + '/' + self.predFileID + '_pred.txt')
        self.predFile = open(self.params.predDir + '/' + self.predFileID + '_pred.txt', 'w')
        self.predFileBeforeOverwrite = open(self.params.predDir + '/' + self.predFileID + '_pred_before_overwrite.txt', 'w')
        self.predFileWithTimeStamps = open(self.params.predDir + '/' + self.predFileID + '_pred_with_timestamps.txt', 'w')

        # self.ch2_mean = self.params.ch2_mean_init
        # self.ch2_variance = self.params.ch2_variance_init
        # self.ch2_oldTotalSampleNum = self.params.ch2_oldTotalSampleNum_init

        '''
        if self.params.useEMG:
            # attributeNames = 'time, predicted class, totalSampleNum, eeg_mean, eeg_variance, ch2_mean, ch2_variance, eeg and ch2 (not standardized), '
            attributeNames = 'time, predicted class, totalSampleNum, eeg_mean, eeg_variance, ch2_mean, ch2_variance'
        else:
            # attributeNames = 'time, predicted class, totalSampleNum, eeg_mean, eeg_variance, eeg and ch2 (not standardized), '
            attributeNames = 'time, predicted class, totalSampleNum, eeg_mean, eeg_variance'
        self.waveOutputFile.write(attributeNames + '\n')   # add at the end of the file
        self.waveOutputFile.flush()
        '''

    def setStagePredictor(self, classifierID):
        paramFileName = 'params.' + classifierID + '.json'
        finalClassifierDir = self.params.finalClassifierDir
        paramsForNetworkStructure = ParameterSetup(paramDir=finalClassifierDir, paramFileName=paramFileName)
        classifier = DeepClassifier(self.classLabels, classifierID=classifierID, paramsForDirectorySetup=self.params, paramsForNetworkStructure=paramsForNetworkStructure)
        # classifier = DeepClassifier(classLabels, excludedFileID='', classifierID='', paramsForDirectorySetup=self.params, paramsForNetworkStructure=self.params)
        ### model_path = pwd + '/' + finalClassifierDir + '/weights.' + classifierID + '.pkl'
        model_path = finalClassifierDir + '/weights.' + classifierID + '.pkl'
        print('model_path = ', model_path)
        classifier.load_weights(model_path)
        # model_path = finalClassifierDir + '/model.pkl'
        # classifier.load_weights(model_path)
        ### self.stagePredictor = StagePredictor(self.params, extractor, classifier, excludedFileID=inputFileID)
        ### self.stagePredictor = StagePredictor(self.params, extractor, classifier, finalClassifierDir, classifierID, markovOrderForPrediction)
        self.stagePredictor = StagePredictor(paramsForNetworkStructure, self.extractor, classifier, finalClassifierDir, classifierID, self.params.markovOrderForPrediction)

    def process(self, dataFromDaq):
        # print('@@@ classifierClient.process() started')
        # print('dataFromDaq = ')
        # print(dataFromDaq)
        # print('@@@')
        if self.connected2serialClient:
            serialClient = self.serialClient
            print('in classifierClient.process(), serialClient = self.serialClient')

        # reset eeg and ch2 statistical values
        # eeg_mean = 0
        # eeg_variance = 0
        # eeg_oldTotalSampleNum = 0
        # ch2_mean = 0
        # ch2_variance= 0

        y_pred_L = []
        sampleID = 0
        timeStampSegment = [_ for _ in range(self.samplePointNum)]
        eegSegment = np.zeros((self.samplePointNum))
        ch2Segment = np.zeros((self.samplePointNum))
        eegPartlyRevisedSegment = np.zeros((self.samplePointNum))
        # timeStampSegment = np.zeros((self.samplePointNum), dtype=str)
        # timeStampSegment_L = []
        ## if self.params.useEMG:
        one_record = np.zeros((self.samplePointNum, 2))
        previous_one_record = np.zeros((self.samplePointNum, 2))
        ch2Segment = np.zeros((self.samplePointNum))
        self.channelNum = 2
        ### else:
            ### one_record = np.zeros((self.samplePointNum, 1))
            ### previous_one_record = np.zeros((self.samplePointNum, 1))
            ### self.channelNum = 1

        timeNow = str(datetime.datetime.now())
        self.logFile.write('timeNow = ' + timeNow + ', len(dataFromDaq) = ' + str(len(dataFromDaq)) + ', R->W thresh = ' + str(self.ch2_thresh_value) + ', self.currentCh2Intensity = ' + str(self.currentCh2Intensity) + '\n')
        self.logFile.flush()

        for inputLine in dataFromDaq.split('\n'):
            if not inputLine:
                continue

            # print('inputLine = ' + inputLine)
            '''
            if inputLine.count('\t') < self.channelNum:
                self.logFile.write('in inputLine, not enough tabs: ' + inputLine + '\n')
                self.logFile.flush()
                continue
            '''

            if inputLine.startswith('\t'):
                self.logFile.write('inputLine starts with a tab: ' + inputLine + '\n')
                self.logFile.flush()
                continue

            if inputLine.endswith('\t'):
                self.logFile.write('inputLine ends with a tab: ' + inputLine + '\n')
                self.logFile.flush()
                continue

            if inputLine.endswith('-'):
                self.logFile.write('inputLine ends with a minus: ' + inputLine + '\n')
                self.logFile.flush()
                continue

            # for testing
            # self.logFile.write(inputLine + '\n')
            # self.logFile.flush()

            # for testing
            # print('inputLine = ' + inputLine)
            # print('useEMG = ' + str(self.params.useEMG))

            input_elems = inputLine.split()
            timeStampSegment[sampleID] = input_elems[0]
            eegSegment[sampleID] = float(input_elems[1])
            if len(input_elems) > 2:
                ch2Segment[sampleID] = float(input_elems[2])

            # if self.params.useEMG:
            #    timeStamp, eeg_value, ch2_value = inputLine.split()
            #    ch2Segment[sampleID] = float(ch2_value)
            #else:
            #    timeStamp, eeg_value = inputLine.split()
            # eegSegment[sampleID] = float(eeg_value)
            # timeStampSegment_L.append(timeStamp)
            # print('timeStampSegment_L[' + sampleID + '] =' + str(timeStampSegment_L[sampleID]))
            if sampleID == 0:
                windowStartTime = timeStampSegment[sampleID]
            # print('timeStamp = ' + str(timeStamp))
            # print('eegSegment[', segmentID, '] = ', eegSegment[segmentID])
            # print('ch2Segment[', segmentID, '] = ', ch2Segment[segmentID])

            # print('sampleID = ' + str(sampleID) + ', self.samplePointNum = ' + str(self.samplePointNum))
            sampleID += 1
            if sampleID % self.updateGraph_samplePointNum == 0 or sampleID == self.samplePointNum:
                # standardize eeg and ch2
                if self.eeg_normalize:
                    processed_eegSegment, self.past_eeg = standardize(eegSegment, self.past_eeg)
                else:
                    # processed_eegSegment, _ = subtractLinearFit(eegSegment, self.previous_eeg, sampleID)
                    processed_eegSegment, self.past_eeg = centralize(eegSegment, self.past_eeg)
                one_record[:,0] = processed_eegSegment
                if self.ch2_normalize:
                # processed_ch2Segment, self.past_ch2 = standardize(ch2Segment, self.past_ch2)
                    processed_ch2Segment, self.past_ch2 = standardize(ch2Segment, self.past_ch2)
                else:
                    processed_ch2Segment = ch2Segment
                if self.params.useEMG:
                    one_record[:,1] = processed_ch2Segment

                # draw a graph
                if self.hasGUI:
                    #print('segment ' + str(self.segmentID) + ' : prediction = ' + stagePrediction) #comentout-by-Natsu
                    self.updateGraphPartially(one_record, self.segmentID)
                    # print("eegSegment = " + str(eegSegment))
                    ### time.sleep(0.1)

            if sampleID == self.samplePointNum:
                self.previous_eeg = eegSegment
                self.previous_ch2 = ch2Segment
                sampleID = 0

                '''
                eeg_old_mean = eeg_mean
                eeg_mean = recompMean(eegSegment, eeg_mean, eeg_oldTotalSampleNum)
                eeg_variance = recompVariance(eegSegment, eeg_variance, eeg_old_mean, eeg_mean, eeg_oldTotalSampleNum)
                if eeg_variance == 0:
                    eeg_variance = 0.0000001
                # print('eegSegment = ' + str(eegSegment))
                one_record[:,0] = (eegSegment - eeg_mean) / np.sqrt(eeg_variance)
                eeg_oldTotalSampleNum += eegSegment.shape[0]
                if self.ch2_normalize:
                    ch2_old_mean = self.ch2_mean
                    self.ch2_mean = recompMean(ch2Segment, self.ch2_mean, self.ch2_oldTotalSampleNum)
                    one_record[:,1] = ch2Segment - self.ch2_mean
                    self.ch2_variance = recompVariance(ch2Segment, self.ch2_variance, ch2_old_mean, self.ch2_mean, self.ch2_oldTotalSampleNum)
                    if self.ch2_variance == 0:
                        self.ch2_variance = 0.0000001
                    one_record[:,1] = (ch2Segment - self.ch2_mean) / np.sqrt(self.ch2_variance)
                    self.ch2_oldTotalSampleNum += ch2Segment.shape[0]
                    # print('self.ch2_oldTotalSampleNum =', self.ch2_oldTotalSampleNum)
                else:
                    one_record[:,1] = ch2Segment
                # print('ch2Segment.mean() =', ch2Segment.mean(), ', one_record[:,1].mean() = ', one_record[:,1].mean())
                '''

                # print('self.predictionState =', self.predictionState)
                if self.predictionState:
                    # stageEstimate is one of ['w', 'n', 'r']
                    if self.connected2serialClient:
                        serialClient.write(b'c')
                        print('clear sent to serialClient to reset')

                    # classifier predicts and returns either w, n, or r.
                    # print('segmentID = ' + str(segmentID))
                    # print('one_record[:,0] = ' + str(one_record[:,0]))
                    # local_mu = np.mean(eegSegment)
                    # local_sigma = np.std(eegSegment)
                    # print('np.mean(one_record[:,0]) = ' + str(np.mean(one_record[:,0])))
                    # print('np.std(one_record[:,0]) = ' + str(np.std(one_record[:,0])))
                    # timeStampSegment = np.array(timeStampSegment_L)
                    # print('timeStampSegment = ' + timeStampSegment[0])
                    '''
                    # voting system. obsolete.
                    if np.sum(previous_one_record) > 0:
                        stagePredictions_L = []
                        # print('one_record.shape = ' + str(one_record.shape) + ', previous_one_record.shape = ' + str(previous_one_record.shape))
                        joined_record = np.r_[previous_one_record, one_record]
                        # print('joined_record.shape = ' + str(joined_record.shape))
                        for offset in range(self.samplePointNum - self.lookBackSamplePointNum, self.samplePointNum+1, self.strideSamplePointNum):
                            slided_record = joined_record[offset:(offset+self.samplePointNum),:]
                            # print('offset = ' + str(offset) + ', slided_record.shape = ' + str(slided_record.shape))
                            stagePrediction = self.stagePredictor.predict(slided_record, timeStampSegment, local_mu, local_sigma)
                            stagePredictions_L.append(stagePrediction)
                        stagePrediction = voteForStagePrediction(stagePredictions_L)
                    else:
                        # print('one_record.shape = ' + str(one_record.shape))
                    '''
                    one_record_ch1 = one_record[:,0]
                    stagePrediction = self.stagePredictor.predict(one_record_ch1, timeStampSegment, self.params.stageLabels4evaluation, self.params.stageLabel2stageID)

                    # print('stagePrediction =', stagePrediction)
                    stagePrediction_before_overwrite = stagePrediction
                    if self.params.useCh2ForReplace:
                        one_record_ch2 = one_record[:,1]
                        stagePrediction = self.replaceToWake(stagePrediction, one_record_ch2)

                    previous_one_record = one_record

                    #----
                    # if the prediction is P, then use the previous one
                    if stagePrediction == 'P':
                        # print('stagePrediction == P for wID = ' + str(wID))
                        if len(y_pred_L) > 0:
                            finalClassifierDirPrediction = y_pred_L[len(y_pred_L)-1]
                            # print('stagePrediction replaced to ' + stagePrediction + ' at ' + str(segmentID))
                        else:
                            stagePrediction = 'M'

                    # print('pred = ', stagePrediction)
                    self.writeToPredFile(stagePrediction, stagePrediction_before_overwrite, timeStampSegment)

                    y_pred_L.append(stagePrediction)

                    # true label
                    ### trueLabel = stageSeq[wID]

                    # record one_record and stagePrediction
                    '''
                    outLine = windowStartTime + ',' + stagePrediction
                    outLine += ',' + str(oldTotalSampleNum) + ',' + '{0:.4f}'.format(eeg_mean) + ',' + '{0:.4f}'.format(eeg_variance)
                    if self.params.useEMG:
                        outLine += ',' + '{0:.4f}'.format(ch2_mean) + ',' + '{0:.4f}'.format(ch2_variance)
                    outLine += '\n'
                    '''

                    #------------------------------------------
                    # writes to waveOutputFile
                    if self.recordWaves:
                        # records raw data without standardization
                        eegOutputLimitNum = eegSegment.shape[0]
                        # below is for testing, print out only first 5 amplitudes
                        # eegOutputLimitNum = 5
                        elems = windowStartTime.split(':')
                        windowStartSecFloat = float(elems[-1])
                        outLine = ''
                        outLine_standardized = ''
                        for i in range(eegOutputLimitNum):
                            secFloat = windowStartSecFloat + (i / self.samplingFreq)
                            timePoint = elems[0] + ':' + elems[1] + ':' + str(secFloat)
                            ### outLine += ',' + str(eegSegment[i])
                            outLine += str(timePoint) + ', ' + str(eegSegment[i]) + ', ' + str(ch2Segment[i]) + '\n'
                            outLine_standardized += str(timePoint) + ', ' + str(processed_eegSegment[i]) + ', ' + str(processed_ch2Segment[i]) + '\n'
                            # outLine_standardized += str(timePoint) + ', ' + str(centralized_eegSegment[i]) + ', ' + str(processed_ch2Segment[i]) + '\n'

                        self.waveOutputFile.write(outLine)   # add at the end of the file
                        self.waveOutputFile_standardized.write(outLine_standardized)   # add at the end of the file
                        self.waveOutputFile.flush()
                        self.waveOutputFile_standardized.flush()
                        # print('file output = ' + outLine)

                    #------------------------------------------

                    # シリアル通信ではバイナリ列でやりとりする必要があるので
                    # バイナリ形式にエンコードする
                    # print('stagePrediction =', stagePrediction)
                    if self.connected2serialClient:
                        #serialClient.write(b'c')
                        # print('stagePrediction =', stagePrediction)
                        stagePrediction_replaced = 'w' if stagePrediction == '?' else stagePrediction
                        # print(' -> sending', stagePrediction_replaced, 'to serialClient')
                        serialClient.write(stagePrediction_replaced.encode('utf-8'))
                else:
                    stagePrediction = '?'

                # draw a graph
                if self.hasGUI:
                    #print('segment ' + str(self.segmentID) + ' : prediction = ' + stagePrediction) #comentout-by-Natsu
                    self.updateGraph(one_record, self.segmentID, stagePrediction, stagePrediction_before_overwrite)
                    # print("eegSegment = " + str(eegSegment))
                    ##### time.sleep(0.01)
                else:
                    pass
                    #print('segment ' + str(self.segmentID) + ' : prediction = ' + stagePrediction) #cometouy-by-Natsu

                # update KS (Kolmogorov-Smirnov test)
                '''
                if self.params.computeKS:
                    if len(one_record.shape) > 1:
                        eeg4KS = one_record[:,0]
                    else:
                        eeg4KS = one_record
                    self.updateKS(eeg4KS, self.segmentID)
                '''
                # reset eegSegment and ch2Segment
                # eegSegment = np.zeros((self.samplePointNum))
                ### if self.params.useEMG:
                # ch2Segment = np.zeros((self.samplePointNum))
                self.segmentID += 1

    def writeToPredFile(self, prediction, prediction_before_overwrite, timeStampSegment):
        prediction_in_capital = self.params.capitalize_for_writing_prediction_to_file[prediction]
        self.predFile.write(prediction_in_capital + '\n')   # add at the end of the file
        self.predFile.flush()
        prediction_before_overwrite_in_capital = self.params.capitalize_for_writing_prediction_to_file[prediction_before_overwrite]
        self.predFileBeforeOverwrite.write(prediction_before_overwrite_in_capital + '\n')
        self.predFileBeforeOverwrite.flush()
        self.predFileWithTimeStamps.write(prediction_in_capital + ',' + timeStampSegment[0] + '\n')   # add at the end of the file
        self.predFileWithTimeStamps.flush()

    def replaceToWake(self, prediction, signal):
        self.currentCh2Intensity = self.getCh2Intensity(signal)
        # print('self.currentCh2Intensity =', '{0:.4f}'.format(self.currentCh2Intensity), ', thresh =', self.ch2_thresh_value, ', min =', '{0:.4f}'.format(self.minimumCh2Intensity), ', max =', '{0:.4f}'.format(self.maximumCh2Intensity))
        if self.minimumCh2Intensity == 0:
            self.minimumCh2Intensity = self.currentCh2Intensity
        elif self.maximumCh2Intensity == 0:
            self.maximumCh2Intensity = self.currentCh2Intensity
        ### if self.currentCh2Intensity > self.minimumCh2Intensity * self.params.replaceToWakeThreshFactor and self.currentCh2Intensity > self.maximumCh2Intensity / self.params.replaceToWakeThreshFactor:
        self.updateMinimumAndMaximumCh2Intensity(self.currentCh2Intensity)
        if self.currentCh2Intensity > self.ch2_thresh_value:
            print('prediction', prediction, 'replaced to w because ch2intensity =', '{:1.3f}'.format(self.currentCh2Intensity), '>', self.ch2_thresh_value, '= ch2thresh')
            prediction = 'w'
        return prediction

    def updateMinimumAndMaximumCh2Intensity(self, currentCh2Intensity):
        if self.minimumCh2Intensity > currentCh2Intensity:
            self.minimumCh2Intensity = currentCh2Intensity
        if self.maximumCh2Intensity < currentCh2Intensity:
            self.maximumCh2Intensity = currentCh2Intensity

    def getCh2Intensity(self, signal):
        # currentCh2Intensity = np.max(signal)
        # smoothed = (np.array(signal[:-1]) + np.array(signal[1:])) / 2
        # currentCh2Intensity = smoothed.max()
        if self.params.ch2IntensityFunc == 'max_mean':
            segNum = 80
            segLen = len(signal) // segNum
            segments = []
            for segID in range(segNum):
                startID, endID = segID * segLen, (segID + 1) * segLen
                segments.append(signal[startID:endID])
            currentCh2Intensity = np.max([np.array(segment).mean() for segment in segments])
        elif self.params.ch2IntensityFunc == 'max_std':
            segNum = 80
            segLen = len(signal) // segNum
            segments = []
            for segID in range(segNum):
                startID, endID = segID * segLen, (segID + 1) * segLen
                segments.append(signal[startID:endID])
            currentCh2Intensity = np.max([np.array(segment).std() for segment in segments])
        elif self.params.ch2IntensityFunc == 'mean':
            currentCh2Intensity = signal.mean()
        elif self.params.ch2IntensityFunc == 'max':
            currentCh2Intensity = signal.max()
        else:
            currentCh2Intensity = signal.std()
        return currentCh2Intensity

    '''
    def updateKS(self, eeg, segmentID):
        if self.hasGUI:
            if segmentID < self.segmentMax4computingKS:
                # print('self.eeg_till_now.shape = ' + str(self.eeg_till_now.shape) + ', eeg.shape = ' + str(eeg.shape))
                self.eeg_till_now = np.concatenate((self.eeg_till_now, eeg))
                d, pval, chi2 = self.statisticalTester.ks_test(self.eeg_till_now)
                self.dSeries_L.append(d)
                # print('d = ' + "{:.5f}".format(d) + ', chi2 = ' + "{:.2f}".format(chi2))
                # print("{:.5f}".format(d))
                self.chi2ResLabel.setText("{:.3f}".format(np.mean(chi2)))
                self.dResLabel.setText("{:.5f}".format(np.mean(d)))
                dSeriesArray = np.array(self.dSeries_L)
                self.dGraph.setData4KS(dSeriesArray, self.graphColorsForKS)
                if segmentID < self.dTensorSegmentNum:
                    self.dHist.setData4KSHist(self.dTensor[:,segmentID,:], d)
                else:
                    self.dHist.setData4KSHist(self.dMat, d)
        else:
            if segmentID > 0 and segmentID % 360 == 0:
                self.eeg_till_now = np.concatenate((self.eeg_till_now, eeg))
                d, pval, chi2 = self.statisticalTester.ks_test(self.eeg_till_now)
                outLine = str(segmentID) + ', ' + str(d) + ', ' + str(chi2) + '\n'
                self.writeKS2file(outLine)

    def writeKS2file(self, outLine):
        self.ksOutputFile.write(outLine)   # add at the end of the file
        self.ksOutputFile.flush()
    '''

    def updateGraphPartially(self, one_record, segmentID):
        # eeg = self.filter(one_record[:,0])
        eeg = one_record[:,0]
        self.listOfGraphs[0][-1].setData(eeg, color=self.graphColors[0], graph_ylim=self.graph_ylim[0])
        # ch2 = self.filter(one_record[:,1])
        ch2 = one_record[:,1]
        self.listOfGraphs[1][-1].setData(ch2, color=self.graphColors[1], graph_ylim=self.graph_ylim[1])
        ########

    def updateGraph(self, one_record, segmentID, stagePrediction, stagePrediction_before_overwrite):
        # print('---- starting updateGraph()')
        for graphID in range(len(self.listOfGraphs[0])-1):
            for targetChan in range(2):
                # print('channel = ' + str(targetChan)+ ', graphID = ' + str(graphID) + '/' + str(len(self.listOfGraphs[targetChan])))
                self.listOfGraphs[targetChan][graphID].setData(self.listOfGraphs[targetChan][graphID+1].getData(), color=self.graphColors[targetChan], graph_ylim=self.graph_ylim[targetChan])
                self.listOfPredictionResults[graphID].setLabel(self.listOfPredictionResults[graphID+1].getLabel(), self.listOfPredictionResults[graphID+1].getStageCode())
        # eeg = self.filter(one_record[:,0])
        eeg = one_record[:,0]
        self.listOfGraphs[0][-1].setData(eeg, color=self.graphColors[0], graph_ylim=self.graph_ylim[0])
        # if self.params.useEMG:
            # ch2 = self.filter(one_record[:,1])
            # self.listOfGraphs[1][-1].setData(ch2, color=self.graphColors[1])
        # ch2 = self.filter(one_record[:,1])
        ch2 = one_record[:,1]
        self.listOfGraphs[1][-1].setData(ch2, color=self.graphColors[1], graph_ylim=self.graph_ylim[1])
        choice = self.params.capitalize_for_display[stagePrediction]
        choice_before_overwrite = self.params.capitalize_for_display[stagePrediction_before_overwrite]
        if choice != choice_before_overwrite:
            choiceLabel = choice_before_overwrite + '->' + choice
        else:
            choiceLabel = choice
        self.listOfPredictionResults[-1].setChoice(segmentID, choice, choiceLabel)

    def setGraph(self, listOfGraphs):
        # print('% classifierClient.setGraph()')
        self.listOfGraphs = listOfGraphs

    def setchi2ResLabel(self, chi2ResLabel):
        self.chi2ResLabel = chi2ResLabel

    def setdResLabel(self, dResLabel):
        self.dResLabel = dResLabel

    def setdGraph(self, dGraph):
       self.dGraph = dGraph

    def setdHist(self, dHist):
       self.dHist = dHist

    def predictionStateOn(self):
        self.predictionState = 1
        print('predictionState is set to 1 at classifierClient')
        # if self.channelNumForPrediction > 1:
        #    self.params.useEMG = 1

    def setPredictionResult(self, listOfPredictionResults):
        self.listOfPredictionResults = listOfPredictionResults

    def filter(self, data, cutoff=8, fs=128, order=6):
        filtered = butter_lowpass_filter(data, cutoff, fs, order)
        return filtered
