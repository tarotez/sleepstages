import os
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
from deepClassifier import DeepClassifier
# from ksstatistics import StatisticalTester
# from fileManagement import readStandardMice, readdMat, readdTensor

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
class ClassifierClient:
    def __init__(self, recordWaves, extractorType, classifierType, classifierID, inputFileID='', offsetWindowID=0):
        self.recordWaves = recordWaves
        self.inputFileID = inputFileID

        self.params = ParameterSetup()
        self.samplingFreq = self.params.samplingFreq
        self.samplePointNum = self.params.windowSizeInSec * self.samplingFreq  # the number of sample points received at once
        self.graphUpdateFreqInHz = self.params.graphUpdateFreqInHz   # frequency of updating the graph (if set to 1, redraws graph every second)
        assert self.samplingFreq / self.graphUpdateFreqInHz == np.floor(self.samplingFreq / self.graphUpdateFreqInHz)   # should be an integer
        self.updateGraph_samplePointNum = np.int(self.samplingFreq / self.graphUpdateFreqInHz)
        print('self.updateGraph_samplePointNum =', self.updateGraph_samplePointNum)
        self.hasGUI = True
        self.graphColors = ['b','g']
        self.ylim_max_eeg, self.ylim_max_ch2 = 2.0, 2.0
        self.graph_ylim = [[-self.ylim_max_eeg, self.ylim_max_eeg], [-self.ylim_max_ch2, self.ylim_max_ch2]]

        self.lightPeriodStartTime = self.params.lightPeriodStartTime
        self.sampleID = 0
        self.segmentID = offsetWindowID

        # makes a classifier class
        label4EMG = self.params.label4withoutEMG

        self.showCh2 = self.params.showCh2
        self.useCh2ForReplace = self.params.useCh2ForReplace

        self.minimumCh2Intensity = 0
        self.maximumCh2Intensity = 0

        self.past_eegSegment, self.past_ch2Segment = np.array([]), np.array([])
        # self.previous_eeg, self.previous_ch2 = np.array([]), np.array([])

        classifierFilePrefix = self.params.classifierFilePrefix

        factory = AlgorithmFactory(extractorType)
        print('generating extractor: ')
        self.extractor = factory.generateExtractor()

        self.classLabels = list(self.params.labelCorrectionDict.keys())[:self.params.maximumStageNum]

        self.setStagePredictor(classifierID)

        presentTime = timeFormatting.presentTimeEscaped()
        logFileName = 'classifier.' + presentTime + '.csv'
        self.logFile = open(self.params.logDir + '/' + logFileName, 'a')

        # connect to an output device
        self.connected2serialClient = False
        # print('in __init__ of classifierClient, self.connected2serialClient = False')
        self.serialClient, self.connected2serialClient = connect_laser_device()

        self.eeg_till_now = np.zeros((0,))
        '''
        # prepare for computing Kolmogorov-Smirnov test
        standardMice_L, files_L = readStandardMice(self.params)
        self.statisticalTester = StatisticalTester(standardMice_L)
        self.dSeries_L = []
        self.dMat = readdMat(self.params)
        self.dTensor = readdTensor(self.params)
        self.dTensorSegmentNum = self.dTensor.shape[1]
        self.segmentMax4computingKS = self.dTensorSegmentNum * 4
        self.segmentMax4computingKS = self.dTensorSegmentNum
        print('Computes KS (Kolmogorov-Smirnov) till the input reaches segment', self.segmentMax4computingKS)
        '''

        # opens a file for recording waves and prediction results
        if self.inputFileID == '':
            outputFileID = timeFormatting.presentTimeEscaped()
        else:
            outputFileID = self.inputFileID
        waveFileName = outputFileID + '_wave.csv'
        # ksFileName = outputFileID + '_ks.csv'

        self.ch2_mode = "Video"
        self.ch2_thresh_value = self.params.ch2_thresh_default
        self.eeg_normalize_for_prediction = 1
        self.ch2_normalize_for_prediction = 0
        self.eeg_graph_normalize = 0
        self.ch2_graph_normalize = 0
        self.currentCh2Intensity = 0

        if self.recordWaves:
            self.waveOutputFile = open(self.params.waveOutputDir + '/' + waveFileName, 'a')
            self.waveOutputFile_standardized = open(self.params.waveOutputDir + '/standardized_' + waveFileName, 'a')

        self.predictionState = 0
        self.one_record = np.zeros((self.samplePointNum, 2))
        self.raw_one_record = np.zeros((self.samplePointNum, 2))
        self.one_record_for_graph = np.zeros((self.samplePointNum, 2))
        self.windowStartTime = ''
        self.y_pred_L = []

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

    def setStagePredictor(self, classifierID):
        paramFileName = 'params.' + classifierID + '.json'
        finalClassifierDir = self.params.finalClassifierDir
        paramsForNetworkStructure = ParameterSetup(paramDir=finalClassifierDir, paramFileName=paramFileName)
        classifier = DeepClassifier(self.classLabels, classifierID=classifierID, paramsForDirectorySetup=self.params, paramsForNetworkStructure=paramsForNetworkStructure)
        model_path = finalClassifierDir + '/weights.' + classifierID + '.pkl'
        print('model_path = ', model_path)
        classifier.load_weights(model_path)
        self.stagePredictor = StagePredictor(paramsForNetworkStructure, self.extractor, classifier, finalClassifierDir, classifierID, self.params.markovOrderForPrediction)

    def normalize_eeg(self, eegFragment, ch2Fragment, past_eegSegment, past_ch2Segment):
        one_record_partial = np.zeros((self.updateGraph_samplePointNum, 2))
        raw_one_record_partial = np.zeros((self.updateGraph_samplePointNum, 2))
        if self.eeg_normalize_for_prediction:
            processed_eegFragment, _ = standardize(eegFragment, past_eegSegment)
        else:
            # processed_eegFragment, _ = centralize(eegFragment, past_eegSegment)
            processed_eegFragment = eegFragment
        one_record_partial[:,0] = processed_eegFragment
        raw_one_record_partial[:,0] = eegFragment
        if self.ch2_normalize_for_prediction:
            processed_ch2Fragment, _ = standardize(ch2Fragment, past_ch2Segment)
        else:
            processed_ch2Fragment = ch2Fragment
        if self.showCh2 or self.useCh2ForReplace:
            one_record_partial[:,1] = processed_ch2Fragment
            raw_one_record_partial[:,1] = ch2Fragment
        return one_record_partial, raw_one_record_partial

    def process(self, dataFromDaq):
        # print('in client, dataToClient.shape =', dataToClient.shape)
        # print('in client, dataFromDaq =', dataFromDaq)
        timeStampSegment = [_ for _ in range(self.updateGraph_samplePointNum)]
        eegFragment = np.zeros((self.updateGraph_samplePointNum))
        ch2Fragment = np.zeros((self.updateGraph_samplePointNum))

        timeNow = str(datetime.datetime.now())
        self.logFile.write('timeNow = ' + timeNow + ', len(dataFromDaq) = ' + str(len(dataFromDaq)) + ', R->W thresh = ' + str(self.ch2_thresh_value) + ', self.currentCh2Intensity = ' + str(self.currentCh2Intensity) + '\n')
        self.logFile.flush()

        for sampleCnt, inputLine in enumerate(dataFromDaq.split('\n')):
            if not inputLine:
                continue

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

            input_elems = inputLine.split()
            timeStampSegment[sampleCnt] = input_elems[0]
            eegFragment[sampleCnt] = float(input_elems[1])
            if len(input_elems) > 2:
                ch2Fragment[sampleCnt] = float(input_elems[2])

        if self.sampleID == 0:
            self.windowStartTime = timeStampSegment[0]

        # print('eegFragment =', eegFragment)

        one_record_partial, raw_one_record_partial = self.normalize_eeg(eegFragment, ch2Fragment, self.past_eegSegment, self.past_ch2Segment)
        self.one_record[self.sampleID:(self.sampleID+self.updateGraph_samplePointNum),:] = one_record_partial
        self.raw_one_record[self.sampleID:(self.sampleID+self.updateGraph_samplePointNum),:] = raw_one_record_partial
        one_record_for_graph_partial = self.normalize_one_record_partial_for_graph(raw_one_record_partial, self.past_eegSegment, self.past_ch2Segment)
        self.one_record_for_graph[self.sampleID:(self.sampleID+self.updateGraph_samplePointNum),:] = one_record_for_graph_partial
        # print('one_record_partial =', one_record_partial)
        # print('self.sampleID =', self.sampleID)
        # print('self.updateGraph_samplePointNum =', self.updateGraph_samplePointNum)
        # print('eegFragment.shape =', eegFragment.shape)
        # print('one_record_partial.shape =', one_record_partial.shape)
        # print('self.one_record.shape =', self.one_record.shape)
        # print('self.one_record[:32,0] =', self.one_record[:32,0])
        # if self.sampleID > 32:
        #    exit()
        if self.hasGUI:
            self.updateGraphPartially(self.one_record_for_graph)
        self.sampleID += self.updateGraph_samplePointNum

        if self.sampleID == self.samplePointNum:
            self.sampleID = 0
            # copy to previous
            eegSegment = self.raw_one_record[:,0]
            self.past_eegSegment = np.r_[self.past_eegSegment, eegSegment]
            # self.previous_eeg = eegSegment
            if self.showCh2 or self.useCh2ForReplace:
                ch2Segment = self.raw_one_record[:,1]
                self.past_ch2Segment = np.r_[self.past_ch2Segment, ch2Segment]
                # self.previous_ch2 = ch2Segment

            # print('self.predictionState =', self.predictionState)
            replaced = False
            if self.predictionState:
                # stageEstimate is one of ['w', 'n', 'r']
                if self.connected2serialClient:
                    serialClient.write(b'c')
                    print('clear sent to serialClient to reset')

                stagePrediction = self.stagePredictor.predict(eegSegment, timeStampSegment, self.params.stageLabels4evaluation, self.params.stageLabel2stageID)

                # print('stagePrediction =', stagePrediction)
                stagePrediction_before_overwrite = stagePrediction
                if self.useCh2ForReplace:
                    stagePrediction, replaced = self.replaceToWake(stagePrediction, ch2Segment)

            else:
                stagePrediction = '?'

            # update prediction results in graphs by moving all graphs one window
            if self.hasGUI:
                self.updateGraph(self.segmentID, stagePrediction, stagePrediction_before_overwrite, replaced)

            # write out to file
            if self.predictionState:
                #----
                # if the prediction is P, then use the previous one
                if stagePrediction == 'P':
                    # print('stagePrediction == P for wID = ' + str(wID))
                    if len(self.y_pred_L) > 0:
                        finalClassifierDirPrediction = self.y_pred_L[len(self.y_pred_L)-1]
                        # print('stagePrediction replaced to ' + stagePrediction + ' at ' + str(segmentID))
                    else:
                        stagePrediction = 'M'

                # print('pred = ', stagePrediction)
                self.writeToPredFile(stagePrediction, stagePrediction_before_overwrite, timeStampSegment)
                self.y_pred_L.append(stagePrediction)

                #------------------------------------------
                # writes to waveOutputFile
                if self.recordWaves:
                    # records raw data without standardization

                    eegOutputLimitNum = eegSegment.shape[0]
                    # below is for testing, print out only first 5 amplitudes
                    # eegOutputLimitNum = 5
                    elems = self.windowStartTime.split(':')
                    windowStartSecFloat = float(elems[-1])
                    outLine = ''
                    outLine_standardized = ''
                    for i in range(eegOutputLimitNum):
                        secFloat = windowStartSecFloat + (i / self.samplingFreq)
                        timePoint = elems[0] + ':' + elems[1] + ':' + str(secFloat)
                        outLine += str(timePoint) + ', ' + str(self.raw_one_record[i,0]) + ', ' + str(self.raw_one_record[i,1]) + '\n'
                        outLine_standardized += str(timePoint) + ', ' + str(self.one_record[i,0]) + ', ' + str(self.one_record[i,1]) + '\n'

                    self.waveOutputFile.write(outLine)   # add at the end of the file
                    self.waveOutputFile_standardized.write(outLine_standardized)   # add at the end of the file
                    self.waveOutputFile.flush()
                    self.waveOutputFile_standardized.flush()

                #------------------------------------------
                # Encode to binary for serial connection.
                # print('stagePrediction =', stagePrediction)
                if self.connected2serialClient:
                    serialClient = self.serialClient
                    print('in classifierClient.process(), serialClient = self.serialClient')
                    stagePrediction_replaced = 'w' if stagePrediction == '?' else stagePrediction
                    # print(' -> sending', stagePrediction_replaced, 'to serialClient')
                    serialClient.write(stagePrediction_replaced.encode('utf-8'))

            self.one_record = np.zeros((self.samplePointNum, 2))
            self.raw_one_record = np.zeros((self.samplePointNum, 2))
            self.one_record_for_graph = np.zeros((self.samplePointNum, 2))
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
        self.updateMinimumAndMaximumCh2Intensity(self.currentCh2Intensity)
        # print('self.currentCh2Intensity =', self.currentCh2Intensity)
        # print('self.ch2_thresh_value =', self.ch2_thresh_value)
        replaced = False
        if self.currentCh2Intensity > self.ch2_thresh_value:
            print('prediction', prediction, 'replaced to w because ch2intensity =', '{:1.3f}'.format(self.currentCh2Intensity), '>', self.ch2_thresh_value, '= ch2thresh')
            prediction = 'w'
            replaced = True
        return prediction, replaced

    def updateMinimumAndMaximumCh2Intensity(self, currentCh2Intensity):
        if self.minimumCh2Intensity > currentCh2Intensity:
            self.minimumCh2Intensity = currentCh2Intensity
        if self.maximumCh2Intensity < currentCh2Intensity:
            self.maximumCh2Intensity = currentCh2Intensity

    def getCh2Intensity(self, signal):
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

    def update_ylim(self, targetChan):
        if hasattr(self, 'listOfGraphs'):
            for graphID in range(len(self.listOfGraphs[0])):
                for targetChan in range(2):
                    self.listOfGraphs[targetChan][graphID].setData(self.listOfGraphs[targetChan][graphID].getData(), color=self.graphColors[targetChan], graph_ylim=self.graph_ylim[targetChan])

    def normalize_one_record_partial_for_graph(self, raw_one_record_partial, past_eegSegment, past_ch2Segment):
        graph_one_record_partial = np.zeros((self.updateGraph_samplePointNum, 2))
        if self.eeg_graph_normalize:
            graph_one_record_partial[:,0], _ = standardize(raw_one_record_partial[:,0], past_eegSegment)
        else:
            graph_one_record_partial[:,0] = raw_one_record_partial[:,0]
        if self.ch2_graph_normalize:
            graph_one_record_partial[:,1], _ = standardize(raw_one_record_partial[:,1], past_ch2Segment)
        else:
            graph_one_record_partial[:,1] = raw_one_record_partial[:,1]
        return graph_one_record_partial

    def updateGraphPartially(self, one_record_for_graph):
        for targetChan in range(2):
            self.listOfGraphs[targetChan][-1].setData(one_record_for_graph[:,targetChan], color=self.graphColors[targetChan], graph_ylim=self.graph_ylim[targetChan])

    def updateGraph(self, segmentID, stagePrediction, stagePrediction_before_overwrite, replaced):
        choice = self.params.capitalize_for_display[stagePrediction]
        choice_before_overwrite = self.params.capitalize_for_display[stagePrediction_before_overwrite]
        # if choice != choice_before_overwrite:
        if replaced:
            choiceLabel = choice_before_overwrite + '->' + choice
        else:
            choiceLabel = choice
        self.listOfPredictionResults[-1].setChoice(segmentID, choice, choiceLabel)
        for graphID in range(len(self.listOfGraphs[0])-1):
            for targetChan in range(2):
                self.listOfGraphs[targetChan][graphID].setData(self.listOfGraphs[targetChan][graphID+1].getData(), color=self.graphColors[targetChan], graph_ylim=self.graph_ylim[targetChan])
                self.listOfPredictionResults[graphID].setLabel(self.listOfPredictionResults[graphID+1].getLabel(), self.listOfPredictionResults[graphID+1].getStageCode())
        self.listOfPredictionResults[-1].setLabel(str(segmentID + 2) + ' : ', 0)

    def setGraph(self, listOfGraphs):
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

    def setPredictionResult(self, listOfPredictionResults):
        self.listOfPredictionResults = listOfPredictionResults

    def filter(self, data, cutoff=8, fs=128, order=6):
        filtered = butter_lowpass_filter(data, cutoff, fs, order)
        return filtered
