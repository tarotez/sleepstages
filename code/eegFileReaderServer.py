import numpy as np
from datetime import datetime, timedelta
import time
from os import listdir
from random import shuffle
from parameterSetup import ParameterSetup
import timeFormatting
from dataReader import DataReader
from sampler import up_or_down_sampling

class EEGFileReaderServer:

    def __init__(self, client, eegFilePath, model_samplingFreq=128, model_epochTime=10, observed_samplingFreq=128, observed_epochTime=10, channelOpt=1):

        self.client = client
        if channelOpt == 1:
            self.useEMG = 0
        if channelOpt == 2:
            self.useEMG = 1

        self.params = ParameterSetup()
        # pickledDir = self.params.pickledDir
        # classifierType = self.params.classifierType
        # classifierParams = self.params.classifierParams
        # samplingFreq = self.params.samplingFreq
        # windowSizeInSec = self.params.windowSizeInSec
        # self.wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
        if observed_samplingFreq == model_samplingFreq:
            self.wsizeInTimePoints = self.client.updateGraph_samplePointNum
        else:
            self.wsizeInTimePoints = model_samplingFreq

        dataReader = DataReader()
        print('for EEG, reading file ' + eegFilePath)
        # print('for stageSeq, reading file ' + stageFilePath)
        # stageSeq = dataReader.readStageSeq(stageFilePath)
        # print('timeStamps[0] =', timeStamps[0])
        # print('self.samplingFreq =', self.samplingFreq)
        # print('observed_epochTime =', observed_epochTime)
        model_samplePointNum = model_samplingFreq * model_epochTime
        observed_samplePointNum = observed_samplingFreq * observed_epochTime
        self.model_samplingFreq = model_samplingFreq
        # print('observed_samplePointNum =', observed_samplePointNum)

        # self.eeg = eeg
        # self.emg = emg
        # self.timeStamps = timeStamps
        eeg, emg, timeStamps = dataReader.readEEG(eegFilePath)
        # print('Starting to resample. Before resampling, eeg.shape =', eeg.shape)
        self.eeg = up_or_down_sampling(eeg, model_samplePointNum, observed_samplePointNum)
        if self.useEMG:
            self.emg = up_or_down_sampling(emg, model_samplePointNum, observed_samplePointNum)
        else:
            self.emg = []
        # print('Finished resampling. After resampling, eeg.shape =', self.eeg.shape)
        # print('before resampling: len(timeStamps) =', len(timeStamps))
        resampledLen = self.eeg.shape[0]
        self.timeStamps = self.convertTimeStamps(timeStamps, resampledLen, model_samplingFreq, model_samplePointNum, observed_samplePointNum)        
        # print('after resampling: len(timeStamps) =', len(self.timeStamps))
        self.eegLen = self.eeg.shape[0]
        # print('eegLen =', self.eegLen)
        presentTime = timeFormatting.presentTimeEscaped()
        fileName = 'daq.' + presentTime + '.csv'
        self.logFile = open(self.params.logDir + '/' + fileName, 'a')
        self.serve()

    def convertTimeStamps(self, timeStamps, resampledLen, model_samplingFreq, model_samplePointNum, observed_samplePointNum):
        # print('timeStamps[0] =', timeStamps[0])
        hour_str, minute_str, second_microsecond_str = timeStamps[0].split(':')
        second = int(np.floor(float(second_microsecond_str)))
        microsecond = int(1000000 * (float(second_microsecond_str) - second))
        year, month, day = 2022, 1, 1
        startDT = datetime(year,month,day,int(hour_str),int(minute_str),second,microsecond)
        # print('model_samplePointNum =', model_samplePointNum)
        converted_timeStamps = []
        dt = startDT
        for i in range(resampledLen):
            hour_str, minute_str = str(dt.hour), str(dt.minute)
            second_microsecond_str = str(dt.second + (dt.microsecond / 1000000))
            converted_timeStamps += [hour_str + ':' + minute_str + ':' + second_microsecond_str]
            dt += timedelta(seconds= 1 / model_samplingFreq)
        return converted_timeStamps

    def read_data(self, taskHandle, data):
        current_time = datetime.now()
        return current_time, data

    @staticmethod
    def updateTimeStamp(now, t, dt):
        """
        # Params
        - t (float)
        - dt (float)
        """
        delta = timedelta(microseconds=t * dt * 1000 * 1000)
        current_time = now + delta
        return current_time

    def serve(self):

        # global_t = 0
        # dt = 1.0 / self.model_samplingFreq
        # print('eeg.shape[0] =', self.eeg.shape[0])
        # print('eegLen =', self.eegLen)
        # print('wsizeInTimePoints =', self.wsizeInTimePoints)
        for startSamplePoint in range(0, self.eegLen, self.wsizeInTimePoints):
            # now = datetime.now()
            endSamplePoint = startSamplePoint + self.wsizeInTimePoints
            timeStamps_fragment = self.timeStamps[startSamplePoint:endSamplePoint]
            eeg_fragment = self.eeg[startSamplePoint:endSamplePoint]
            if self.useEMG:
                emg_fragment = self.emg[startSamplePoint:endSamplePoint]

            # startSamplePoint = endSamplePoint
            eeg_fragmentLength = eeg_fragment.shape[0]
            # presentTime = timeFormatting.presentTimeEscaped()
            # self.logFile.write(timeStamps_fragment[0] + ', ' + str(eeg_fragmentLength) + '\n')
            # self.logFile.flush()

            dataToAIClient = ''
            # print('eeg_fragmentLength =', eeg_fragmentLength)
            # print('len(timeStamps_fragment) =', len(timeStamps_fragment))
            # print('len(eeg_fragment) =', len(eeg_fragment))
            # print('eeg_fragment =', eeg_fragment)
            for t in range(eeg_fragmentLength):
                dataToAIClient += timeStamps_fragment[t] + '\t' + '{0:.6f}'.format(eeg_fragment[t])
                if self.useEMG:
                    dataToAIClient += '\t' + '{0:.6f}'.format(emg_fragment[t]) + '\n'
                else:
                    dataToAIClient += '\n'

            # print('sending dataToAIClient to client')
            # print('in server, dataToAIClient =', dataToAIClient)
            self.client.process(dataToAIClient)
