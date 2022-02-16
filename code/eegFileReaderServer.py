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

    def __init__(self, client, eegFilePath, samplingFreq=128, channelOpt=1):

        self.client = client
        self.samplingFreq = samplingFreq
        if channelOpt == 1:
            self.useEMG = 0
        if channelOpt == 2:
            self.useEMG = 1

        self.params = ParameterSetup()
        pickledDir = self.params.pickledDir
        classifierType = self.params.classifierType
        classifierParams = self.params.classifierParams
        # samplingFreq = self.params.samplingFreq
        windowSizeInSec = self.params.windowSizeInSec
        # self.wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
        self.wsizeInTimePoints = self.client.updateGraph_samplePointNum

        dataReader = DataReader()
        print('for EEG, reading file ' + eegFilePath)
        eeg, emg, timeStamps = dataReader.readEEG(eegFilePath)
        # print('for stageSeq, reading file ' + stageFilePath)
        # stageSeq = dataReader.readStageSeq(stageFilePath)
        print('timeStamps[0] =', timeStamps[0])

        print('self.samplingFreq =', self.samplingFreq)

        observed_samplePointNum = self.samplingFreq * self.params.windowSizeInSec

        print('observed_samplePointNum =', observed_samplePointNum)

        # self.eeg = eeg
        # self.emg = emg
        # self.timeStamps = timeStamps

        #####
        model_samplingFreq = 128
        model_samplePointNum = model_samplingFreq * 10
        #######

        self.eeg = up_or_down_sampling(eeg, model_samplePointNum, observed_samplePointNum)
        self.emg = up_or_down_sampling(emg, model_samplePointNum, observed_samplePointNum)
        self.timeStamps = self.convertTimeStamps(timeStamps, model_samplingFreq, model_samplePointNum, observed_samplePointNum)
        self.wNum = self.eeg.shape[0]

        presentTime = timeFormatting.presentTimeEscaped()
        fileName = 'daq.' + presentTime + '.csv'
        self.logFile = open(self.params.logDir + '/' + fileName, 'a')

        self.serve()

    def convertTimeStamps(self, timeStamps, model_samplingFreq, model_samplePointNum, observed_samplePointNum):
        print('timeStamps[0] =', timeStamps[0])
        hour_str, minute_str, second_microsecond_str = timeStamps[0].split(':')
        second = int(np.floor(float(second_microsecond_str)))
        microsecond = int(1000000 * (float(second_microsecond_str) - second))
        year, month, day = 2022, 1, 1
        startDT = datetime(year,month,day,int(hour_str),int(minute_str),second,microsecond)
        converted_timeStamps = []
        dt = startDT
        for i in range(model_samplePointNum):
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

        global_t = 0
        dt = 1.0 / self.samplingFreq
        for startSamplePoint in range(0, self.wNum, self.wsizeInTimePoints):
            now = datetime.now()

            endSamplePoint = startSamplePoint + self.wsizeInTimePoints

            timeStamps_fragment = self.timeStamps[startSamplePoint:endSamplePoint]
            eeg_fragment = self.eeg[startSamplePoint:endSamplePoint]
            if self.useEMG:
                emg_fragment = self.emg[startSamplePoint:endSamplePoint]

            # startSamplePoint = endSamplePoint
            eeg_fragmentLength = eeg_fragment.shape[0]
            # presentTime = timeFormatting.presentTimeEscaped()
            self.logFile.write(timeStamps_fragment[0] + ', ' + str(eeg_fragmentLength) + '\n')
            self.logFile.flush()

            dataToAIClient = ''
            for t in range(eeg_fragmentLength):
                dataToAIClient += timeStamps_fragment[t] + '\t' + '{0:.6f}'.format(eeg_fragment[t])
                if self.useEMG:
                    dataToAIClient += '\t' + '{0:.6f}'.format(emg_fragment[t]) + '\n'
                else:
                    dataToAIClient += '\n'

            # print('sending dataToAIClient to client')
            # print('in server, dataToAIClient =', dataToAIClient)
            self.client.process(dataToAIClient)
