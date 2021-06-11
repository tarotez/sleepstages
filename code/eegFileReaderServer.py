import numpy
import datetime
import time
from os import listdir
from random import shuffle
from parameterSetup import ParameterSetup
import timeFormatting
from dataReader import DataReader

class EEGFileReaderServer:

    def __init__(self, client, eegFilePath, sampRate=128, channelOpt=1):

        self.client = client
        self.sampRate = sampRate
        if channelOpt == 1:
            self.useEMG = 0
        if channelOpt == 2:
            self.useEMG = 1

        self.params = ParameterSetup()
        pickledDir = self.params.pickledDir
        classifierType = self.params.classifierType
        classifierParams = self.params.classifierParams
        samplingFreq = self.params.samplingFreq
        windowSizeInSec = self.params.windowSizeInSec
        # self.wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
        self.wsizeInTimePoints = self.client.updateGraph_samplePointNum
        # eegFilePath = fileID + '.txt'
        # stageFilePath = fileID + '.csv'

        dataReader = DataReader()
        print('for EEG, reading file ' + eegFilePath)
        eeg, emg, timeStamps = dataReader.readEEG(eegFilePath)
        # print('for stageSeq, reading file ' + stageFilePath)
        # stageSeq = dataReader.readStageSeq(stageFilePath)

        self.timeStamps = timeStamps
        self.eeg = eeg
        self.emg = emg
        # self.stageSeq = stageSeq
        # self.sLen = len(stageSeq)
        # self.wNum = min(eeg.shape[0], self.sLen * self.wsizeInTimePoints)
        self.wNum = eeg.shape[0]
        # self.dataToClientSizeLimit = 4096

        presentTime = timeFormatting.presentTimeEscaped()
        fileName = 'daq.' + presentTime + '.csv'
        self.logFile = open(self.params.logDir + '/' + fileName, 'a')

        self.serve()

    def read_data(self, taskHandle, data):
        current_time = datetime.datetime.now()
        return current_time, data

    @staticmethod
    def updateTimeStamp(now, t, dt):
        """
        # Params
        - t (float)
        - dt (float)
        """
        delta = datetime.timedelta(microseconds=t * dt * 1000 * 1000)
        current_time = now + delta
        return current_time

    def serve(self):

        global_t = 0
        dt = 1.0 / self.sampRate
        # while True:
        # print('%%%% self.wsizeInTimePoints = ' + str(self.wsizeInTimePoints))
        # startSamplePoint = 0
        # while startSamplePoint + self.wsizeInTimePoints <= self.wNum:
        for startSamplePoint in range(0, self.wNum, self.wsizeInTimePoints):
            now = datetime.datetime.now()
            # print('sampling rate             : {}'.format(self.sampRate))

            endSamplePoint = startSamplePoint + self.wsizeInTimePoints

            timeStamps_fragment = self.timeStamps[startSamplePoint:endSamplePoint]
            eeg_fragment = self.eeg[startSamplePoint:endSamplePoint]
            if self.useEMG:
                emg_fragment = self.emg[startSamplePoint:endSamplePoint]

            # startSamplePoint = endSamplePoint
            dataToClient = ''
            eeg_fragmentLength = eeg_fragment.shape[0]
            # print('%%%% eeg_fragment.shape[0] = ' + str(eeg_fragmentLength))

            # presentTime = timeFormatting.presentTimeEscaped()
            self.logFile.write(timeStamps_fragment[0] + ', ' + str(eeg_fragmentLength) + '\n')
            self.logFile.flush()

            for t in range(eeg_fragmentLength):
                dataToClient += timeStamps_fragment[t] + '\t' + '{0:.6f}'.format(eeg_fragment[t])
                if self.useEMG:
                    dataToClient += '\t' + '{0:.6f}'.format(emg_fragment[t]) + '\n'
                else:
                    dataToClient += '\n'
            # print('sending dataToClient to client')
            # print(dataToClient)
            self.client.process(dataToClient)
            ###### connectedLine = connectedLine + dataFromDaq.decode('utf-8')
            # if len(connectedLine.split('\n')) > self.connectedLineThresh:
            #    break

            # if len(dataToClient.encode('utf-8')) > self.dataToClientSizeLimit or t == (eeg_fragmentLength - 1):
                # print("len(dataToClient.encode('utf-8')) = " + str(len(dataToClient.encode('utf-8'))))
                # readDaqServerにデータを送信する
                # dataToClient = dataToClient.rstrip()
                ###### must send a whole data (connected)
                # client.process(dataToClient.encode('utf-8'))
                # dataToClient = ''
                ### time.sleep(0.01)
            # time.sleep(0.01)
