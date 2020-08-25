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
        self.wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.

        # eegFilePath = fileID + '.txt'
        # stageFilePath = fileID + '.csv'

        dataReader = DataReader()
        print('for EEG, reading file ' + eegFilePath)
        eeg, emg, timeStamps = dataReader.readEEG(eegFilePath)
        # print('for stageSeq, reading file ' + stageFilePath)
        # stageSeq = dataReader.readStageSeq(stageFilePath)

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

        self.serve(client)

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

    def serve(self, client):

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

            eeg_data = self.eeg[startSamplePoint:endSamplePoint]
            if self.useEMG:
                emg_data = self.emg[startSamplePoint:endSamplePoint]

            # startSamplePoint = endSamplePoint
            dataToClient = ''
            eegLength = eeg_data.shape[0]
            # print('%%%% eeg_data.shape[0] = ' + str(eegLength))

            presentTime = timeFormatting.presentTimeEscaped()
            self.logFile.write(presentTime + ', ' + str(eegLength) + '\n')
            self.logFile.flush()

            for t in range(eegLength):
                current_time = self.updateTimeStamp(now, t, dt)
                ftime = current_time.strftime('%H:%M:%S.')
                ftime += '%06d' % current_time.microsecond

                dataToClient += ftime + '\t' + '{0:.6f}'.format(eeg_data[t])
                if self.useEMG:
                    dataToClient += '\t' + '{0:.6f}'.format(emg_data[t]) + '\n'
                else:
                    dataToClient += '\n'
            # print('sending dataToClient to client')
            client.process(dataToClient)
                ###### connectedLine = connectedLine + dataFromDaq.decode('utf-8')
                # if len(connectedLine.split('\n')) > self.connectedLineThresh:
                #    break

                # if len(dataToClient.encode('utf-8')) > self.dataToClientSizeLimit or t == (eegLength - 1):
                    # print("len(dataToClient.encode('utf-8')) = " + str(len(dataToClient.encode('utf-8'))))
                    # readDaqServerにデータを送信する
                    # dataToClient = dataToClient.rstrip()
                    ###### must send a whole data (connected)
                    # client.process(dataToClient.encode('utf-8'))
                    # dataToClient = ''
                    ### time.sleep(0.01)
            # time.sleep(0.01)
