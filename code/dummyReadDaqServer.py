from ctypes import byref
import numpy as np
import datetime
import tqdm
import time
import math
import pickle
from os import listdir
from random import shuffle
from parameterSetup import ParameterSetup
import timeFormatting
# import tensorflow as tf
# from tensorflow.keras import backend as K

class DummyReadDAQServer:
    def __init__(self, client, fileID, recordWaves, channelNum, offsetWindowID, sleepTime,
                 sampRate=128, numEpoch=600000, eeg_std=None, ch2_std=None, channelOpt=2):
        """
        # Params

        - sampRate (int): サンプリングレート
        - numEpoch (int): 予測を行うエポック数
        - eeg_std (float?): EEGについて決め打ちstdがある場合はこれを指定する
        - ch2_std (float?): ch2について決め打ちstdがある場合はこれを指定する

        float? はオプショナル型的 (= float or None)．
        """

        self.client = client
        self.recordWaves = recordWaves
        self.channelNum = channelNum

        self.offsetWindowID = offsetWindowID
        self.sleepTime = sleepTime

        self.sampRate = sampRate
        self.numEpoch = numEpoch

        # あらかじめ定められた標準偏差がある場合、その値を保存する
        self.eeg_std = eeg_std
        self.ch2_std = ch2_std

        self.params = ParameterSetup()
        pickledDir = self.params.pickledDir
        classifierType = self.params.classifierType
        classifierParams = self.params.classifierParams
        samplingFreq = self.params.samplingFreq
        windowSizeInSec = self.params.windowSizeInSec
        # self.wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
        self.wsizeInTimePoints = self.client.updateGraph_samplePointNum
        eegFilePrefix = 'eegAndStage'

        if fileID.startswith('m'):
            files_L = listdir(pickledDir)
            shuffle(files_L)
            for fileFullName in files_L:
                if fileFullName.startswith(eegFilePrefix):
                    break
        else:
            fileFullName = eegFilePrefix + '.' + fileID + '.pkl'

        print('reading file ' + fileFullName)
        dataFileHandler = open(pickledDir + '/' + fileFullName, 'rb')
        (eeg, ch2, stageSeq, timeStamps) = pickle.load(dataFileHandler)
        self.eeg = eeg
        self.ch2 = ch2
        self.timeStamps = timeStamps
        self.stageSeq = stageSeq
        if self.offsetWindowID > 0:
            offsetSampleNum = self.offsetWindowID * self.wsizeInTimePoints
            self.eeg = self.eeg[offsetSampleNum:]
            self.ch2 = self.ch2[offsetSampleNum:]
            self.timeStamps = self.timeStamps[offsetSampleNum:]
            self.stageSeq = self.stageSeq[self.offsetWindowID:]
        self.sLen = len(stageSeq)
        self.wNum = min(eeg.shape[0], self.sLen * self.wsizeInTimePoints)

        presentTime = timeFormatting.presentTimeEscaped()
        fileName = 'daq.' + presentTime + '.csv'
        self.logFile = open(self.params.logDir + '/' + fileName, 'a')

    def serve(self):
        global_t = 0
        dt = 1.0 / self.sampRate

        # for testing (stopping early), 2019.3.25
        # self.wNum = 5000
        for startSamplePoint in range(0, self.wNum, self.wsizeInTimePoints):
            endSamplePoint = startSamplePoint + self.wsizeInTimePoints
            eeg_data = self.eeg[startSamplePoint:endSamplePoint]
            # if self.useEMG:
            #    emg_data = self.emg[startSamplePoint:endSamplePoint]
            ch2_data = self.ch2[startSamplePoint:endSamplePoint]
            timeStamps_data = self.timeStamps[startSamplePoint:endSamplePoint]

            # startSamplePoint = endSamplePoint
            dataToClient = ''
            eegLength = eeg_data.shape[0]
            # print('eeg_data.shape[0] = ' + str(eegLength))

            presentTime = timeFormatting.presentTimeEscaped()
            self.logFile.write(presentTime + ', ' + str(eegLength) + '\n')
            self.logFile.flush()

            for t in range(eegLength):
                ftime = timeStamps_data[t]
                dataToClient += ftime + '\t' + '{0:.6f}'.format(eeg_data[t])
                if self.channelNum == 2:
                    dataToClient += '\t' + '{0:.6f}'.format(ch2_data[t]) + '\n'
                else:
                    dataToClient += '\n'
            self.client.process(dataToClient)
            time.sleep(self.sleepTime)
