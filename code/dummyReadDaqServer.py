from ctypes import byref
import numpy
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
    def __init__(self, client, fileID, recordWaves, offsetWindowID, sleepTime,
                 sampRate=128, numEpoch=600000, eeg_std=None, ch2_std=None,
                 prefix='/Users/ssg/Projects/rem', channelOpt=2):
        """
        # Params

        - sampRate (int): サンプリングレート
        - numEpoch (int): 予測を行うエポック数
        - eeg_std (float?): EEGについて決め打ちstdがある場合はこれを指定する
        - ch2_std (float?): ch2について決め打ちstdがある場合はこれを指定する
        - prefix (str): このリポジトリのディレクトリパス

        float? はオプショナル型的な何かのつもり (= float or None)．
        """

        self.client = client
        self.recordWaves = recordWaves
        self.offsetWindowID = offsetWindowID
        self.sleepTime = sleepTime

        self.sampRate = sampRate
        self.numEpoch = numEpoch
        self.sampsPerChan = sampRate * numEpoch

        # prefixはこのプログラムが存在するディレクトリを指定する変数
        # 現在は/Users/ssg/Projects/remというディレクトリに配置されてるため、
        # prefix = '/Users/ssg/Projects/rem'
        self.prefix = prefix

        # あらかじめ定められた標準偏差がある場合、
        # その値を保存する
        self.eeg_std = eeg_std
        self.ch2_std = ch2_std

        # self.t = threading.Thread(target=self.serve, daemon=True)

        # if channelOpt == 1:
        #    self.useEMG = 0
        # if channelOpt == 2:
        #    self.useEMG = 1

        self.params = ParameterSetup()
        pickledDir = self.params.pickledDir
        classifierType = self.params.classifierType
        classifierParams = self.params.classifierParams
        samplingFreq = self.params.samplingFreq
        windowSizeInSec = self.params.windowSizeInSec
        self.wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
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
        # self.dataToClientSizeLimit = 4096
        # self.dataToClientSizeLimit = 2048

        presentTime = timeFormatting.presentTimeEscaped()
        fileName = 'daq.' + presentTime + '.csv'
        self.logFile = open(self.params.logDir + '/' + fileName, 'a')
        ### self.graph = tf.get_default_graph()

    # def run(self):
    #    self.t.run()

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
        ### with self.graph.as_default():
            global_t = 0
            dt = 1.0 / self.sampRate
            # while True:
            # print('self.wsizeInTimePoints = ' + str(self.wsizeInTimePoints))
            # startSamplePoint = 0
            # while startSamplePoint + self.wsizeInTimePoints <= self.wNum:

            # for testing (stopping early), 2019.3.25
            # self.wNum = 5000
            for startSamplePoint in range(0, self.wNum, self.wsizeInTimePoints):
                ### now = datetime.datetime.now()
                # print('sampling rate             : {}'.format(self.sampRate))
                # print('total number of acquiring : {}'.format(self.sampsPerChan))
                # print('number of epochs          : {}'.format(self.numEpoch))
                # print('startSamplePoint = ' + str(startSamplePoint))

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
                    ### current_time = self.updateTimeStamp(now, t, dt)
                    ### ftime = current_time.strftime('%H:%M:%S.')
                    ### ftime += '%06d' % current_time.microsecond
                    ftime = timeStamps_data[t]
                    dataToClient += ftime + '\t' + '{0:.6f}'.format(eeg_data[t])
                    dataToClient += '\t' + '{0:.6f}'.format(ch2_data[t]) + '\n'
                    # if self.useEMG:
                    #    dataToClient += '\t' + '{0:.6f}'.format(emg_data[t]) + '\n'
                    #else:
                    #    dataToClient += '\n'

                self.client.process(dataToClient)
                time.sleep(self.sleepTime)
