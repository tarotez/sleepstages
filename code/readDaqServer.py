from PyDAQmx import DAQmxCreateTask
from PyDAQmx import DAQmxCreateAIVoltageChan
from PyDAQmx import DAQmxCfgSampClkTiming
from PyDAQmx import DAQmxStartTask
from PyDAQmx import DAQmxReadAnalogF64
from PyDAQmx import DAQmxStopTask
from PyDAQmx import DAQmxClearTask
from PyDAQmx import TaskHandle
from PyDAQmx import DAQmx_Val_Volts
from PyDAQmx import DAQmx_Val_Cfg_Default, DAQmx_Val_Diff, DAQmx_Val_RSE, DAQmx_Val_NRSE
from PyDAQmx import DAQmx_Val_Rising, DAQmx_Val_Falling
from PyDAQmx import DAQmx_Val_FiniteSamps, DAQmx_Val_ContSamps
from PyDAQmx import DAQmx_Val_GroupByChannel
from PyDAQmx import DAQError
from PyDAQmx import int32
from ctypes import byref
import numpy
import datetime
import tqdm
import time
import math

import pdb

from parameterSetup import ParameterSetup
import timeFormatting

class ReadDAQServer:
    def __init__(self, client, recordWaves, channelOpt=2, sampRate=128, dataAcquisitionFreq=0.1,
                 timeout=500, numEpoch=600000, eeg_std=None, ch2_std=None,
                 prefix='/Users/ssg/Projects/rem'):
        """
        # Params

        - sampRate (float): サンプリングレート (Hz)
        - dataAcquisitionFreq (float): data acuisition frequency (Hz)
        - timeout (float): how long the program waits in sec (set to -1 to wait indefinitely)
        - numEpoch (int): 予測を行うエポック数
        - eeg_std (float?): EEGについて決め打ちstdがある場合はこれを指定する
        - ch2_std (float?): ch2について決め打ちstdがある場合はこれを指定する
        - prefix (str): このリポジトリのディレクトリパス

        float? はオプショナル型的な何かのつもり (= float or None)．
        """

        self.client = client
        self.recordWaves = recordWaves

        if channelOpt == 1:
            self.channelNum = 1
        else:
            self.channelNum = 2

        self.sampRate = sampRate
        self.dataAcquisitionFreq = dataAcquisitionFreq
        self.timeout = timeout  # set to -1 to wait indefinitely
        self.numEpoch = numEpoch
        self.numSampsPerChan = round(self.sampRate / self.dataAcquisitionFreq)

        # prefixはこのプログラムが存在するディレクトリを指定する変数
        # 現在は/Users/ssg/Projects/remというディレクトリに配置されてるため、
        # prefix = '/Users/ssg/Projects/rem'
        self.prefix = prefix

        # あらかじめ定められた標準偏差がある場合、
        # その値を保存する
        self.eeg_std = eeg_std
        self.ch2_std = ch2_std

        self.params = ParameterSetup()
        presentTime = timeFormatting.presentTimeEscaped()
        fileName = 'daq.' + presentTime + '.csv'
        self.logFile = open(self.params.logDir + '/' + fileName, 'a')


    def read_data(self, taskHandle, data):
        """ read data points

        # Params
        - taskHandle (taskHandle型)
        - numSampsPerChan (int) : number of samples per channel
        - timeout (float) : time out in sec
        - fillMode:
        - readArray (numpy.ndarray): array for writing out output
        - arraySizeInSamps : the size of the readArray array
        - numSampsPerChanRead : the number of samples that were actually read out
        - reserved
        """

        # self.logFile.write('in read_data, self.numSampsPerChan = ' + str(self.numSampsPerChan) + '\n')
        # self.logFile.write('in read_data, self.channelNum = ' + str(self.channelNum) + '\n')
        # self.logFile.write('in read_data, data.shape = ' + str(data.shape) + '\n')
        # self.logFile.write('in read_data, self.timeout = ' + str(self.timeout) + '\n')
        # self.logFile.flush()

        try:
            DAQmxReadAnalogF64(taskHandle, self.numSampsPerChan, self.timeout,
                           DAQmx_Val_GroupByChannel, data, self.numSampsPerChan * self.channelNum,
                           byref(int32()), None)

        except:
            import sys
            traceback = sys.exc_info()[2]
            pdb.post_morte(traceback)

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

        while True:
            now = datetime.datetime.now()
            print('sampling rate             : {}'.format(self.sampRate))
            print('samples per channel : {}'.format(self.numSampsPerChan))
            print('number of epochs          : {}'.format(self.numEpoch))
            print('number of channels          : {}'.format(self.channelNum))

            dt = 1.0 / self.sampRate

            taskHandle = TaskHandle()
            data = numpy.zeros((self.numSampsPerChan * self.channelNum,), dtype=numpy.float64)

            try:
                # DAQmx Configure Code
                DAQmxCreateTask("", byref(taskHandle))
                # param: :physicalChannel
                # param: :nameToAssignToChannel
                # param: :terminalConfig
                # param: :minVal
                # param: :maxVal
                # param: :units
                # param: :customScaleName

                def createChannel(devID, channelIDs):                    
                    try:
                        DAQmxCreateAIVoltageChan(taskHandle, "Dev" + str(devID) + "/ai" + channelIDs, "",
                                         ### DAQmx_Val_Cfg_Default, -10.0, 10.0,
                                         ### DAQmx_Val_Diff, -10.0, 10.0,
                                         ### DAQmx_Val_NRSE, -10.0, 10.0,
                                         self.DAQmx_Val_dict[self.terminal_config], -10.0, 10.0,
                                         DAQmx_Val_Volts, None)
                        channelNum = 2 if len(channelIDs) > 1 else 1
                        print("at DAQmxCreateAIVoltageChan, created succesfully with channelNum == " + str(channelNum) + ".")
                        return 1
                    except DAQError as err:
                        print("at DAQmxCreateAIVoltageChan, DAQmx Error: %s" % err)
                        self.logFile.write("at DAQmxCreateAIVoltageChan, DAQmx Error: %s" % err)
                        self.logFile.flush()
                        return 0

                if self.channelNum == 2:
                    if not createChannel("1", "1:2"):
                        if not createChannel("2","1:2"):
                            if not createChannel("0","1:2"):
                                pass
                else:
                    if not createChannel("1", "1"):
                        if not createChannel("2","1"):
                            if not createChannel("0","1"):
                                pass
                    '''
                    try:
                        DAQmxCreateAIVoltageChan(taskHandle, "Dev1/ai1", "",
                                         ### DAQmx_Val_Diff, -10.0, 10.0,
                                         DAQmx_Val_Cfg_Default, -10.0, 10.0,
                                         DAQmx_Val_Volts, None)
                        print("at DAQmxCreateAIVoltageChan, created succesfully with channelNum == 1.")
                    except DAQError as err:
                        print("at DAQmxCreateAIVoltageChan, DAQmx Error: %s" % err)
                        self.logFile.write("at DAQmxCreateAIVoltageChan, DAQmx Error: %s" % err)
                        self.logFile.flush()
                        try:
                            DAQmxCreateAIVoltageChan(taskHandle, "Dev0/ai1", "",
                                         ### DAQmx_Val_Diff, -10.0, 10.0,
                                         DAQmx_Val_Cfg_Default, -10.0, 10.0,
                                         DAQmx_Val_Volts, None)
                        except DAQError as err:
                            print("at DAQmxCreateAIVoltageChan, DAQmx Error: %s" % err)
                            self.logFile.write("at DAQmxCreateAIVoltageChan, DAQmx Error: %s" % err)
                            self.logFile.flush()
                            DAQmxCreateAIVoltageChan(taskHandle, "Dev2/ai1", "",
                                         ### DAQmx_Val_Diff, -10.0, 10.0,
                                         DAQmx_Val_Cfg_Default, -10.0, 10.0,
                                         DAQmx_Val_Volts, None)
                                         '''

                # param: taskHandle
                # param: source (const char[])
                # param: rate (float) : sapmling rate (Hz)
                # param: activeEdge
                # param: sampleMode (int32) : DAQmx_Val_FiniteSamps
                #        or DAQmx_Val_ContSamps or DAQmx_Val_HWTimedSinglePoint
                # param: numSampsPerChan (int) : number of samples per channel
                DAQmxCfgSampClkTiming(taskHandle, "", self.sampRate,
                                      ### DAQmx_Val_Diff, DAQmx_Val_ContSamps,
                                      ### DAQmx_Val_Falling, DAQmx_Val_ContSamps,
                                      DAQmx_Val_Rising, DAQmx_Val_ContSamps,
                                      self.numSampsPerChan)

                # DAQmx Start Code
                DAQmxStartTask(taskHandle)
                for timestep in tqdm.tqdm(range(1, self.numEpoch + 1)):

                    # self.logFile.write('before read_data\n')
                    # self.logFile.flush()

                    # Starting a new task for each segment may make delays,
                    # so it's better to move it outside "for timestep" loop,
                    # but then there will be an error stating
                    # "PyDAQmx.DAQmxFunctions.DAQError: <err>Attempted to read a sample beyond the final sample acquired. The acquisition has stopped, therefore the sample specified will never be available."
                    #### DAQmxStartTask(taskHandle)

                    now, data = self.read_data(taskHandle, data)
                    # self.logFile.write('after read_data\n')
                    # self.logFile.flush()

                    if self.channelNum == 2:
                        sampleNum = data.shape[0] // 2
                        eeg_data = data[:sampleNum]
                        ch2_data = data[sampleNum:]
                    else:
                        sampleNum = data.shape[0]
                        eeg_data = data[:]

                    dataToClient = ''
                    for t in range(sampleNum):
                        current_time = self.updateTimeStamp(now, t, dt)
                        ftime = current_time.strftime('%H:%M:%S.')
                        ftime += '%06d' % current_time.microsecond

                        dataToClient += ftime + '\t' + str(eeg_data[t])
                        if self.channelNum == 2:
                            dataToClient += '\t' + str(ch2_data[t]) + '\n'
                        else:
                            dataToClient += '\n'

                    # record log
                    presentTime = timeFormatting.presentTimeEscaped()
                    if self.recordWaves:
                        self.logFile.write(presentTime + ', ' + str(eeg_data.shape[0]) + ', ' + str(eeg_data) + '\n')
                    else:
                        self.logFile.write(presentTime + ', ' + str(eeg_data.shape[0]) + '\n')
                    self.logFile.flush()

                    # classifierClientにデータを送信する
                    # dataToClient = dataToClient.rstrip()
                    self.client.process(dataToClient)

            except DAQError as err:
                print("DAQmx Error: %s" % err)
                self.logFile.write("DAQmx Error: %s" % err)
                self.logFile.flush()
                # raise err

            finally:
                if taskHandle:
                    DAQmxStopTask(taskHandle)
                    DAQmxClearTask(taskHandle)
