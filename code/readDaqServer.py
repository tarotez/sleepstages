from PyDAQmx import DAQmxCreateTask
from PyDAQmx import DAQmxCreateAIVoltageChan
from PyDAQmx import DAQmxCfgSampClkTiming
from PyDAQmx import DAQmxStartTask
from PyDAQmx import DAQmxReadAnalogF64
from PyDAQmx import DAQmxStopTask
from PyDAQmx import DAQmxClearTask
from PyDAQmx import TaskHandle
from PyDAQmx import DAQmx_Val_Volts
from PyDAQmx import DAQmx_Val_Cfg_Default, DAQmx_Val_Diff, DAQmx_Val_RSE, DAQmx_Val_NRSE, DAQmx_Val_PseudoDiff
from PyDAQmx import DAQmx_Val_Rising, DAQmx_Val_Falling
from PyDAQmx import DAQmx_Val_FiniteSamps, DAQmx_Val_ContSamps
from PyDAQmx import DAQmx_Val_GroupByChannel, DAQmx_Val_GroupByScanNumber
from PyDAQmx import DAQError
# from PyDAQmx import DAQmxSetReadOverWrite
# from PyDAQmx import DAQmx_Val_OverwriteUnreadSamps, DAQmx_Val_DoNotOverwriteUnreadSamps
from PyDAQmx import int32
from ctypes import byref
import numpy as np
import datetime
import tqdm
import time
import math

import pdb

from parameterSetup import ParameterSetup
import timeFormatting

class ReadDAQServer:
    def __init__(self, client, recordWaves, channelNum, samplingFreq,
                 timeout=500, maxNumEpoch=600000, eeg_std=None, ch2_std=None):
        """
        # Params
        - samplingFreq (float): サンプリングレート (Hz)
        - timeout (float): how long the program waits in sec (set to -1 to wait indefinitely)
        - maxNumEpoch (int): 予測を行うエポック数
        - eeg_std (float?): EEGについて決め打ちstdがある場合はこれを指定する
        - ch2_std (float?): ch2について決め打ちstdがある場合はこれを指定する
        float? はオプショナル型的 (= float or None)．
        """

        self.client = client
        self.recordWaves = recordWaves
        self.channelNum = channelNum
        self.samplingFreq = samplingFreq
        self.timeout = timeout  # set to -1 to wait indefinitely
        self.maxNumEpoch = maxNumEpoch
        self.numSampsPerChan = self.client.updateGraph_samplePointNum

        # あらかじめ定められた標準偏差がある場合、その値を保存する
        self.eeg_std = eeg_std
        self.ch2_std = ch2_std

        self.past_eeg, self.past_ch2 = np.array([]), np.array([])

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
        - fillMode: DAQmx_Val_GroupByChannel or DAQmx_Val_GroupByScanNumber
        - data (np.ndarray): array for writing out output
        - arraySizeInSamps : the size of the readArray array
        - byref(int32()) : the number of samples that were actually read out
        - reserved
        """
        try:
            DAQmxReadAnalogF64(taskHandle, self.numSampsPerChan, self.timeout,
                    DAQmx_Val_GroupByChannel, data, self.numSampsPerChan, byref(int32()), None)
            ### DAQmxReadAnalogF64(taskHandle, self.numSampsPerChan, self.timeout,
            ###             DAQmx_Val_GroupByChannel, data, self.numSampsPerChan * self.channelNum, byref(int32()), None)
            # DAQmxReadAnalogF64(taskHandle, 1, self.timeout,
            #             DAQmx_Val_GroupByChannel, data, self.channelNum, byref(int32()), None)
            # DAQmxReadAnalogF64(taskHandle, self.numSampsPerChan, self.timeout,
            #             DAQmx_Val_GroupByScanNumber, data, self.numSampsPerChan * self.channelNum, byref(int32()), None)

        except:
            import sys
            traceback = sys.exc_info()[2]
            pdb.post_mortem(traceback)

        current_time = datetime.datetime.now()
        return current_time

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
            print('sampling frequency          : {}'.format(self.samplingFreq))
            print('samples per channel         : {}'.format(self.numSampsPerChan))
            print('maximum number of epochs    : {}'.format(self.maxNumEpoch))
            print('number of channels          : {}'.format(self.channelNum))

            dt = 1.0 / self.samplingFreq

            taskHandle = TaskHandle()

            try:
                # DAQmx Configure Code
                DAQmxCreateTask("", byref(taskHandle))

                def createChannel(devID, channelIDs):
                    try:
                        ### device_and_channelsL = ["Dev" + str(devID) + "/ai" + str(channelID) for channelID in channelIDs]
                        ### device_and_channels = ", ".join(device_and_channelsL)
                        ### print('device_and_channels =', device_and_channels)
                        DAQmx_Val_dict = {'DIFF' : DAQmx_Val_Diff, 'RSE' : DAQmx_Val_RSE, 'NRSE' : DAQmx_Val_NRSE, 'PseudoDIFF' : DAQmx_Val_PseudoDiff}
                        ### DAQmxCreateAIVoltageChan(taskHandle, device_and_channels, "",
                        ###        DAQmx_Val_dict[self.terminal_config], -10.0, 10.0, DAQmx_Val_Volts, None)
                        for channelID in channelIDs:
                            device_and_channel = "Dev" + str(devID) + "/ai" + str(channelID)
                            DAQmxCreateAIVoltageChan(taskHandle, device_and_channel, "",
                                DAQmx_Val_dict[self.terminal_config], -10.0, 10.0, DAQmx_Val_Volts, None)

                        print("at DAQmxCreateAIVoltageChan, created succesfully with channelNum == " + str(self.channelNum) + ".")
                        return 1
                    except DAQError as err:
                        print("at DAQmxCreateAIVoltageChan, DAQmx Error: %s" % err)
                        self.logFile.write("at DAQmxCreateAIVoltageChan, DAQmx Error: %s" % err)
                        self.logFile.flush()
                        return 0

                if self.channelNum == 2:
                    if not createChannel(1, [1,0]):
                        if not createChannel(2, [1,0]):
                            if not createChannel(0, [1,0]):
                                pass
                else:
                    if not createChannel(1, [1]):
                        if not createChannel(2, [1]):
                            if not createChannel(0, [1]):
                                pass

                # param: taskHandle
                # param: source (const char[])
                # param: rate (float) : sapmling rate (Hz)
                # param: activeEdge
                # param: sampleMode (int32) : DAQmx_Val_FiniteSamps
                #        or DAQmx_Val_ContSamps or DAQmx_Val_HWTimedSinglePoint
                # param: numSampsPerChan (int) : number of samples per channel
                DAQmxCfgSampClkTiming(taskHandle, "", self.samplingFreq, DAQmx_Val_Rising,
                                      DAQmx_Val_ContSamps,
                                      # DAQmx_Val_FiniteSamps,
                                      self.numSampsPerChan)

                # DAQmx Start Code
                DAQmxStartTask(taskHandle)
                # DAQmxSetReadOverWrite(taskHandle, DAQmx_Val_OverwriteUnreadSamps)

                for timestep in tqdm.tqdm(range(1, self.maxNumEpoch + 1)):
                    data = np.zeros((self.numSampsPerChan * self.channelNum,), dtype=np.float64)
                    now = self.read_data(taskHandle, data)
                    # print('data.shape =', data.shape)

                    if self.channelNum == 2:
                        sampleNum = data.shape[0] // 2
                        eeg_data = data[:sampleNum]
                        ch2_data = data[sampleNum:]
                    else:
                        sampleNum = data.shape[0]
                        eeg_data = data[:]

                    print('eeg_data[:16] =', eeg_data[:16])
                    if self.channelNum == 2:
                        print('ch2_data[:16] =', ch2_data[:16])

                    dataToClient = ''
                    for sampleID in range(sampleNum):
                        current_time = self.updateTimeStamp(now, sampleID, dt)
                        ftime = current_time.strftime('%H:%M:%S.')
                        ftime += '%06d' % current_time.microsecond

                        dataToClient += ftime + '\t' + str(eeg_data[sampleID])
                        if self.channelNum == 2:
                            dataToClient += '\t' + str(ch2_data[sampleID]) + '\n'
                        else:
                            dataToClient += '\n'

                    # dataToClient = dataToClient.rstrip()
                    # print('in server, dataToClient.shape =', dataToClient.shape)
                    # print('in server, dataToClient =', dataToClient)
                    self.client.process(dataToClient)

                    # record log
                    presentTime = timeFormatting.presentTimeEscaped()
                    if self.recordWaves:
                        self.logFile.write(presentTime + ', ' + str(eeg_data.shape[0]) + ', ' + str(eeg_data) + '\n')
                    else:
                        self.logFile.write(presentTime + ', ' + str(eeg_data.shape[0]) + '\n')
                    self.logFile.flush()

            except DAQError as err:
                print("DAQmx Error: %s" % err)
                self.logFile.write("DAQmx Error: %s" % err)
                self.logFile.flush()

            finally:
                if taskHandle:
                    DAQmxStopTask(taskHandle)
                    DAQmxClearTask(taskHandle)
