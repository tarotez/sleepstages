import sys
import socket
import struct
import numpy as np
from functools import reduce
from os import listdir
from datetime import datetime
from functools import reduce
from time import sleep
from dataReader import DataReader
from parameterSetup import ParameterSetup

# sends mock signal to online.py by connectin to networkServer

# print('HOST:', HOST)
# PORT = 45123       # The port used by the server
server_PORT = 45123

# sampleNum = 100
# chamberIDL = (random.randint(0,3) for _ in range(sampleNum))
# epochDict = {}

params = ParameterSetup()
# samplingFreq = 128
# samplingFreq = 512
# epochTime = 10
samplingFreq = params.samplingFreq
epochTime = params.windowSizeInSec

args = sys.argv
if len(args) > 1:
    server_HOST = args[1]
else:
    # HOST = '192.168.0.2'  # The server's hostname or IP address
    server_HOST = 'localhost'

if len(args) > 2:
    epochWaitTime = float(args[2])
else:
    epochWaitTime = epochTime
# channelNum = params.input_channel_num

epochSampleNum = samplingFreq * epochTime
# signal = [float(i + 3.1416) for i in range(samplingFreq * epochTime)]
chamberNum = 1
### chamberNum = 2
# chamberNum = 1

all_postFiles = listdir(params.postDir)
postFiles = []
for fileName in all_postFiles:
    if fileName[0] != '.':
        postFiles.append(fileName)

print('postFiles =', postFiles)

eegL = []
for _, inputFileName in zip(range(chamberNum), postFiles):
    dataReader = DataReader()
    eegFilePath = params.postDir + '/' + inputFileName
    print('for EEG, reading file ' + eegFilePath)


    eeg, emg, timeStamps = dataReader.readEEG(eegFilePath)
    ### eeg = dataReader.readMultiChannelEEGfromEDF(eegFilePath, channelNum)


    eegL.append(eeg)

max_eegLength = reduce(max, [len(eeg) for eeg in eegL])
print('max_eegLen =', max_eegLength)
epochNum = np.int(np.floor(max_eegLength / epochSampleNum))
print('epochNum =', epochNum)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    s.connect((server_HOST, server_PORT))

    # send setup information
    samplingFreqByte = samplingFreq.to_bytes(2, 'little')
    epochTimeByte = epochTime.to_bytes(2, 'little')
    setupByte = samplingFreqByte + epochTimeByte
    print('in mock, len(setupByte) = ', len(setupByte))
    s.sendall(setupByte)
    setupRespByte = s.recv(2)
    setupResp = struct.unpack_from('H', setupRespByte, 0)[0]
    print('setupResp =', setupResp)
    if setupResp == 0:
        print('network server returned an error code.')
        exit()

    # send eeg information
    ### for samples in range(sampleNum):
    for epochID in range(epochNum):
        epochByte = epochID.to_bytes(4, 'little')
        chamberID_permuted = np.random.permutation([c for c in range(chamberNum)])

        for chamberID in chamberID_permuted:
            chamberByte = np.int(chamberID).to_bytes(2, 'little')
            dt = datetime.now()
            yearByte, monthByte, dayByte, hourByte, minuteByte, secondByte = map(lambda x: x.to_bytes(2, 'little'), (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
            microsecByte = dt.microsecond.to_bytes(4, 'little')
            datetimeByte = yearByte + monthByte + dayByte + hourByte + minuteByte + secondByte + microsecByte
            reqByte = chamberByte + epochByte + datetimeByte

            startSampleID = epochSampleNum * epochID
            endSampleID = startSampleID + epochSampleNum
            for sample in eegL[chamberID][startSampleID:endSampleID]:
                reqByte += struct.pack('<f', sample)
            #print('len(eegL[chamberID][startSampleID:endSampleID]) =', len(eegL[chamberID][startSampleID:endSampleID]))
            # signalByte = signal.to_bytes(5120, 'little')
            # signalByte = reduce(lambda a, x: a + x, map(lambda x: x.to_bytes(4, 'little'), signal))
            # reqByte = chamberByte + epochByte + datetimeByte + signalByte
            # print('in eegSourceClient, reqByte =', reqByte)
            # print('in eegSourceClient, len(reqByte) =', len(reqByte))

            s.sendall(reqByte)
            resp = s.recv(8)
            resp_chamberID = struct.unpack_from('H', resp, 0)[0]    #WORD
            resp_epochID = struct.unpack_from('I', resp, 2)[0]    #DWORD
            resp_judge = struct.unpack_from('H', resp, 6)[0]

            print('c, e, j =', resp_chamberID, resp_epochID, resp_judge)

        sleep(epochWaitTime)
