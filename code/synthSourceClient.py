import sys
import socket
import struct
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import random
from math import pi, sin

# sends mock signal to online.py by connectin to networkServer

args = sys.argv
if len(args) > 1:
    inputSignalFreqHz = args[1]
else:
    inputSignalFreqHz = 0.125

HOST = '192.168.0.3'  # The server's hostname or IP address
# print('HOST:', HOST)
PORT = 45123       # The port used by the server

chamberNum = 4
epochNum = 10
sampleNum = chamberNum * epochNum
chamberIDL = np.random.permutation(reduce(lambda a, x: a + x, [[chamberID for chamberID in range(chamberNum)] for _ in range(epochNum)], []))
# print('chamberIDL =', chamberIDL)

### samplingFreq = 128
samplingFreq = 512
epochTime = 10
### signal = [float(i + 3.1416) for i in range(samplingFreq * epochTime)]
all_signal = [sin(2 * pi * inputSignalFreqHz * i / samplingFreq) for i in range(samplingFreq * epochTime * epochNum)]

def signal_generator(source_data, segmentLength):
    startSample = 0
    epochID = 0
    while True:
        endSample = startSample + segmentLength
        yield source_data[startSample:endSample], epochID
        startSample += segmentLength
        epochID += 1

signals = [signal_generator(all_signal, samplingFreq * epochTime) for _ in range(chamberNum)]
# print('len(signals) =', len(signals))

dtL = [datetime.now() for _ in range(chamberNum)]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    s.connect((HOST, PORT))

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

    for chamberID in chamberIDL:

        segment, epochID = signals[chamberID].__next__()
        chamberByte = int(chamberID).to_bytes(2, 'little')
        epochByte = epochID.to_bytes(4, 'little')
        dt = dtL[chamberID]
        yearByte, monthByte, dayByte, hourByte, minuteByte, secondByte = map(lambda x: x.to_bytes(2, 'little'), (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
        microsecByte = dt.microsecond.to_bytes(4, 'little')
        datetimeByte = yearByte + monthByte + dayByte + hourByte + minuteByte + secondByte + microsecByte

        reqByte = chamberByte + epochByte + datetimeByte

        print('chamberID =', chamberID)
        print('epochID =', epochID)
        print('in mock, len(segment) =', len(segment))
        for sample in segment:
            reqByte += struct.pack('<f', sample)

        # print('in mock, reqByte =', reqByte)
        print('in mock, len(reqByte) =', len(reqByte))

        s.sendall(reqByte)
        resp = s.recv(8)

        resp_chamberID = struct.unpack_from('H', resp, 0)[0]    #WORD
        resp_epochID = struct.unpack_from('I', resp, 2)[0]    #DWORD
        resp_judge = struct.unpack_from('H', resp, 6)[0]

        dtL[chamberID] += timedelta(seconds=epochTime)
        # print('c, e, j =', resp_chamberID, resp_epochID, resp_judge)
