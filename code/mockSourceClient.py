import sys
import socket
import struct
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import random

# sends mock signal to online.py by connectin to networkServer

args = sys.argv
if len(args) > 1:
    HOST = args[1]
else:
    HOST = '192.168.0.3'  # The server's hostname or IP address

# print('HOST:', HOST)
PORT = 45123       # The port used by the server

sampleNum = 100
chamberIDL = (random.randint(0,3) for _ in range(sampleNum))
epochDict = {}

### samplingFreq = 128
samplingFreq = 512
epochTime = 10
signal = [float(i + 3.1416) for i in range(samplingFreq * epochTime)]

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

    for samples in range(sampleNum):

        chamberID = chamberIDL.__next__()

        if chamberID in epochDict.keys():
            epochDict[chamberID] += 1
        else:
            epochDict[chamberID] = 0
        epochID = epochDict[chamberID]

        chamberByte = chamberID.to_bytes(2, 'little')
        epochByte = epochID.to_bytes(4, 'little')

        dt = datetime.now()
        yearByte, monthByte, dayByte, hourByte, minuteByte, secondByte = map(lambda x: x.to_bytes(2, 'little'), (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
        microsecByte = dt.microsecond.to_bytes(4, 'little')
        datetimeByte = yearByte + monthByte + dayByte + hourByte + minuteByte + secondByte + microsecByte

        reqByte = chamberByte + epochByte + datetimeByte

        for sample in signal:
            reqByte += struct.pack('<f', sample)
        # signalByte = signal.to_bytes(5120, 'little')
        # signalByte = reduce(lambda a, x: a + x, map(lambda x: x.to_bytes(4, 'little'), signal))
        # reqByte = chamberByte + epochByte + datetimeByte + signalByte

        # print('in mock, reqByte =', reqByte)
        print('in mock, len(reqByte) =', len(reqByte))

        s.sendall(reqByte)
        resp = s.recv(8)

        resp_chamberID = struct.unpack_from('H', resp, 0)[0]    #WORD
        resp_epochID = struct.unpack_from('I', resp, 2)[0]    #DWORD
        resp_judge = struct.unpack_from('H', resp, 6)[0]

        print('c, e, j =', resp_chamberID, resp_epochID, resp_judge)
