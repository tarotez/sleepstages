import threading
import time
import socket
import struct
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import random

# sends mock signal to online.py by connectin to networkServer

HOST = '192.168.0.2'  # The server's hostname or IP address
PORT = 45123       # The port used by the server

sampleNum = 100
chamberIDL = (random.randint(0,3) for _ in range(sampleNum))
epochDict = {}
signal = [float(i + 3.1416) for i in range(1280)]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    s.connect((HOST, PORT))

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
        # print('in mock, len(reqByte) =', len(reqByte))

        s.sendall(reqByte)
        resp = s.recv(8)

        resp_chamberID = struct.unpack_from('H', resp, 0)[0]    #WORD
        resp_epochID = struct.unpack_from('I', resp, 2)[0]    #DWORD
        resp_judge = struct.unpack_from('H', resp, 6)[0]

        print('c, e, j =', resp_chamberID, resp_epochID, resp_judge)
