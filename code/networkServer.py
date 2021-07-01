import threading
import time
from socket import *
import struct
import numpy as np
import os
import time
import datetime
from datetime import timedelta
from collections import defaultdict

# thread used to maintain a TCP connection with this or a different machine
class ServerThread(threading.Thread):
    def __init__(self, PORT=45123):
        threading.Thread.__init__(self)
        self.data = ' '
        self.kill_flag = False
        self.HOST = gethostname()
        self.BUFSIZE = 10240

        self.PORT = PORT
        # self.ADDR = (gethostbyname_ex(self.HOST)[2][1], self.PORT)  #　using the second address
        self.ADDR = ('', self.PORT)

        # bind
        self.tcpServSock = socket(AF_INET, SOCK_STREAM)
        self.tcpServSock.bind(self.ADDR) # HOST, PORTでbinding
        self.tcpServSock.listen(1)
        self.sendAddr = 0
        self.epochNo = -1
        self.chamberNo = -1

    def run(self):
        self.tcp_client, self.sendAddr = self.tcpServSock.accept()
        while True:
            try:
                data = self.tcp_client.recv(self.BUFSIZE)
                dataLength = len(data)

                if dataLength == 5142:

                    # obtain the chamber number
                    chanberNum = struct.unpack_from('H', data, 0)    #WORD
                    self.chamberNo = chanberNum[0]

                    # obtain the epoch number
                    epochNum = struct.unpack_from('I', data, 2)    #DWORD
                    self.epochNo = epochNum[0]

                    # obtain the time that the record started
                    dt = struct.unpack_from('HHHHHHI', data, 6)
                    self.startDT = datetime.datetime(dt[0],dt[1],dt[2],dt[3],dt[4],dt[5],dt[6])

                    # EEG data
                    fmt = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
                    fmt += 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'

                    dataW = struct.unpack_from(fmt, data, 22)    #float

                    self.data = dataW
                    #reply = 'OK'.encode('utf-8')
                    #self.tcp_client.send(reply)

                elif dataLength == 1:
                    # connection test (receive 1 byte)
                    reply = 'Connection OK'.encode('utf-8')
                    self.tcp_client.send(reply)

                #tcp_client.close()
            except:
                pass

# format the raw array to a style accepted by classifierClient
def formatRawArray(startDT, samplingFreq, signal):
    timeStamp = startDT
    formatted = ''
    timeIncrement = timedelta(seconds=1/samplingFreq)
    for samplepoint in signal:
        formatted += str(timeStamp.time()) + '\t' + '{0:.6f}'.format(samplepoint) + '\n'
        timeStamp += timeIncrement
    return formatted

# a server that accepts tcp network connection from this or a different machine
class NetworkServer:

    def __init__(self, ai_client, samplingFreq, graphUpdateFreqInHz):
        # start receiving thread
        self.ai_client = ai_client
        self.samplingFreq = samplingFreq
        self.updateGraph_samplePointNum = np.int(samplingFreq / graphUpdateFreqInHz)
        assert self.updateGraph_samplePointNum > 0
        self.th = ServerThread()
        self.th.setDaemon(True)
        self.th.start()

    def serve(self):

        while True:

            if self.th.data != ' ':

                startTime = time.time()
                rawarray = np.array(self.th.data, dtype='float64')

                # print( 'senderIP, port =', self.th.sendAddr )
                # print( 'chamberNum = ', self.th.chamberNo )
                # print( 'epochNum = ', self.th.epochNo )
                # print( 'startDt = ', self.th.startDT )
                # print( 'rawarray:' )
                # print( rawarray )
                # print(' rawarray.shape =', rawarray.shape)
                # print('--------------------')

                # Loops because classifierClients accepts segments, not full epochs, in order to visualize waves in GUI.
                # Before the final segment, judgeStr is '-'.
                startID = 0
                while startID < rawarray.shape[0]:
                    # print('startID =', startID)
                    dataToAIClient = formatRawArray(self.th.startDT, self.samplingFreq, rawarray[startID:startID+self.updateGraph_samplePointNum])
                    # print('in server, dataToAIClient =', dataToAIClient)
                    judgeStr = self.ai_client.process(dataToAIClient)
                    startID += self.updateGraph_samplePointNum

                # print('judgeStr =', judgeStr)
                encode_judge = defaultdict(lambda: 3, w=0, n=1, r=2)
                judge = encode_judge[judgeStr]
                # print('judge =', judge)
                # print('')

                # chamber number
                retCham = self.th.chamberNo
                cByte = retCham.to_bytes(2, 'little')

                # epoch number
                retEpoch = self.th.epochNo
                eByte = retEpoch.to_bytes(4, 'little')

                # decision result
                jByte = judge.to_bytes(2, 'little')

                retByte = cByte + eByte + jByte

                # return
                self.th.tcp_client.send(retByte)

                elapsed_time = time.time() - startTime

                self.th.data = ' '
