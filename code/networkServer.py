from socket import socket, AF_INET, SOCK_STREAM
import struct
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from functools import reduce

# format the raw array to a style accepted by classifierClient
def formatRawArray(timeStamp, samplingFreq, signal):
    formatted = ''
    timeIncrement = timedelta(seconds=1/samplingFreq)
    for samplepoint in signal:
        formatted += str(timeStamp.time()) + '\t' + '{0:.6f}'.format(samplepoint) + '\n'
        timeStamp += timeIncrement
    return formatted

# a server that accepts tcp network connection from this or a different machine
class NetworkServer:

    def __init__(self, ai_client, samplingFreq, graphUpdateFreqInHz):

        self.ai_client = ai_client
        self.samplingFreq = samplingFreq
        self.updateGraph_samplePointNum = np.int(samplingFreq / graphUpdateFreqInHz)
        assert self.updateGraph_samplePointNum > 0

    def serve(self):

        PORT = 45123
        BUFSIZE = 10240
        fmt = reduce(lambda a, _: a + 'f', range(1280), '')  # range used for unpacking EEG from received data
        encode_judge = defaultdict(lambda: 3, w=0, n=1, r=2)  # for encoding judge result to a number

        # bind, listen, and accept a client
        tcpServSock = socket(AF_INET, SOCK_STREAM)
        tcpServSock.bind(('', PORT))
        tcpServSock.listen(1)
        # print('waiting to accept a client...')
        tcp_client, sendAddr = tcpServSock.accept()
        # print('accepted a client from', sendAddr)

        while True:
            try:
                received_data = tcp_client.recv(BUFSIZE)

                if len(received_data) == 5142:

                    # obtain the chamber number
                    chamberNo = struct.unpack_from('H', received_data, 0)[0]    #WORD

                    # obtain the epoch number
                    epochNo = struct.unpack_from('I', received_data, 2)[0]    #DWORD

                    # obtain the time that the record started
                    dt = struct.unpack_from('HHHHHHI', received_data, 6)
                    startDT = datetime(dt[0],dt[1],dt[2],dt[3],dt[4],dt[5],dt[6])

                    # EEG data
                    signalW = struct.unpack_from(fmt, received_data, 22)    #float
                    signal_rawarray = np.array(signalW, dtype='float64')

                    # Loops because classifierClients accepts segments, not full epochs, in order to visualize waves in GUI.
                    # Before the final segment, judgeStr is '-'.
                    startID = 0
                    while startID < signal_rawarray.shape[0]:
                        # print('startID =', startID)
                        dataToAIClient = formatRawArray(startDT, self.samplingFreq, signal_rawarray[startID:startID+self.updateGraph_samplePointNum])
                        # print('in server, dataToAIClient =', dataToAIClient)
                        judgeStr = self.ai_client.process(dataToAIClient)
                        startID += self.updateGraph_samplePointNum

                    cByte = chamberNo.to_bytes(2, 'little')
                    eByte = epochNo.to_bytes(4, 'little')
                    jByte = encode_judge[judgeStr].to_bytes(2, 'little')

                    # return to the client
                    retByte = cByte + eByte + jByte
                    tcp_client.send(retByte)

                elif len(received_data) == 1:
                    # connection test (receive 1 byte)
                    resp = 'Connection OK'.encode('utf-8')
                    tcp_client.send(resp)

            except Exception as tcpException:
                print('Exception in network connection:', tcpException)
                break
