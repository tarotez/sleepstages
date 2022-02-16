from socket import socket, AF_INET, SOCK_STREAM
import struct
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from functools import reduce
from fileManagement import selectClassifierID
from classifierClient import ClassifierClient
from sampler import up_or_down_sampling

# format the raw array to a style accepted by classifierClient
def formatRawArray(timeStamp, samplingFreq, signal):
    formatted = ''
    timeIncrement = timedelta(seconds=1/samplingFreq)
    # print('--------------------------------')
    # print('in formatRawArray, len(signal) =', len(signal))
    # print('--------------------------------')
    for samplepoint in signal:
        formatted += str(timeStamp.time()) + '\t' + '{0:.6f}'.format(samplepoint) + '\n'
        timeStamp += timeIncrement
    return formatted

def generateClassifier(params, chamberID, observed_samplingFreq, observed_epochTime):
    # self.recordWaves = params.writeWholeWaves
    # self.extractorType = params.extractorType
    # self.finalClassifierDir = params.finalClassifierDir
    networkName = 'UTSN-L'
    classifierID, model_samplingFreq, model_epochTime = selectClassifierID(params.finalClassifierDir, networkName, observed_samplingFreq, observed_epochTime)
    # classifier_samplingFreq, classifier_epochTime = classifierMetadata(params.finalClassifierDir, classifierID)
    # check if classifier's samplingFreq and epochTime matches with requested samplingFreq and epochTime
    client = ClassifierClient(params.writeWholeWaves, params.extractorType, params.classifierType, classifierID, chamberID=chamberID, samplingFreq=model_samplingFreq, epochTime=model_epochTime)
    client.predictionStateOn()
    client.hasGUI = False
    return client

# a server that accepts tcp network connection from this or a different machine
class NetworkServer:

    def __init__(self, params_for_classifier):
        self.params_for_classifier = params_for_classifier

    def serve(self):

        PORT = 45123
        ### BUFSIZE = 10240
        BUFSIZE = 10240 * 16   # must be this big to process 1024 Hz
        networkName = 'UTSN-L'
        encode_judge = defaultdict(lambda: 3, w=0, n=1, r=2)  # for encoding judge result to a number
        ai_clients = {}

        # bind, listen, and accept a client
        tcpServSock = socket(AF_INET, SOCK_STREAM)
        tcpServSock.bind(('', PORT))
        tcpServSock.listen(1)
        # print('waiting to accept a client...')
        tcp_client, sendAddr = tcpServSock.accept()
        # print('accepted a client from', sendAddr)

        while True:
            # for setting up sampling frequency and epoch width
            received_data = tcp_client.recv(BUFSIZE)
            observed_samplingFreq = struct.unpack_from('H', received_data, 0)[0]    #WORD
            observed_epochTime = struct.unpack_from('H', received_data, 2)[0]    #WORD
            print('observed_samplingFreq =', observed_samplingFreq)
            print('observed_epochTime =', observed_epochTime)
            observed_samplePointNum = observed_samplingFreq * observed_epochTime
            fmt = reduce(lambda a, _: a + 'f', range(observed_samplingFreq * observed_epochTime), '')  # range used for unpacking EEG from received data
            classifierID, model_samplingFreq, model_epochTime = selectClassifierID(self.params_for_classifier.finalClassifierDir, networkName, observed_samplingFreq, observed_epochTime)
            model_samplePointNum = model_samplingFreq * model_epochTime

            if classifierID == -1:
                res = 0
                retByte = res.to_bytes(2, 'little')
                tcp_client.send(retByte)
                break
            else:
                res = 1
                retByte = res.to_bytes(2, 'little')
                tcp_client.send(retByte)

                while True:
                    # try:
                        received_data = tcp_client.recv(BUFSIZE)
                        # print('received_data =', received_data)
                        #print('len(received_data) =', len(received_data))

                        if len(received_data) == 0:
                            exit()

                        elif len(received_data) == 1:   # the received data is for connection check
                            # connection test (received 1 byte)
                            resp = 'Connection OK'.encode('utf-8')
                            tcp_client.send(resp)

                        elif len(received_data) == 6:   # the received data is for resetting
                            commandID = struct.unpack_from('I', received_data, 0)[0]    #DWORD
                            chamberID = struct.unpack_from('H', received_data, 4)[0]    #WORD
                            resetCommand = 901
                            if commandID == resetCommand:
                                print('resetting chamber', chamberID)
                                ai_clients[chamberID] = generateClassifier(self.params_for_classifier, chamberID, observed_samplingFreq, observed_epochTime)
                                reset_status = 1
                            else:
                                reset_status = 0
                            respByte = reset_status.to_bytes(2, 'little')
                            tcp_client.send(respByte)

                        else:
                        ### elif len(received_data) == 5142:  # the received data is signal + metadata

                            # obtain the chamber number
                            # print('### len(received_data) =', len(received_data))
                            chamberID = struct.unpack_from('H', received_data, 0)[0]    #WORD
                            # print('chamberID =', int(chamberID))

                            # obtain the epoch number
                            epochID = struct.unpack_from('I', received_data, 2)[0]    #DWORD
                            # print('epochID =', int(epochID))

                            # obtain the time that the record started
                            dt = struct.unpack_from('HHHHHHI', received_data, 6)
                            # print('%%% dt =', dt)
                            startDT = datetime(dt[0],dt[1],dt[2],dt[3],dt[4],dt[5],dt[6])
                            # print('startDT =', startDT)

                            # EEG data
                            signalW = struct.unpack_from(fmt, received_data, 22)    #float
                            #print('len(signalW) =', len(signalW))
                            signal_rawarray = np.array(signalW, dtype='float64')
                            #print('before up/down sampling, signal_rawarray.shape =', signal_rawarray.shape)

                            #print('model_samplePointNum =', model_samplePointNum)
                            #print('observed_samplePointNum =', observed_samplePointNum)

                            signal_rawarray = up_or_down_sampling(signal_rawarray, model_samplePointNum, observed_samplePointNum)
                            # print('after up/down sampling, signal_rawarray.shape =', signal_rawarray.shape)

                            # generate a new classifierClient when new chamberID comes.
                            if chamberID not in ai_clients.keys():
                                ai_clients[chamberID] = generateClassifier(self.params_for_classifier, chamberID, model_samplingFreq, observed_epochTime)

                            # Loops because classifierClients accepts segments, not full epochs, in order to visualize waves in GUI.
                            # Before the final segment, judgeStr is '-'.
                            assert model_samplingFreq % self.params_for_classifier.graphUpdateFreqInHz == 0
                            updateGraph_samplePointNum = np.int(model_samplingFreq / self.params_for_classifier.graphUpdateFreqInHz)
                            assert updateGraph_samplePointNum > 0
                            startID = 0
                            while startID < signal_rawarray.shape[0]:
                                # print('startID =', startID)
                                # print('len(signal_rawarray[startID:startID+updateGraph_samplePointNum]) =', len(signal_rawarray[startID:startID+updateGraph_samplePointNum]))
                                dataToAIClient = formatRawArray(startDT, model_samplingFreq, signal_rawarray[startID:startID+updateGraph_samplePointNum])
                                # print('in server, dataToAIClient =', dataToAIClient)
                                judgeStr = ai_clients[chamberID].process(dataToAIClient)
                                startID += updateGraph_samplePointNum

                            cByte = chamberID.to_bytes(2, 'little')
                            eByte = epochID.to_bytes(4, 'little')
                            jByte = encode_judge[judgeStr].to_bytes(2, 'little')

                            # return to the client
                            retByte = cByte + eByte + jByte
                            tcp_client.send(retByte)

                                        # except Exception as tcpException:
                #     print('Exception in the tcp connection:', tcpException)
                #     break
