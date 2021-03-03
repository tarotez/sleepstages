#!/Users/ssg/.pyenv/shims/python
# -*- coding: utf-8 -*-
import sys
import time
from os import listdir
from os.path import split, splitext, isfile
from parameterSetup import ParameterSetup
from classifierClient import ClassifierClient
from eegFileReaderServer import EEGFileReaderServer
from fileManagement import selectClassifierID

class RemOfflineApplication:

    def __init__(self, args):
        self.args = args
        self.classifier_type = 'UTSN-L'
        pass

    def start(self):
        channelOpt = 1
        params = ParameterSetup()
        self.recordWaves = params.writeWholeWaves
        self.extractorType = params.extractorType
        self.classifierType = params.classifierType
        self.postDir = params.postDir
        self.predDir = params.predDir
        self.finalClassifierDir = params.finalClassifierDir

        # eegFilePath = args[1]
        # inputFileID = splitext(split(eegFilePath)[1])[0]
        postFiles = listdir(self.postDir)
        fileCnt = 0
        for inputFileName in postFiles:
            if not inputFileName.startswith('.'):
                print('inputFileName = ' + inputFileName)
                inputFileID = splitext(inputFileName)[0]
                print('inputFileID = ' + inputFileID)
                predFileFullPath = self.predDir + '/' + inputFileID + '_pred.txt'
                print('predFileFullPath = ' + predFileFullPath)

                if not isfile(predFileFullPath):
                    fileCnt += 1
                    print('  processing ' + inputFileID)
                    try:
                        classifierID = selectClassifierID(self.finalClassifierDir, self.classifier_type)
                        self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID)
                        ### self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID, inputFileID=inputFileID)
                        self.client.predictionStateOn()
                        self.client.hasGUI = False
                        # sys.stdout.write('classifierClient started by ' + str(channelOpt) + ' channel.')

                    except Exception as e:
                        print(str(e))
                        raise e

                    try:
                        eegFilePath = self.postDir + '/' + inputFileName
                        self.server = EEGFileReaderServer(self.client, eegFilePath)

                    except Exception as e:
                        print(str(e))
                        raise e

                else:
                    print('  skipping ' + inputFileID)


if __name__ == '__main__':
    args = sys.argv
    mainapp = RemOfflineApplication(args)
    mainapp.start()
    # while True:
        # print('*')
        # time.sleep(5)
    # sys.exit(app.exec_())
