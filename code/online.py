# -*- coding: utf-8 -*-
from parameterSetup import ParameterSetup
from networkServer import NetworkServer

class OnlineApplication:

    def __init__(self):
        self.params = ParameterSetup()
        # self.classifierType = 'deep'
        # self.networkType = 'UTSN-L'

    def start(self):
        # self.recordWaves = params.writeWholeWaves
        # self.extractorType = params.extractorType
        # self.finalClassifierDir = params.finalClassifierDir
        # classifierID = selectClassifierID(self.finalClassifierDir, self.networkType)
        # self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID)
        # self.client.predictionStateOn()
        # self.client.hasGUI = False
        self.server = NetworkServer(self.params.samplingFreq, self.params.graphUpdateFreqInHz, self.params)
        self.server.serve()


if __name__ == '__main__':
    mainapp = OnlineApplication()
    mainapp.start()
