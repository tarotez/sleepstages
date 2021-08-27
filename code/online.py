# -*- coding: utf-8 -*-
from parameterSetup import ParameterSetup
from networkServer import NetworkServer

class OnlineApplication:

    def __init__(self):
        self.params = ParameterSetup()

    def start(self):
        self.server = NetworkServer(self.params.samplingFreq, self.params.graphUpdateFreqInHz, self.params)
        self.server.serve()

if __name__ == '__main__':
    mainapp = OnlineApplication()
    mainapp.start()
