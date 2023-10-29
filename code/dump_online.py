# -*- coding: utf-8 -*-
from parameterSetup import ParameterSetup
from dumpServer import DumpServer

class OnlineApplication:

    def __init__(self):
        self.params = ParameterSetup()

    def start(self):
        self.server = DumpServer(self.params)
        self.server.serve()

if __name__ == '__main__':
    mainapp = OnlineApplication()
    mainapp.start()
