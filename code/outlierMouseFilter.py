from __future__ import print_function
from os.path import splitext
from parameterSetup import ParameterSetup

class OutlierMouseFilter(object):

    def __init__(self):
        params = ParameterSetup()
        pickledDir = params.pickledDir
        self.fileIDs_L = []
        try:
            outlierHandler = open(pickledDir + '/' + 'outliers.txt', 'r')
            # print(outlierHandler)
            for line in outlierHandler:
                elems = line.split(' ')
                parts = elems[2].split('-')
                fileID = parts[2] + '-' + parts[3] + '-' + parts[4]
                # print(fileID)
                self.fileIDs_L.append(fileID)
        except EnvironmentError:
            pass

    def notOutlier(self, fileID):
        if fileID in self.fileIDs_L:
            return False
        else:
            return True

    def isOutlier(self, fileID):
        if fileID in self.fileIDs_L:
            return True
        else:
            return False

    def printOutliers(self):
        for outlier in self.fileIDs_L:
            print(outlier)
