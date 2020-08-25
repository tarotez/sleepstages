from __future__ import print_function
import pickle
import numpy as np

class FeatureConnector():

    def __init__(self):
        self.outputTensorExists = 0

    def featureConnecting(self, inputFeaturePath):
        print('inputFeaturePath = ' + inputFeaturePath)
        inputFileHandler = open(inputFeaturePath, 'rb')
        inputTensor = pickle.load(inputFileHandler)
        print('inputTensor.shape = ' + str(inputTensor.shape))
        if self.outputTensorExists:
            self.outputTensor = np.r_(outputTensor, inputTensor)
        else:
            self.outputTensor = inputTensor
            self.outputTensorExists = 1
        print('outputTensor.shape = ' + str(outputTensor.shape))
        # print('outputTensor = ' + str(outputTensor))

    def saveOutputTensor(self, outputFeaturePath):
        print('outputFeaturePath = ' + outputFeaturePath)
        outputFileHandler = open(outputFeaturePath, 'wb')
        pickle.dump(self.outputTensor, outputFileHandler)
