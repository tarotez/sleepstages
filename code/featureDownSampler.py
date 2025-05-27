from __future__ import print_function
import pickle
import numpy as np

class FeatureDownSampler():

    def __init__(self):
        pass

    def featureDownSampling(self, inputFeaturePath, outputFeaturePath, outputDim):
        print('inputFeaturePath = ' + inputFeaturePath)
        print('outputFeaturePath = ' + outputFeaturePath)
        inputFileHandler = open(inputFeaturePath, 'rb')
        inputTensor = pickle.load(inputFileHandler)
        outputTensor = self.downSample(inputTensor, outputDim)
        outputFileHandler = open(outputFeaturePath, 'wb')
        pickle.dump(outputTensor, outputFileHandler)

    def downSample(self, inputTensor, outputDim):
        # print('inputTensor.shape = ' + str(inputTensor.shape))
        inputDim = inputTensor.shape[-1]
        if inputDim == outputDim:
            # print('not downsampling')
            outputTensor = inputTensor
        else:
            poolingSize = int(np.floor(np.float(inputDim/outputDim)))
            poolingStrideSize = poolingSize
            # print('poolingSize = ' + str(poolingSize))
            # downsample by appling max(arg()) to regions
            outputTensor = np.zeros((inputTensor.shape[0], inputTensor.shape[1], outputDim))
            for outputIDstart in range(outputDim):
                inputIDstart = outputIDstart * poolingStrideSize
                inputIDs = range(inputIDstart, inputIDstart+poolingSize-1)
                # print('outputIDstart = ' + str(outputIDstart) + ', inputIDs = ' + str(inputIDs))
                outputTensor[:,:,outputIDstart] = np.max(inputTensor[:,:,inputIDs], axis=-1)
                # outputTensor[:,:,outputIDstart] = np.mean(np.abs(inputTensor[:,:,inputIDs]), axis=-1)
        # print('outputTensor.shape = ' + str(outputTensor.shape))
        # print('outputTensor = ' + str(outputTensor))
        return outputTensor
