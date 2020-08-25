from __future__ import print_function
from freqAnalysisTools import band
import json

class ParameterSetupForNetwork(ParameterSetup):

    def __init__(self, paramDir='', paramFileName='', outputDir=''):

        self.paramSetupType = 'network'
        pathFilePath = open('path.json')
        p = json.load(pathFilePath)
        self.pathPrefix = p['pathPrefix']
        # directory and file name
        if paramDir == '':
            parameterFileHandler= open(self.pathPrefix + '/' + p['paramsDir'] + '/paramsForNetwork.json')
        else:
            parameterFileHandler = open(paramDir + '/' + paramFileName)

        d = json.load(parameterFileHandler)
        if outputDir == '':
            self.classifierDir = self.pathPrefix + '/' + d['classifierDir']
            self.deepParamsDir = self.pathPrefix + '/' + d['deepParamsDir']
            self.predDir = self.pathPrefix + '/' + d['predDir']
            self.modelDirRoot = d['modelDirRoot']
        else:
            self.classifierDir = outputDir
            self.deepParamsDir = outputDir
            self.predDir = outputDir
            self.modelDirRoot = outputDir

        # feature extractor
        self.extractorType = d['extractorType']

        # classifier
        self.networkType = d['networkType']

        print('#$#$#$#$#$#$#$# in ParameterSetupForNetwork, self.networkType =', self.networkType)

        self.classifierParams = d['classifierParams']
        if self.useEMG:
            label4EMG = self.label4withEMG
        else:
            label4EMG = self.label4withoutEMG
        self.classifierName = self.classifierPrefix + '.' + label4EMG

        # optimization parameters for deep learning
        self.deep_epochs = d['deep_epochs']
        self.deep_steps_per_epoch = d['deep_steps_per_epoch']
        self.deep_batch_size = d['deep_batch_size']

        # kernel and pooling size
        self.deep_kernelSize = d['deep_kernelSize']
        self.deep_kernelSize = d['deep_kernelStrideSize']
        self.deep_poolingSize = d['deep_poolingSize']
        self.deep_poolingStrideSize = d['deep_poolingStrideSize']

        # feature downsampling
        self.downsample_outputDim = d['downsample_outputDim']

        # network structure for deep learning
        self.deep_FCN_node_nums_by_layers = d['deep_FCN_node_nums_by_layers']
        self.deep_CNN_filter_nums_by_layers = d['deep_CNN_filter_nums_by_layers']
        self.deep_CNN_kernel_sizes_by_layers = d['deep_CNN_kernel_sizes_by_layers']
        self.deep_CNN_kernel_stride_sizes_by_layers = d['deep_CNN_kernel_stride_sizes_by_layers']
