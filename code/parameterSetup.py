from __future__ import print_function
from freqAnalysisTools import band
import json
import numpy as np

class ParameterSetup(object):

    def __init__(self, paramDir='', paramFileName='', outputDir=''):

        self.paramSetupType = 'directories'
        pathFilePath = open('path.json')
        p = json.load(pathFilePath)
        self.pathPrefix = p['pathPrefix']
        # print('self.pathPrefix = ', self.pathPrefix)
        # directory and file name
        if paramDir == '':
            if paramFileName == '':
                paramFilePath = self.pathPrefix + '/' + p['paramsDir'] + '/params.json'
            else:
                paramFilePath = self.pathPrefix + '/' + p['paramsDir'] + '/' + paramFileName
        else:
            paramFilePath = paramDir + '/' + paramFileName

        print('in ParameterSetup, paramFilePath =', paramFilePath)
        self.parameterFileHandler = open(paramFilePath)
        d = json.load(self.parameterFileHandler)
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

        self.dataDir = self.pathPrefix + '/' + d['dataDir']
        self.pickledDir = self.pathPrefix + '/' + d['pickledDir']
        self.eegDir = self.pathPrefix + '/' + d['eegDir']
        self.featureDir = self.pathPrefix + '/' + d['featureDir']
        self.batchEvalDir = self.pathPrefix + '/' + d['batchEvalDir']
        self.standardMiceDir = self.pathPrefix + '/' + d['standardMiceDir']
        self.ksDir = self.pathPrefix + '/' + d['ksDir']
        self.finalClassifierDir = self.pathPrefix + '/' + d['finalclassifierDir']
        self.waveOutputDir = self.pathPrefix + '/' + d['wavesDir']
        self.logDir = self.pathPrefix + '/' + d['logDir']
        self.postDir = self.pathPrefix + '/' + d['postDir']

        self.classifierPrefix = d['classifierPrefix']
        self.label4withEMG = d['label4withEMG']
        self.label4withoutEMG = d['label4withoutEMG']

        # for signal processing
        self.windowSizeInSec = d['windowSizeInSec']   # size of window in time for estimating the state
        self.samplingFreq = d['samplingFreq']   # sampling frequency of data


        if 'terminalConfigDefaultValue' in d:
            self.terminal_config_default_value = d['terminalConfigDefaultValue']
        else:
            self.terminal_config_default_value = 'RSE'

        self.writeWholeWaves = d['writeWholeWaves']   # sampling frequency of data
        self.computeKS = d['computeKS']

        # for using history
        self.preContextSize = d['preContextSize']   # number of time windows in EEG to look back in time
        self.postContextSize = d['postContextSize']   # number of time windows in EEG to look back in time
        self.pastStageLookUpNum = d['pastStageLookUpNum']   # number of stage labels to look back in time

        # for wavelets
        self.waveletWidths = d['waveletWidths']

        # for using EMG
        self.useEMG = d['useEMG']
        self.emgTimeFrameNum = d['emgTimeFrameNum']

        # for making a histogram
        self.wholeBand = band(d['bandMin'], d['bandMax'])
        self.binWidth4freqHisto = d['binWidth4freqHisto']    # bin width in the frequency domain for visualizing spectrum as a histogram

        # file prefix
        self.eegFilePrefix = d['eegFilePrefix']
        self.trainDataFilePrefix = d['trainDataFilePrefix']
        self.featureFilePrefix = d['featureFilePrefix']
        self.classifierFilePrefix = d['classifierFilePrefix']

        # feature extractor
        self.extractorType = d['extractorType']
        self.lightPeriodStartTime = d['lightPeriodStartTime']

        # classifier
        self.classifierType = d['classifierType']
        self.networkType = d['networkType']
        # print('&%&%&%&% in ParameterSetup, self.networkType =', self.networkType)
        self.classifierParams = d['classifierParams']
        if self.useEMG:
            label4EMG = self.label4withEMG
        else:
            label4EMG = self.label4withoutEMG
        self.classifierName = self.classifierPrefix + '.' + label4EMG

        self.sampleClassLabels = d['sampleClassLabels']
        self.subsampleRatios = d['subsampleRatios']
        self.supersample = d['supersample']

        self.predict_by_batch = d['predict_by_batch']

        # self.replacesWWWRtoWWWW = d['replacesWWWRtoWWWW']
        self.numOfConsecutiveWsThatProhibitsR = d['numOfConsecutiveWsThatProhibitsR']

        # stride size used for prediction
        self.timeWindowStrideInSec = d['timeWindowStrideInSec']
        # self.lookBackTimeWindowNum = d['lookBackTimeWindowNum']

        self.useRawData = d['useRawData']
        self.useFreqHisto = d['useFreqHisto']
        self.useTime = d['useTime']

        if 'useSTFT' in d:
            self.useSTFT =  d['useSTFT']
        else:
            self.useSTFT = 0

        # parameters for the optimzer
        self.optimizerType = d['optimizerType']
        self.adam_learningRate = d['adam_learningRate']
        self.sgd_learningRate = d['sgd_learningRate']
        self.sgd_decay = np.float(d['sgd_decay'])
        self.sgd_momentum = d['sgd_momentum']

        # optimization parameters for deep learning
        self.deep_epochs = d['deep_epochs']
        self.deep_steps_per_epoch = d['deep_steps_per_epoch']
        self.deep_batch_size = d['deep_batch_size']

        # network structure for deep learning
        if 'torch_loss_function' in d:
            self.torch_loss_function = d['torch_loss_function']
        else:
            self.torch_loss_function = 'cross_entropy'
        self.torch_filter_nums = d['torch_filter_nums']
        self.torch_kernel_sizes = d['torch_kernel_sizes']
        self.torch_strides = d['torch_strides']
        self.torch_skip_by = d['torch_skip_by']
        self.torch_patience = d['torch_patience']
        if 'torch_lstm_length' in d:
            self.torch_lstm_length = d['torch_lstm_length']
        if 'torch_lstm_num_layers' in d:
            self.torch_lstm_num_layers = d['torch_lstm_num_layers']
        if 'torch_lstm_hidden_size' in d:
            self.torch_lstm_hidden_size = d['torch_lstm_hidden_size']
        if 'torch_lstm_inputDim' in d:
            self.torch_lstm_inputDim = d['torch_lstm_inputDim']
        if 'torch_lstm_bidirectional' in d:
            self.torch_lstm_bidirectional = d['torch_lstm_bidirectional']

        self.torch_resnet_layer_nums = d['torch_resnet_layer_nums']
        self.torch_resnet_conv_channels = d['torch_resnet_conv_channels']
        self.torch_resnet_output_channels_coeffs = d['torch_resnet_output_channels_coeffs']
        self.torch_resnet_output_channels_coeffs = d['torch_resnet_output_channels_coeffs']
        self.torch_resnet_resblock_stride_nums = d['torch_resnet_resblock_stride_nums']
        self.torch_resnet_avg_pool_size = d['torch_resnet_avg_pool_size']

        self.deep_FCN_node_nums_by_layers = d['deep_FCN_node_nums_by_layers']
        self.deep_CNN_filter_nums_by_layers = d['deep_CNN_filter_nums_by_layers']
        self.deep_CNN_kernel_sizes_by_layers = d['deep_CNN_kernel_sizes_by_layers']
        self.deep_CNN_kernel_stride_sizes_by_layers = d['deep_CNN_kernel_stride_sizes_by_layers']
        self.deep_skipConnectionLayerNum = d['deep_skipConnectionLayerNum']

        # dropoutRate
        self.dropoutRate = d['dropoutRate']

        # feature downsampling
        self.downsample_outputDim = d['downsample_outputDim']

        # features used in rawDataWithFreqHistoWithTime
        self.additionalFeatureDim = d['additionalFeatureDim']

        # markov order
        self.markovOrderForTraining = d['markovOrderForTraining']
        self.markovOrderForPrediction = d['markovOrderForPrediction']

        # number of stages to consider
        self.maximumStageNum = d['maximumStageNum']

        # maximum number of samples to be params_used
        self.maxSampleNum = d['maxSampleNum']

        # replace R to W if EMG or some other motion indicator is larger
        # by this factor, when compared to the segment having the smallest value
        # of the indicator among all past segments.
        self.useCh2ForReplace = d['useCh2ForReplace']
        self.ch2_thresh_default = d['ch2_thresh_default']
        if 'ch2IntensityFunc' in d:
            self.ch2IntensityFunc = d['ch2IntensityFunc']
        else:
            self.ch2IntensityFunc = 'max_mean'

        if 'stft_time_bin_in_seconds' in d:
            self.stft_time_bin_in_seconds =  d['stft_time_bin_in_seconds']
        else:
            self.stft_time_bin_in_seconds = 1

        if 'outputDim_cnn_for_stft' in d:
            self.outputDim_cnn_for_stft = d['outputDim_cnn_for_stft']
        else:
            self.outputDim_cnn_for_stft = 8 * 3 * 2

        # label correction (dictionary)
        self.labelCorrectionDict = {'S' : 'n', 'W' : 'w', 'R' : 'r', 'RW' : 'w', 'M' : 'm', 'P' : 'P', 'F2' : 'F2', '?' : '?', '-' : '-'}
        ### self.stageLabel2stageID = {'W': 0, 'S': 1, 'R': 2, 'M': 3, 'P': 4, 'RW': 5, 'F2' : 6}
        # self.stageLabels = ['W', 'S', 'R', 'M']
        # self.stageLabels4evaluation = ['W', 'S', 'R', 'M']
        self.capitalize_for_writing_prediction_to_file = {'n' : '1', 'w' : 'W', 'r' : 'R', 'RW' : 'RW', 'm' : 'M', 'p' : 'P', 'F2' : 'F2', '?' : '?'}
        self.capitalize_for_display = {'n' : 'NREM', 'w' : 'Wake', 'r' : 'REM', 'RW' : 'RW', 'm' : 'M', 'p' : 'P', 'F2' : 'F2', '?' : '?'}
        self.capitalize_for_graphs = {'n' : 'S', 'w' : 'W', 'r' : 'R', 'RW' : 'RW', 'm' : 'M', 'p' : 'P', 'F2' : 'F2', '?' : '?'}

        # for reading data files
        self.metaDataLineNumUpperBound4eeg = 100
        self.metaDataLineNumUpperBound4stage = 100
        self.cueWhereEEGDataStarts = 'Time'
        # self.cueWhereStageDataStarts = 'No.,Epoch'
        self.cueWhereStageDataStarts = ',,,%,%,uV^2,,uV^2'

        # ID for the classifierp
        # self.classifierID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

        orig_stageLabels = ['S', 'W', 'R', 'RW', 'M', 'P', 'F2', '?', '-']
        self.stagesByDepth = ['r', 'n', 'w', '?']
        self.stageLabel2stageID = {stage : stageID for stage, stageID in zip(orig_stageLabels[:self.maximumStageNum], range(self.maximumStageNum))}
        self.correctedLabel2depthID = {stage : stageID for stage, stageID in zip(self.stagesByDepth, range(len(self.stagesByDepth)))}
        ### self.stageLabels4evaluation = [key for key in self.stageLabel2stageID.keys()]
        self.stageLabels4evaluation = orig_stageLabels[:self.maximumStageNum]

        self.ch2_mean_init = d['ch2_mean_init']
        self.ch2_variance_init = d['ch2_variance_init']
        self.ch2_oldTotalSampleNum_init = d['ch2_oldTotalSampleNum_init']

    def stageID2stageLabel(self, stageID):
        for key in self.stageLabel2stageID.keys():
            if self.stageLabel2stageID[key] == stageID:
                return key
        return null

    def reverseLabel(self, label):
        for key in self.labelCorrectionDict:
            if self.labelCorrectionDict[key] == label:
                return key
        return ''

    def writeAllParams(self, outputDir, classifierID):
        outputDir = self.pickledDir if outputDir == '' else outputDir
        # outputPath = outputDir + '/params_used_' + self.paramSetupType + '_' + self.classifierID + '.txt'
        # outputPath = outputDir + '/parameterSetup.' + classifierID + '.json'
        # print('writing parameterSetup.__dict__ to', outputPath)
        # fileHandler = open(outputPath, 'w')
        # for key in self.__dict__:
            # fileHandler.write(key + ' : ' + str(self.__dict__[key]) + '\n')
        backupPath = outputDir + '/params.' + classifierID + '.json'
        print('copying params.json to', backupPath)
        with open(backupPath, 'w') as backupFileHandler:
            self.parameterFileHandler.seek(0)
            for line in self.parameterFileHandler:
                backupFileHandler.write(line)
            self.parameterFileHandler.seek(0)
