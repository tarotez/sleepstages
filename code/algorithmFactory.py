import importlib

class AlgorithmFactory:

    def __init__(self, params, extractorType):
        self.params = params
        self.extractorType = extractorType

    def generateExtractor(self):
        # print('@@@@@@@ in AlgorithmFactory, extractorType = ' + self.extractorType)
        if self.extractorType == 'classical':
            module = importlib.import_module('featureExtractorClassical')
            extractor = module.FeatureExtractorClassical(self.params)
        elif self.extractorType == 'freqHisto':
            module = importlib.import_module('featureExtractorFreqHisto')
            extractor = module.FeatureExtractorFreqHisto(self.params)
        elif self.extractorType == 'freqHistoWithTime':
            module = importlib.import_module('featureExtractorFreqHistoWithTime')
            extractor = module.FeatureExtractorFreqHistoWithTime(self.params)
        elif self.extractorType == 'wavelet':
            module = importlib.import_module('featureExtractorWavelet')
            extractor = module.FeatureExtractorWavelet(self.params)
        elif self.extractorType == 'wavelet-downsampled':
            module = importlib.import_module('featureExtractorWaveletDownSampled')
            extractor = module.FeatureExtractorWaveletDownSampled(self.params)
        elif self.extractorType == 'realFourier':
            module = importlib.import_module('featureExtractorRealFourier')
            extractor = module.FeatureExtractorRealFourier(self.params)
        elif self.extractorType == 'realFourier-downsampled':
            module = importlib.import_module('featureExtractorRealFourierDownSampled')
            extractor = module.FeatureExtractorRealFourierDownSampled(self.params)
        elif self.extractorType == 'complexFourier':
            module = importlib.import_module('featureExtractorComplexFourier')
            extractor = module.FeatureExtractorComplexFourier(self.params)
        elif self.extractorType == 'rawData':
            module = importlib.import_module('featureExtractorRawData')
            extractor = module.FeatureExtractorRawData(self.params)
        elif self.extractorType == 'rawDataWithFreqHistoWithTime':
            module = importlib.import_module('featureExtractorRawDataWithFreqHistoWithTime')
            extractor = module.FeatureExtractorRawDataWithFreqHistoWithTime(self.params)
        elif self.extractorType == 'rawDataWithSTFT':
            module = importlib.import_module('featureExtractorRawDataWithSTFT')
            extractor = module.FeatureExtractorRawDataWithSTFT(self.params)
        elif self.extractorType == 'rawDataWithSTFTWithTime':
            module = importlib.import_module('featureExtractorRawDataWithSTFTWithTime')
            extractor = module.FeatureExtractorRawDataWithSTFTWithTime(self.params)
        # elif self.extractorType.fine(',') > -1:
        #    module = importlib.import_module('featureExtractorMerged')
        #    print('using FeatureExtractorMerged')
        #    extractor = module.FeatureExtractorMerged(extractorType = self.extractorType)
        else:
            print('Extractor ' + self.extractorType + ' not available.')
            exit()
        return extractor
