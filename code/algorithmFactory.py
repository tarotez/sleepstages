import importlib

class AlgorithmFactory:

    def __init__(self, extractorType):
        self.extractorType = extractorType

    def generateExtractor(self):
        # print('@@@@@@@ in AlgorithmFactory, extractorType = ' + self.extractorType)
        if self.extractorType == 'classical':
            module = importlib.import_module('featureExtractorClassical')
            extractor = module.FeatureExtractorClassical()
        elif self.extractorType == 'freqHisto':
            module = importlib.import_module('featureExtractorFreqHisto')
            extractor = module.FeatureExtractorFreqHisto()
        elif self.extractorType == 'freqHistoWithTime':
            module = importlib.import_module('featureExtractorFreqHistoWithTime')
            extractor = module.FeatureExtractorFreqHistoWithTime()
        elif self.extractorType == 'wavelet':
            module = importlib.import_module('featureExtractorWavelet')
            extractor = module.FeatureExtractorWavelet()
        elif self.extractorType == 'wavelet-downsampled':
            module = importlib.import_module('featureExtractorWaveletDownSampled')
            extractor = module.FeatureExtractorWaveletDownSampled()
        elif self.extractorType == 'realFourier':
            module = importlib.import_module('featureExtractorRealFourier')
            extractor = module.FeatureExtractorRealFourier()
        elif self.extractorType == 'realFourier-downsampled':
            module = importlib.import_module('featureExtractorRealFourierDownSampled')
            extractor = module.FeatureExtractorRealFourierDownSampled()
        elif self.extractorType == 'complexFourier':
            module = importlib.import_module('featureExtractorComplexFourier')
            extractor = module.FeatureExtractorComplexFourier()
        elif self.extractorType == 'rawData':
            module = importlib.import_module('featureExtractorRawData')
            extractor = module.FeatureExtractorRawData()
        elif self.extractorType == 'rawDataWithFreqHistoWithTime':
            module = importlib.import_module('featureExtractorRawDataWithFreqHistoWithTime')
            extractor = module.FeatureExtractorRawDataWithFreqHistoWithTime()
        elif self.extractorType == 'rawDataWithSTFT':
            module = importlib.import_module('featureExtractorRawDataWithSTFT')
            extractor = module.FeatureExtractorRawDataWithSTFT()            
        elif self.extractorType == 'rawDataWithSTFTWithTime':
            module = importlib.import_module('featureExtractorRawDataWithSTFTWithTime')
            extractor = module.FeatureExtractorRawDataWithSTFTWithTime()
        # elif self.extractorType.fine(',') > -1:
        #    module = importlib.import_module('featureExtractorMerged')
        #    print('using FeatureExtractorMerged')
        #    extractor = module.FeatureExtractorMerged(extractorType = self.extractorType)
        else:
            print('Extractor ' + self.extractorType + ' not available.')
            exit()
        return extractor
