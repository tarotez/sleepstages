from __future__ import print_function
import numpy as np
from itertools import groupby
from parameterSetup import ParameterSetup
from featureExtractor import FeatureExtractor
from globalTimeManagement import getTimeDiffInSeconds

class FeatureExtractorRawDataWithFreqHistoWithTime(FeatureExtractor):

    def __init__(self):
        params = ParameterSetup()
        self.extractorType = 'rawDataWithFreqHistoWithTime'
        self.lightPeriodStartTime = params.lightPeriodStartTime
        self.wholeBand = params.wholeBand
        binNum4spectrum = round(self.wholeBand.getBandWidth() / params.binWidth4freqHisto)
        self.binArray4spectrum = np.linspace(self.wholeBand.bottom, self.wholeBand.top, binNum4spectrum + 1)

    def getFeatures(self, eegSegment, timeStampSegment, time_step):

        #---------------
        # compute power spectrum and sort it

        powerSpect = np.abs(np.fft.fft(eegSegment)) ** 2
        freqs = np.fft.fftfreq(len(powerSpect), d = time_step)
        idx = np.argsort(freqs)
        sortedFreqs = freqs[idx]
        sortedPowerSpect = powerSpect[idx]

        # print(' ')
        # print('in getFeatures():')
        # print('time_step = ' + str(time_step))
        # print('eegSegment = ' + str(eegSegment))
        # print('powerSpect = ' + str(powerSpect))
        # print('idx = ' + str(idx))
        # print('freqs = ' + str(freqs))
        # print('sortedFreqs = ' + str(sortedFreqs))
        # print('sortedPowerSpect = ' + str(sortedPowerSpect))

        #---------------
        # bin spectrum
        freqs4wholeBand = self.wholeBand.extractPowerSpectrum(sortedFreqs, sortedFreqs)
        binnedFreqs = np.digitize(freqs4wholeBand, self.binArray4spectrum, right=False)

        #----------------
        # make a feature vector that contains context windows
        extractedPowerSpect = self.wholeBand.extractPowerSpectrum(sortedFreqs, sortedPowerSpect)
        # print('binnedFreqs = ' + str(binnedFreqs))
        # print('extractedPowerSpect = ' + str(extractedPowerSpect))

        #----------------
        # extract freqHistoWithContext
        freqHisto = np.array([], dtype = np.float)
        for key, items in groupby (zip(binnedFreqs, extractedPowerSpect), lambda i: i[0]):
            itemsA = np.array(list(items))
            powerSum = np.sum(np.array([x for x in itemsA[:,1]]))
            freqHisto = np.r_[freqHisto, powerSum]

        #----------------
        # add time after light period started as a features
        # print('timeStampSegment[0] = ' + str(timeStampSegment[0]))
        # print('timeStampSegment[-1] = ' + str(timeStampSegment[-1]))

        timeSinceLight = getTimeDiffInSeconds(self.lightPeriodStartTime, timeStampSegment[0])
        rawDataWithFreqHistoWithTime = np.r_[eegSegment, freqHisto, timeSinceLight]

        return rawDataWithFreqHistoWithTime
