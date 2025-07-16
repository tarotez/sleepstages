from __future__ import print_function
import numpy as np
from itertools import groupby
from featureExtractor import FeatureExtractor

class FeatureExtractorFreqHisto(FeatureExtractor):

    def __init__(self, params):
        self.params = params
        self.extractorType = 'freqHisto'

    def getFeatures(self, eegSegment, timeStampSegment, time_step, local_mu, local_sigma):

        #---------------
        # compute power spectrum and sort it
        params = self.params
        wholeBand = params.wholeBand
        binWidth4freqHisto = params.binWidth4freqHisto
        binNum4spectrum = round(wholeBand.getBandWidth() / binWidth4freqHisto)
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
        binArray4spectrum = np.linspace(wholeBand.bottom, wholeBand.top, binNum4spectrum + 1)
        freqs4wholeBand = wholeBand.extractPowerSpectrum(sortedFreqs, sortedFreqs)
        binnedFreqs = np.digitize(freqs4wholeBand, binArray4spectrum, right=False)

        #----------------
        # make a feature vector that contains context windows
        extractedPowerSpect = wholeBand.extractPowerSpectrum(sortedFreqs, sortedPowerSpect)
        # print('binnedFreqs = ' + str(binnedFreqs))
        # print('extractedPowerSpect = ' + str(extractedPowerSpect))

        #----------------
        # extract freqHistoWithContext
        freqHisto = np.array([], dtype = np.float)
        for key, items in groupby (zip(binnedFreqs, extractedPowerSpect), lambda i: i[0]):
            itemsA = np.array(list(items))
            powerSum = np.sum(np.array([x for x in itemsA[:,1]]))
            freqHisto = np.r_[freqHisto, powerSum]

        return freqHisto
