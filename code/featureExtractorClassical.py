from __future__ import print_function
import numpy as np
from freqAnalysisTools import band
from featureExtractor import FeatureExtractor

class FeatureExtractorClassical(FeatureExtractor):

    def __init__(self, params):
        self.params = params
        self.extractorType = 'classical'

    def getFeatures(self, eegSegment, timeStampSegment, time_step, local_mu, local_sigma):

        targetBand = band(1, 12)
        deltaBand = band(2.5, 3.5)
        thetaBand = band(7.0, 8.0)
        wideThetaBand = band(6.5, 8.0)
        alphaBand = band(9.0, 9.5)
        smallDelta = 0.000000001

        #---------------
        # compute power spectrum and sort it
        powerSpect = np.abs(np.fft.fft(eegSegment)) ** 2
        freqs = np.fft.fftfreq(len(powerSpect), d = time_step)
        idx = np.argsort(freqs)
        sortedFreqs = freqs[idx]
        sortedPowerSpect = powerSpect[idx]

        # print(' ')
        # print('in featureExtractorClassical.getFeaturesClassical():')
        # print(' time_step = ' + str(time_step))
        # print(' eegSegment = ' + str(eegSegment))
        # print(' powerSpect = ' + str(powerSpect))
        # print(' idx = ' + str(idx))
        # print(' freqs = ' + str(freqs))
        # print(' sortedFreqs = ' + str(sortedFreqs))
        # print(' sortedPowerSpect = ' + str(sortedPowerSpect))

        null = 0
        cv = local_sigma / (np.abs(local_mu) + smallDelta)
        # cv = local_sigma
        integral = targetBand.getSumPower(sortedFreqs, sortedPowerSpect)
        deltaPower = deltaBand.getSumPower(sortedFreqs, sortedPowerSpect)
        thetaPower = thetaBand.getSumPower(sortedFreqs, sortedPowerSpect)
        alphaPower = alphaBand.getSumPower(sortedFreqs, sortedPowerSpect)
        wideThetaPower = wideThetaBand.getSumPower(sortedFreqs, sortedPowerSpect)
        deltaRatio = deltaPower / (deltaPower + thetaPower + alphaPower + smallDelta)
        thetaRatio = thetaPower / (deltaPower+ smallDelta)
        ### return np.array([cv, integral, deltaRatio, deltaPower, alphaPower, thetaPower])
        return np.array([cv, integral, deltaRatio, deltaRatio, thetaRatio, deltaPower, integral, alphaPower, thetaPower, thetaPower, integral])
        ### return np.array([cv, null, integral, deltaRatio, deltaRatio, integral, deltaPower, integral, alphaPower, thetaPower, thetaPower, integral])
        ### return np.array([cv, integral, deltaRatio, deltaRatio, cv, integral, deltaPower, integral, alphaPower, thetaPower, thetaPower, integral])
        ### return np.array([cv, null, integral, deltaRatio, deltaRatio, thetaPower, deltaPower, integral, alphaPower, thetaPower, thetaPower, integral])
