from __future__ import print_function
import numpy as np
from itertools import groupby
from functools import reduce
from operator import add
from scipy import signal
from parameterSetup import ParameterSetup
from featureExtractor import FeatureExtractor
from globalTimeManagement import getTimeDiffInSeconds

class FeatureExtractorRawDataWithSTFTWithTime(FeatureExtractor):

    def __init__(self):
        params = ParameterSetup()
        self.samplingFreq = params.samplingFreq
        self.stft_time_bin_in_seconds = params.stft_time_bin_in_seconds
        self.stft_nperseg = np.int(np.floor(256.0 * self.stft_time_bin_in_seconds))
        self.extractorType = 'rawDataWithSTFTWithTime'
        self.lightPeriodStartTime = params.lightPeriodStartTime
        self.wholeBand = params.wholeBand
        self.binNum4spectrum = round(self.wholeBand.getBandWidth() / params.binWidth4freqHisto)
        # self.binArray4spectrum = np.linspace(self.wholeBand.bottom, self.wholeBand.top, self.binNum4spectrum + 1)

    def filtering(self, Zxx, freqs, lowerFreq, upperFreq):
        zipped = list(filter(lambda x: lowerFreq <= x[1] and x[1] < upperFreq, zip(Zxx, freqs)))
        return np.array([e[0] for e in zipped]), np.array([e[1] for e in zipped])

    def binning(self, Zxx, freqs, freqBinNum):
        binSize = int(np.floor(1.0 * len(Zxx) / freqBinNum))
        Zxx_binned = np.array([np.sum(np.abs(Zxx[(binID*binSize):((binID+1)*binSize)]),axis=0) for binID in range(freqBinNum)])
        freqs_binned = np.array([np.mean(freqs[(binID*binSize):((binID+1)*binSize)],axis=0) for binID in range(freqBinNum)])
        return Zxx_binned, freqs_binned

    def getFeatures(self, eegSegment, timeStampSegment, time_step):
        # compute power spectrum and sort it
        # print('eegSegment.shape =', eegSegment.shape)
        # print('self.samplingFreq =', self.samplingFreq)
        # print('self.stft_time_bin_in_seconds =', self.stft_time_bin_in_seconds)
        # print('self.stft_nperseg =', self.stft_nperseg)
        # print('self.binNum4spectrum =', self.binNum4spectrum)
        # freqs, segment_times, Zxx = signal.stft(eegSegment, fs=self.samplingFreq, nperseg=self.stft_nperseg)
        # freqs, segment_times, Zxx = signal.stft(eegSegment, fs=64)
        freqs, segment_times, Zxx = signal.stft(eegSegment, fs=self.samplingFreq, nperseg=self.stft_nperseg)
        # print('freqs.shape =', freqs.shape)
        # print('Zxx.shape =', Zxx.shape)
        # print('freqs =', freqs)
        # print('segment_times =', segment_times)
        # print('Zxx =', Zxx)
        Zxx_filtered, freqs_filtered = self.filtering(Zxx, freqs, self.wholeBand.bottom, self.wholeBand.top)
        # print('Zxx_filtered.shape =', Zxx_filtered.shape)
        Zxx_binned, freqs_binned = self.binning(Zxx_filtered, freqs_filtered, self.binNum4spectrum)
        # print('Zxx_binned.shape =', Zxx_binned.shape)
        Zxx_flattened = Zxx_binned.reshape(-1)
        #----------------
        # add time after light period started as a features
        # print('timeStampSegment[0] = ' + str(timeStampSegment[0]))
        # print('timeStampSegment[-1] = ' + str(timeStampSegment[-1]))
        timeSinceLight = getTimeDiffInSeconds(self.lightPeriodStartTime, timeStampSegment[0])
        rawDataWithSTFTWithTime = np.r_[eegSegment, Zxx_flattened, timeSinceLight]
        # print('eegSegment.shape =', eegSegment.shape)
        # print('Zxx_flattened.shape =', Zxx_flattened.shape)
        # print('rawDataWithSTFTWithTime.shape =', rawDataWithSTFTWithTime.shape)
        return rawDataWithSTFTWithTime
