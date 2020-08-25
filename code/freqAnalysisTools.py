import numpy as np

#---------------
# define classes and static dictionaries
 
class band:
    """frequency band"""
    def __init__(self, b, t):
        self.bottom = b
        self.top = t

    def extractPowerSpectrum(self, sortedFreqs, sortedPowerSpect):
        lowFreqs = np.extract(sortedFreqs < self.top, sortedFreqs)
        exFreqs = np.extract(self.bottom < lowFreqs, lowFreqs)
        return np.extract(self.bottom < lowFreqs, np.extract(sortedFreqs < self.top, sortedPowerSpect))

    def getSumPower(self, sortedFreqs, sortedPowerSpect):
        exPowerSpect = self.extractPowerSpectrum(sortedFreqs, sortedPowerSpect)
        return np.sum(exPowerSpect)

    def getMaxPower(self, sortedFreqs, sortedPowerSpect):
        exPowerSpect = self.extractPowerSpectrum(sortedFreqs, sortedPowerSpect)
        return np.max(exPowerSpect)

    def getBandWidth(self):
        return self.top - self.bottom

    def getBarIDrange(self, freqBinWidth):
        return range(round(self.bottom / freqBinWidth), round(self.top / freqBinWidth))
