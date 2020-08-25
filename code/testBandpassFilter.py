import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bandpassfilter import butter_bandpass_filter, butter_lowpass_filter, bandpass_kaiser
from scipy.signal import freqz

# Filter requirements.
order = 3
### cutoffFreq = 8  # desired cutoffFreq frequency of the filter, Hz

cutoffFreq = 30  # desired cutoffFreq frequency of the filter, Hz

figureSize = (8, 4)
fs = 1000
timeLength = 1
nsamples = timeLength * fs
t = np.linspace(0, timeLength, nsamples, endpoint=False)

freq1 = 4
freq2 = 16
freq3 = 32
amp1 = 0.8
amp2 = 0.8
amp3 = 0.8

# lowcut = 6
# highcut = 20

wave1 = amp1 * np.sin(2 * np.pi * freq1 * t)
wave2 = amp2 * np.sin(2 * np.pi * freq2 * t)
wave3 = amp3 * np.sin(2 * np.pi * freq3 * t)

data = wave1 + wave2 + wave3

# Filter the data, and plot both the original and failtered signals.
# taps_kaiser16 = bandpass_kaiser(nsamples, lowcut, highcut, fs=fs, width=1.6)
# taps_kaiser10 = bandpass_kaiser(nsamples, lowcut, highcut, fs=fs, width=1.0)
# w, h = freqz(taps_kaiser10, 1, worN=2000)
# filtered = butter_bandpass_filter(data, lowcut, highcut, fs, order=5)
# filtered = data
filtered = butter_lowpass_filter(data, cutoffFreq, fs, order)

plt.plot(t, data)
plt.plot(t, filtered)
plt.show()


