from __future__ import print_function
import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

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

stage2color = {'W':'b', 'R':'r', 'S':'k'}

#---------------
# set up parameters

# for file reading
path = '../data/RAR files/'
fileName4eeg = 'D61DBLREMOPT.txt'
fileName4stage = 'D61DBLREMOPTstage.txt'

# fileName4eeg = 'D63HETOPT.txt'
# fileName4stage = 'D63HETOPTVgr.txt'

# for data handling
metaDataLineNum4eeg = 18
metaDataLineNum4stage = 29

# for signal processing
wsizeInSec = 10   # size of window in time for estimating the state
samplingFreq = 128   # sampling frequency of data

# for training / test data extraction
# if trainWindowNum = 0, all data is used for training.
### trainWindowNum = 1500
trainWindowNum = 500
### trainWindowNum = 0

# for feature extraction
deltaBand = band(0.5, 4)
thetaBand = band(6, 9)
targetBands = (deltaBand, thetaBand)
lookBackWindowNum = 6

# for drawing spectrum
wholeBand = band(0, 16)
binWidth4freqHisto = 0.5    # bin width in the frequency domain for visualizing spectrum as a histogram
voltageRange = (-300,300)
powerRange = (0, 2 * 10 ** 8)
binnedPowerRange = (0, 2 * 10 ** 9)

# for GUI
contextSize = 5
figureSize = (15, 5)
fontSize = 12
edgecolor4selected = 'g'

markerSize4orig = 30
markerSize4selected = 50
lineWidth4orig = 0
lineWidth4selected = 10

root = Tk.Tk()
root.wm_title("EEG classifier")

#----------------
# compute parameters

wsizeInTimePoints = samplingFreq * wsizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.

#------------
# read label data (wake, REM, non-REM)

stage_fp = open(path + fileName4stage, 'r')
for i in range(metaDataLineNum4stage):    # skip 29 lines that describes metadata
    stage_fp.readline()

stagesL = []
durationWindNumsL = []
for line in stage_fp:
    line = line.rstrip()
    elems = line.split('\t')
    stagesL.append(elems[3])
    durationWindNumsL.append(elems[4])

stageSeq = []
stageColorSeq = []

for sID in range(len(stagesL)):
    repeatedStagesl = [stagesL[sID]] * int(durationWindNumsL[sID])
    repeatedColors = [stage2color[stagesL[sID]]] * int(durationWindNumsL[sID])
    stageSeq = stageSeq + repeatedStagesl
    stageColorSeq = stageColorSeq + repeatedColors

#---------------
# read eeg data

eeg_fp = open(path + fileName4eeg, 'r')
for i in range(metaDataLineNum4eeg):    # skip 18 lines that describes metadata
    eeg_fp.readline()

timestampsL = []
eegL = []
for line in eeg_fp:
    line = line.rstrip()
    elems = line.split('\t')
    timestampsL.append(elems[0].split(' ')[2].split(':')[2])
    eegL.append(elems[1])

eeg = np.array(eegL)
timestamps = np.array(timestampsL)
### samplePointNum = eeg.shape[0]

if trainWindowNum == 0:
    trainWindowNum = math.floor(eeg.shape[0] / wsizeInTimePoints)

trainSamplePointNum = trainWindowNum * wsizeInTimePoints

#---------------
# compute power spectrum and sort it

timeSegments = []
eegSegmented = []
powerSpect = np.empty((0, wsizeInTimePoints), float)   # initialize power spectrum

#----------------
# extract only for train windows
startSamplePoint = 0
while startSamplePoint < trainSamplePointNum:
    endSamplePoint = startSamplePoint + wsizeInTimePoints
    timeSegments.append(list(range(startSamplePoint, endSamplePoint)))
    eegSegmented.append(eeg[startSamplePoint:endSamplePoint])
    powerSpect = np.append(powerSpect, [np.abs(np.fft.fft(eeg[startSamplePoint:endSamplePoint])) ** 2], axis = 0)
    startSamplePoint = endSamplePoint

stageColorSeq4train = stageColorSeq[0:trainWindowNum]

# wNum = powerSpect.shape[0]
time_step = 1 / samplingFreq
freqs = np.fft.fftfreq(powerSpect.shape[1], d = time_step)
idx = np.argsort(freqs)
sortedFreqs = freqs[idx]
sortedPowerSpect = powerSpect[:,idx]
freqs4wholeBand = wholeBand.extractPowerSpectrum(sortedFreqs, sortedFreqs)

#---------------
# bin spectrum

binNum4spectrum = round(wholeBand.getBandWidth() / binWidth4freqHisto)
binArray4spectrum = np.linspace(wholeBand.bottom, wholeBand.top, binNum4spectrum + 1)
binnedFreqs4visIndices = np.digitize(freqs4wholeBand, binArray4spectrum, right=False)

#----------------
# extract total power of target bands

sumPowers = np.empty((trainWindowNum, len(targetBands)))
for wID in range(trainWindowNum):
    for bandID in range(len(targetBands)):
        sumPowers[wID,bandID] = targetBands[bandID].getSumPower(sortedFreqs, sortedPowerSpect[wID,:])

#----------------
# normalize power using total power of all bands

normalizedPowers = np.empty((trainWindowNum, len(targetBands)))
for wID in range(trainWindowNum):
    totalPower = wholeBand.getSumPower(sortedFreqs, sortedPowerSpect[wID,:])
    for bandID in range(len(targetBands)):
        normalizedPowers[wID,bandID] = sumPowers[wID,bandID] / totalPower

#----------------
# sum over past windows

sumPowersWithPast = np.empty((trainWindowNum - lookBackWindowNum, len(targetBands)))
for wID in range(trainWindowNum - lookBackWindowNum):
    for bandID in range(len(targetBands)):
        sumPowersWithPast[wID,bandID] = sumPowers[(wID - lookBackWindowNum):wID+1,bandID].sum()

#----------------
# extract max power in the target band

maxPowers = np.empty((trainWindowNum, len(targetBands)))
for wID in range(trainWindowNum):
    for bandID in range(len(targetBands)):
        maxPowers[wID,bandID] = targetBands[bandID].getMaxPower(sortedFreqs, sortedPowerSpect[wID,:])

def setAx(fig, targetLoc, title, xlabel, ylabel):
    ax = fig.add_subplot(3, 1 + contextSize, targetLoc)
    ax.set_title(title, fontsize=fontSize)
    ax.set_xlabel(xlabel, fontsize=fontSize)
    ax.set_ylabel(ylabel, fontsize=fontSize)
    return ax

#----------------
# define action for picking a marker

def onPickMarker(event):
    global selectedWID
    selectedWID = event.ind[0]
    # print('marker picked is :', selectedWID)
    fig_global.clf()
    repaintGraphs(fig_global)
    canvas4graphs.show()
    Tk.mainloop()

#---------------
# plot scatter

def showScatter(fig, wID):

    markersizeSeq = [markerSize4orig] * trainWindowNum
    markersizeSeq[wID] = markerSize4selected
    lineWidthSeq = [lineWidth4orig] * trainWindowNum
    lineWidthSeq[wID] = lineWidth4selected

    #----------------
    # scatter plot each window in the feature space using sumlax = setAx(fig, 1, 'total power', 'delta power', 'theta power')

    elemNum = len(markersizeSeq)

    # reorder markers such that the one corresponding to the selected time window (wID) comes to top.

    ax = setAx(fig, 1, 'total power', 'delta power', 'theta power')
    # ax.scatter(sumPowers[:,0], sumPowers[:,1], c=stageColorSeq, s=markersizeSeq, linewidths=lineWidthSeq, edgecolors=edgecolor4selected)
    reordered = np.array(list(range(0,trainWindowNum)) + list(range(wID+1,trainWindowNum)) + [wID])
    scatter_total_power = ax.scatter([sumPowers[i,0] for i in reordered], [sumPowers[i,1] for i in reordered], c=[stageColorSeq4train[i] for i in reordered], s=[markersizeSeq[i] for i in reordered], linewidths=[lineWidthSeq[i] for i in reordered], edgecolors=edgecolor4selected, picker=True)

    #### In above, by using sumPowers[reordered,0] and markerSizeSeq[reordered] and etc,
    #### the selected marker appears on op of other markers. 
    #### In reordered, wID should come to last front. 

    # classColors = stage2colors.values
    # for i in range(0,len(classColors)):
    #     recs.append(mpatches.Rectangle((0,0),1,1,fc=classColors[i]))
    # plt.legend(recs, classes, loc=4)

    #----------------
    # scatter plot each window in the feature space with history

    ax = setAx(fig, 2 + contextSize, 'total power with past ' + str(lookBackWindowNum) + 'windows', 'delta power', 'theta power')
    # ax.scatter(sumPowersWithPast[:,0], sumPowersWithPast[:,1], c=stageColorSeq, s=markersizeSeq, linewidths=lineWidthSeq, edgecolors=edgecolor4selected)
    reordered4withPast = np.array(list(range(0,trainWindowNum - lookBackWindowNum)) + list(range(wID+1, trainWindowNum - lookBackWindowNum)) + [wID])
    scatter_with_history = ax.scatter([sumPowersWithPast[i,0] for i in reordered4withPast], [sumPowersWithPast[i,1] for i in reordered4withPast], c=[stageColorSeq4train[i] for i in reordered4withPast], s=[markersizeSeq[i] for i in reordered4withPast], linewidths=[lineWidthSeq[i] for i in reordered4withPast], edgecolors=edgecolor4selected, picker=True)

    #----------------
    # plot each window in the feature space using max
    '''
    ax = setAx(fig, 9, 'max power', 'delta power', 'theta power')
    ax.scatter(maxPowers[:,0], maxPowers[:,1], c=stageColorSeq, s=markersizeSeq, linewidths=lineWidthSeq, edgecolors=edgecolor4selected)
    '''

    #----------------
    # scatter plot each window using normalized total power

    ax = setAx(fig, 3 + (2 * contextSize), 'normalized power', 'delta power', 'theta power')
    # ax.scatter(normalizedPowers[:,0], normalizedPowers[:,1], c=stageColorSeq, s=markersizeSeq, linewidths=lineWidthSeq, edgecolors=edgecolor4selected)
    scatter_normalized = ax.scatter([normalizedPowers[i,0] for i in reordered], [normalizedPowers[i,1] for i in reordered], c=[stageColorSeq4train[i] for i in reordered], s=[markersizeSeq[i] for i in reordered], linewidths=[lineWidthSeq[i] for i in reordered], edgecolors=edgecolor4selected, picker=True)

    return scatter_total_power
#---------------
# show context of the target window

def showSpectrum(fig, wID4visCenter):

    for add2wID in range(- math.floor(contextSize / 2), math.ceil(contextSize / 2)):

        #---------------
        # extract data for visualization

        if wID4visCenter == 0 and add2wID == -1:
            continue

        if wID4visCenter == trainWindowNum and add2wID == 1:
            continue

        wID4context = wID4visCenter + add2wID

        powerSpect4show = wholeBand.extractPowerSpectrum(sortedFreqs, sortedPowerSpect[wID4context,:])
        timePointsInSec = list(np.array(timeSegments[wID4context]) / samplingFreq)
        extractedEEG = eegSegmented[wID4context]

        histo4context = np.array([], dtype = np.float)
        for key, items in groupby (zip(binnedFreqs4visIndices, powerSpect4show), lambda i: i[0]):
            itemsL = list(items)
            itemsA = np.array(itemsL)
            powerSum = np.sum(np.array([x for x in itemsA[:,1]]))
            histo4context = np.r_[histo4context, powerSum]

        #---------------
        # plot eeg in time domain

        ax = setAx(fig, 1 + math.ceil(contextSize / 2) + add2wID, 'wID = ' + str(wID4context) + ', ' + stageSeq[wID4context], '', '')
        ax.set_ylim(voltageRange)
        ax.get_xaxis().set_visible(False)
        ax.plot(timePointsInSec, extractedEEG)
        if add2wID > - math.floor(contextSize / 2):
            ax.get_yaxis().set_visible(False)

        #---------------
        # plot power spectrum

        ax = setAx(fig, 2 + contextSize + math.ceil(contextSize / 2) + add2wID, '', 'Hz', '')
        ax.set_ylim(powerRange)
        ax.plot(freqs4wholeBand, powerSpect4show)
        if add2wID > - math.floor(contextSize / 2):
            ax.get_yaxis().set_visible(False)

        #----------------
        # plot binned power spectrum by colored bar graph

        ax = setAx(fig, 3 + (2 * contextSize) + math.ceil(contextSize / 2) + add2wID, '', 'Hz', '')
        ax.set_ylim(binnedPowerRange)
        if add2wID > - math.floor(contextSize / 2):
            ax.get_yaxis().set_visible(False)
        barL = ax.bar(binArray4spectrum[:-1], histo4context, width=0.5, edgecolor='k')
        barColors = ('y','m')
        for cID in range(len(targetBands)):
            for bID in targetBands[cID].getBarIDrange(binWidth4freqHisto):
                barL[bID].set_color(barColors[cID])
                # barL[bID].set_linewidth(1)
                barL[bID].set_edgecolor(barColors[cID])


#----------------
# generate a figure

fig_global = plt.figure(figsize=figureSize)

#-----------------
# repaint graphs

# def repaintGraphs(fig, wID):
def repaintGraphs(fig):
    global scatters
    scatters = showScatter(fig, selectedWID)
    showSpectrum(fig, selectedWID)

#---------------
# set target for visualization

def randomSelectWindow(fig):
    global selectedWID
    fig.clf()
    selectedWID = np.random.randint(trainWindowNum)   # target window for drawing time series and spectrum
    repaintGraphs(fig)

randomSelectWindow(fig_global)

#----------------
# create random select of window button

def _randomSelectButtonPressed():
    # print('random select button pressed.')
    randomSelectWindow(fig_global)
    canvas4graphs.show()
    Tk.mainloop()
    
'''
def on_key_event(event):
    print('you pressed %s' % event.key)
        key_press_handler(event, canvas1, toolbar1)

canvas1.mpl_connect('key_press_event', on_key_event)
'''

#----------------
# add a quit button

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

#----------------
# define action for pressing the right / left buttons

def _moveRight():
    global selectedWID
    # print('move right button pressed.')
    selectedWID = selectedWID + 1
    fig_global.clf()
    repaintGraphs(fig_global)
    canvas4graphs.show()
    Tk.mainloop()

def _moveLeft():
    global selectedWID
    # print('move left button pressed.')
    selectedWID = selectedWID - 1
    fig_global.clf()
    repaintGraphs(fig_global)
    canvas4graphs.show()
    Tk.mainloop()

#----------------
# draw a canvas for Tk

canvas4graphs = FigureCanvasTkAgg(fig_global, master=root)
canvas4graphs.show()
canvas4graphs.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
canvas4graphs._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

#----------------
# place buttons

randomSelectButton = Tk.Button(master=root, text='Random select', command=_randomSelectButtonPressed)
quitButton = Tk.Button(master=root, text='Quit', command=_quit)

moveLeftButton = Tk.Button(master=root, text='<-', command=_moveLeft)
moveRightButton = Tk.Button(master=root, text='->', command=_moveRight)

#----------------
# place panels and pack buttons

m1 = Tk.PanedWindow()
m1.pack(fill=Tk.BOTH, expand=1, side=Tk.BOTTOM)
# m1.add(canvas4graphs.get_tk_widget())
# m1.add(canvas4graphs._tkcanvas)

randomSelectButton.pack(side=Tk.BOTTOM)
quitButton.pack(side=Tk.BOTTOM)
moveLeftButton.pack(side=Tk.LEFT)
moveRightButton.pack(side=Tk.RIGHT)

# m2 = Tk.PanedWindow(m1, orient=Tk.HORIZONTAL)

# m2.add(randomSelectButton)
# m2.add(quitButton)

scatters.figure.canvas.mpl_connect('pick_event', onPickMarker)

Tk.mainloop()
