from __future__ import print_function
from freqAnalysisTools import band
import pickle
import math
import matplotlib
matplotlib.use('TkAgg')
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import random
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
# set up parameters

# for file reading
path = '../data/pickled/'

files = listdir(path)
fileIDs = []
for fileFullName in files:
    fileID, file_extension = splitext(fileFullName)
    if fileID.endswith('OPT') and file_extension == '.pkl':
        fileIDs.append(fileID)

fileID = fileIDs[random.randrange(len(fileIDs))]

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
root.wm_title(fileID)

#----------------
# compute parameters

wsizeInTimePoints = samplingFreq * wsizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.

#----------------
# read data

def readFromFile(fileID):
    fileName = open(path + fileID + '.pkl', 'rb')
    return pickle.load(fileName)

(sumPowers, normalizedPowers, sumPowersWithPast, maxPoｗers, stageColorSeq4train, sortedFreqs, sortedPowerSpect, timeSegments, eegSegmented, binnedFreqs4visIndices, stageSeq, freqs4wholeBand, binArray4spectrum) = readFromFile(fileID)

#-----------------
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

randomSelectButton = Tk.Button(master=root, text='Random select', command=_randomSelectButtonPressed, padx=0, pady=0)
quitButton = Tk.Button(master=root, text='Quit', command=_quit, padx=0, pady=0)

moveLeftButton = Tk.Button(master=root, text='<-', command=_moveLeft)
moveRightButton = Tk.Button(master=root, text='->', command=_moveRight)

#----------------
# place pull down menu

def selectFromPullDown():
    global sumPowers, normalizedPowers, sumPowersWithPast, maxPoｗers, stageColorSeq4train, sortedFreqs, sortedPowerSpect, timeSegments, eegSegmented, binnedFreqs4visIndices, stageSeq, freqs4wholeBand, binArray4spectrum
    fileID = choice_var.get()
    root.wm_title(fileID)
    (sumPowers, normalizedPowers, sumPowersWithPast, maxPoｗers, stageColorSeq4train, sortedFreqs, sortedPowerSpect, timeSegments, eegSegmented, binnedFreqs4visIndices, stageSeq, freqs4wholeBand, binArray4spectrum) = readFromFile(fileID)
    fig_global.clf()
    repaintGraphs(fig_global)
    canvas4graphs.show()
    Tk.mainloop()

choice_var = Tk.StringVar(root)
choice_var.set(fileID)
selectButton = Tk.Button(root, text="change file", command=selectFromPullDown)
pullDownMenu = Tk.OptionMenu(root, choice_var, *fileIDs)

#----------------
# place panels and pack buttons

m1 = Tk.PanedWindow()
# m1.add(canvas4graphs.get_tk_widget())
# m1.add(canvas4graphs._tkcanvas)

moveLeftButton.pack(side=Tk.LEFT)
moveRightButton.pack(side=Tk.RIGHT)

selectButton.pack(side='bottom', padx=0, pady=0)
pullDownMenu.pack(side='bottom', padx=0, pady=0)
randomSelectButton.pack(side=Tk.BOTTOM)
quitButton.pack(side=Tk.BOTTOM)
# m1.add(selectButton)
# m1.add(pullDownMenu)
# m1.add(randomSelectButton)
# m1.add(quitButton)
# m1.pack(side='bottom')

# m2 = Tk.PanedWindow(m1, orient=Tk.HORIZONTAL)

# m2.add(randomSelectButton)
# m2.add(quitButton)

scatters.figure.canvas.mpl_connect('pick_event', onPickMarker)

Tk.mainloop()
