from __future__ import unicode_literals
import sys
import os
import random
from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtWidgets, QtCore
else:
    from PyQt5 import QtWidgets, QtCore
# import matplotlib
# Make sure that we are using QT5
# matplotlib.use('Qt5Agg')
# from PyQt5 import QtCore, QtWidgets
import numpy as np
from numpy import arange, sin, pi
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class GraphCanvas(FigureCanvas):

    def __init__(self, parent=None, width=4, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.set_facecolor('#ececec')
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
               QtWidgets.QSizePolicy.Expanding,
               QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.wave = np.array([0])
        # self.color = 'b'

class DynamicGraphCanvas(GraphCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        GraphCanvas.__init__(self, *args, **kwargs)
        '''
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)
        '''

    '''
    def update_figure(self):
        self.axes.cla()
        waveLen = self.wave.shape[0]
        self.axes.plot(np.linspace(0, waveLen-1, waveLen), self.wave, self.color)
        self.axes.set_ylim([-4,4])
        self.axes.set_xticks([400, 800, 1200])
        self.draw()

    def setData(self, wave, color):
        # print('***** drawGraph.setData : ' + str(wave))
        self.wave = wave
        self.color = color
        '''

    def setData(self, wave, color, graph_ylim):
        # print('***** drawGraph.setData : ' + str(wave))
        self.axes.cla()
        self.wave = wave
        self.color = color
        waveLen = wave.shape[0]
        self.axes.plot(np.linspace(0, waveLen-1, waveLen), self.wave, self.color, linewidth=0.5)
        self.axes.set_ylim(graph_ylim)
        self.axes.set_xticks([640, 1280])
        self.axes.set_xticklabels(['5s', '10s'])
        # self.axes.set_facecolor('#ececec')
        # self.axes.set_xlabel('sec')
        self.draw()

    def getData(self):
        if hasattr(self, 'wave'):
            return self.wave
        else:
            return np.array([])

class DynamicGraphCanvas4KS(DynamicGraphCanvas):

    def __init__(self, *args, **kwargs):
        DynamicGraphCanvas.__init__(self, *args, **kwargs)
        self.axes.set_xticks([])
        self.axes.set_yticks([])

    def setData4KS(self, dSeriesArray, colors):
        self.axes.cla()
        ylim = [dSeriesArray.min() * 0.8, dSeriesArray.max() * 1.2]
        self.axes.set_ylim(ylim)
        self.axes.set_xticks([])
        cID = 0
        # print('dSeriesArray.shape = ' + str(dSeriesArray.shape))
        dSeriesLen = dSeriesArray.shape[0]
        for dSeries in dSeriesArray.transpose():
            # print('dSeries = ' + str(dSeries))
            self.axes.plot(np.linspace(0, dSeriesLen-1, dSeriesLen), dSeries, colors[cID])
            cID += 1
        self.draw()

class DynamicGraphCanvas4KSHist(DynamicGraphCanvas):

    def __init__(self, *args, **kwargs):
        DynamicGraphCanvas.__init__(self, *args, **kwargs)
        self.axes.set_xticks([])
        self.axes.set_yticks([])

    def setData4KSHist(self, dMat, newd):
        self.axes.cla()
        dHistVals, bins, patches = self.axes.hist(dMat.reshape(-1), bins=20, color='blue')
        # print('dHistVal = ' + str(dHistVals))
        dHistMax = np.max(dHistVals)
        standardMiceBarHeight = dHistMax * 1.2
        # mean_newd = np.mean(newd)
        # verticalLine = np.repeat(mean_newd, dHistMax)
        verticalLine = np.repeat(newd, standardMiceBarHeight)
        self.axes.hist(verticalLine, bins=20, color='red')
        self.axes.set_ylim([0, standardMiceBarHeight])
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.draw()
