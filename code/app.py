#!/Users/ssg/.pyenv/shims/python
# -*- coding: utf-8 -*-
import sys
import threading
from os import listdir
import platform
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QLineEdit, QLabel, QSlider, QComboBox
from parameterSetup import ParameterSetup
import importlib
from dummyReadDaqServer import DummyReadDAQServer
from eegFileReaderServer import EEGFileReaderServer
from classifierClient import ClassifierClient
from connectionCheckClient import ConnectionCheckClient
# from drawGraph import DynamicGraphCanvas, DynamicGraphCanvas4KS, DynamicGraphCanvas4KSHist
from drawGraph import DynamicGraphCanvas
from predictionResultLabel import PredictionResultLabel
from fileManagement import filterByPrefix, getAllEEGFiles, getFileIDFromEEGFile, selectClassifierID

class RemApplication(QMainWindow):
    """
    GUI application, uses two classes
    - ClassifierClient (in classifierClient.py)
    - ReadDAQServer (in readDaqServer.py)
    """

    def __init__(self, host, port, args):

        # set parameters
        self.args = args
        self.params = ParameterSetup()
        self.label_recordWaves = 'Record original wave'
        self.label_notRecordWaves = 'No wave recording'
        self.label_overwrite = 'ON'
        self.label_notOverwrite = 'OFF'
        self.classifier_types = ['UTSN-L', 'UTSN']
        self.classifier_type = self.classifier_types[0]

        self.recordWaves = self.params.writeWholeWaves
        self.classifierType = "deep"
        self.extractorType = self.params.extractorType

        self.terminal_str_diff = "DIFF"
        self.terminal_str_rse = "RSE"
        self.terminal_str_nrse = "NRSE"
        self.terminal_str_pseudo = "PseudoDIFF"
        self.terminal_config = self.params.terminal_config_default_value

        self.channelNum = 2 if (self.params.showCh2 or self.useCh2ForReplace) else 1

        self.eeg_mode_str_normalize = "Normalize-ON"
        self.eeg_mode_str_none = "Normalize-OFF"

        self.ch2_mode_str_normalize = "Normalize-ON"
        self.ch2_mode_str_none = "Normalize-OFF"

        self.ch2_usage_str_dontshowCh2 = "Don't show Ch2"
        self.ch2_usage_str_showCh2 = "Ch2, show only"
        self.ch2_usage_str_overwrite = "Overwrite with Ch2"

        self.graphNum = 4
        self.port2 = port + 1
        self.defaultSleepTime = 1
        self.prediction_started = False

        if platform.system() == 'Windows':
            self.scale = 1.5
        else:
            self.scale = 1

        super(RemApplication, self).__init__()
        self.initUI()

    def start_reader(self):
        def to_f(inpt):
            try:
                return float(inpt)
            except Exception:
                return None

        statusbar = self.statusBar()
        try:
            # freeze pulldown menu (combobox) after prediction started
            while self.classifier_combobox.count() > 1:
                for itemID in range(self.classifier_combobox.count()):
                    if self.classifier_combobox.itemText(itemID) != self.classifier_combobox.currentText():
                        self.classifier_combobox.removeItem(itemID)
            self.prediction_started = True
            self.client.predictionStateOn()

        except Exception as e:
            print('Exception in self.client = ...')
            statusbar.showMessage(str(e))
            raise e

        self.client.setGraph(self.listOfGraphs)
        for targetChan in range(2):
            self.client.update_ylim(targetChan)
        self.client.setPredictionResult(self.listOfPredictionResults)

        try:
            if self.readFromDaq:
                module = importlib.import_module('readDaqServer')
                ReadDAQServer = getattr(module, 'ReadDAQServer')
                self.server = ReadDAQServer(self.client, self.recordWaves, self.channelNum)
            else:
                if self.args[1] == 'o':
                    postFiles = listdir(self.params.postDir)
                    inputFileName = list(filter(lambda x: not x.startswith('.'), postFiles))[0]
                    eegFilePath = self.params.postDir + '/' + inputFileName
                    self.server = EEGFileReaderServer(self.client, eegFilePath)
                else:
                    self.server = DummyReadDAQServer(self.client, self.inputFileID, self.recordWaves, self.channelNum, self.offsetWindowID, self.sleepTime)

        except Exception as e:
            print(str(e))
            statusbar.showMessage(str(e))
            raise e

        self.server.terminal_config = self.terminal_config
        self.server.serve()
        message = 'successfully started with terminal_config = ' + self.server.terminal_config
        statusbar.showMessage(message)

    def startPrediction(self):
        self.t = threading.Thread(target=self.start_reader, daemon=True)
        self.t.start()
        self.startPredictionButton.setChecked(True)

    def check_connection(self):
        statusbar = self.statusBar()
        try:
            self.check_client = ConnectionCheckClient()
            print('ConnectionCheckClient started.')
        except Exception as e:
            statusbar.showMessage(str(e))
            raise e

    '''
    def toggleOverwriteOrNot(self):
        # if self.overwriteByCh2:
        if self.client.showCh2Replace:
            self.overwriteOrNotButton.setChecked(False)
            self.overwriteOrNotButton.setText(self.label_notOverwrite)
            self.client.showCh2Replace = False
            # self.overwriteByCh2 = False
        else:
            self.overwriteOrNotButton.setChecked(True)
            self.overwriteOrNotButton.setText(self.label_overwrite)
            self.client.showCh2Replace = True
            # self.overwriteByCh2 = True
    '''

    def toggleWaveRecord(self):
        self.recordWaves = True
        self.waveRecordButton.setChecked(True)
        self.waveNotRecordButton.setChecked(False)

    def toggleWaveNotRecord(self):
        self.recordWaves = False
        self.waveRecordButton.setChecked(False)
        self.waveNotRecordButton.setChecked(True)

    def ylim_change_eeg(self):
        self.client.ylim_max_eeg = self.ylim_slider_eeg.value() / 10
        self.client.graph_ylim[0] = [-self.client.ylim_max_eeg, self.client.ylim_max_eeg]
        self.ylim_value_eeg_box.setText(str(self.client.ylim_max_eeg))
        self.client.update_ylim(targetChan=0)

    def ylim_change_ch2(self):
        self.client.ylim_max_ch2 = self.ylim_slider_ch2.value() / 10
        self.client.graph_ylim[1] = [-self.client.ylim_max_ch2, self.client.ylim_max_ch2]
        self.ylim_value_ch2_box.setText(str(self.client.ylim_max_ch2))
        self.client.update_ylim(targetChan=1)

    def ch2_thresh_change(self):
        ch2_thresh_slider_value = self.ch2_thresh_slider.value()
        self.client.ch2_thresh_value = ch2_thresh_slider_value / 4
        self.ch2_thresh.setText(str(self.client.ch2_thresh_value))

    def terminal_choice(self, text):
        self.terminal_config = text

    def eeg_mode_choice(self, text):
        self.eeg_mode = text
        self.client.eeg_mode = self.eeg_mode
        if self.eeg_mode == self.eeg_mode_str_normalize:
            self.client.eeg_normalize = 1
        else:
            self.client.eeg_normalize = 0

    def ch2_mode_choice(self, text):
        self.ch2_mode = text
        self.client.ch2_mode = self.ch2_mode
        if self.ch2_mode == self.ch2_mode_str_normalize:
            self.client.ch2_normalize = 1
        else:
            self.client.ch2_normalize = 0

    def ch2_usage_combobox_setup(self):
        if  self.params.useCh2ForReplace:
            self.ch2_usage_combobox.setCurrentText(self.ch2_usage_str_overwrite)
        elif self.params.showCh2:
            self.ch2_usage_combobox.setCurrentText(self.ch2_usage_str_showCh2)
        else:
            self.ch2_usage_combobox.setCurrentText(self.ch2_usage_str_dontshowCh2)

    def ch2_usage_choice(self, text):
        self.ch2_usage = text
        if self.ch2_usage == self.ch2_usage_str_dontshowCh2:
            self.channelNum = 1
            self.client.showCh2 = False
            self.client.showCh2Replace = False
        elif self.ch2_usage == self.ch2_usage_str_showCh2:
            self.channelNum = 2
            self.client.showCh2 = True
            self.client.useCh2ForReplace = False
        else:
            self.channelNum = 2
            self.client.showCh2 = True
            self.client.useCh2ForReplace = True

    def classifier_choice(self, text):
        print('classifier_choice() activated.')
        if not self.prediction_started:
            self.classifier_type = text
            classifierID = selectClassifierID(self.params.finalClassifierDir, self.classifier_type)
            self.client.setStagePredictor(classifierID)
            print('classifier_type changed to', self.classifier_type)

    def initUI(self):

        self.startPredictionButton = QtWidgets.QPushButton('Predict', self)
        self.startPredictionButton.setCheckable(True)
        self.startPredictionButton.clicked.connect(self.startPrediction)
        self.startPredictionButton.resize(self.startPredictionButton.sizeHint())
        self.startPredictionButton.move(5 * self.scale, 10 * self.scale)

        quitButton = QtWidgets.QPushButton('Quit', self)
        quitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        quitButton.resize(quitButton.sizeHint())
        quitButton.move(85 * self.scale, 10 * self.scale)

        checkConnectionButton = QtWidgets.QPushButton('Test connection', self)
        checkConnectionButton.clicked.connect(self.check_connection)
        checkConnectionButton.resize(checkConnectionButton.sizeHint())
        checkConnectionButton.move(160 * self.scale, 10 * self.scale)

        # change standardization of ch2
        self.nameLabel_terminal_label = QLabel(self)
        self.nameLabel_terminal_label.setText('Terminal:')
        self.nameLabel_terminal_label.move(250 * self.scale, 35 * self.scale)
        self.terminal_combobox = QtWidgets.QComboBox(self)
        self.terminal_combobox.addItem(self.terminal_str_diff)
        self.terminal_combobox.addItem(self.terminal_str_rse)
        self.terminal_combobox.addItem(self.terminal_str_nrse)
        self.terminal_combobox.addItem(self.terminal_str_pseudo)
        self.terminal_combobox.move(310 * self.scale, 38 * self.scale)
        self.terminal_combobox.resize(self.terminal_combobox.sizeHint())
        self.terminal_combobox.activated[str].connect(self.terminal_choice)
        self.terminal_combobox.setCurrentText(self.terminal_config)

        # change model
        self.nameLabel_classifier = QLabel(self)
        self.nameLabel_classifier.setText('Model:')
        self.nameLabel_classifier.move(300 * self.scale, 10 * self.scale)

        self.classifier_combobox = QtWidgets.QComboBox(self)
        for classifier_type in self.classifier_types:
            self.classifier_combobox.addItem(classifier_type)
        self.classifier_combobox.resize(self.classifier_combobox.sizeHint())
        self.classifier_combobox.move(340 * self.scale, 10 * self.scale)
        self.classifier_combobox.activated[str].connect(self.classifier_choice)

        '''
        self.nameLabel_ch2_overwrite_label = QLabel(self)
        self.nameLabel_ch2_overwrite_label.setText('Overwrite by Ch2:')
        self.nameLabel_ch2_overwrite_label.move(450 * self.scale, 5 * self.scale)

        self.overwriteOrNotButton = QtWidgets.QPushButton(self.label_notOverwrite, self)
        self.overwriteOrNotButton.clicked.connect(self.toggleOverwriteOrNot)
        self.overwriteOrNotButton.resize(self.overwriteOrNotButton.sizeHint())
        self.overwriteOrNotButton.update()
        self.overwriteOrNotButton.move(530 * self.scale, 5 * self.scale)
        self.overwriteOrNotButton.setCheckable(True)
        '''

        # change standardization of eeg
        self.nameLabel_eeg_mode_label = QLabel(self)
        self.nameLabel_eeg_mode_label.setText('EEG mode:')
        self.nameLabel_eeg_mode_label.move(610 * self.scale, 2 * self.scale)
        self.eeg_mode_combobox = QtWidgets.QComboBox(self)
        self.eeg_mode_combobox.addItem(self.eeg_mode_str_normalize)
        self.eeg_mode_combobox.addItem(self.eeg_mode_str_none)
        self.eeg_mode_combobox.move(680 * self.scale, 5 * self.scale)
        self.eeg_mode_combobox.resize(self.eeg_mode_combobox.sizeHint())
        self.eeg_mode_combobox.activated[str].connect(self.eeg_mode_choice)

        # change standardization of ch2
        self.nameLabel_ch2_mode_label = QLabel(self)
        self.nameLabel_ch2_mode_label.setText('Ch2 mode:')
        self.nameLabel_ch2_mode_label.move(610 * self.scale, 30 * self.scale)
        self.ch2_mode_combobox = QtWidgets.QComboBox(self)
        self.ch2_mode_combobox.addItem(self.ch2_mode_str_normalize)
        self.ch2_mode_combobox.addItem(self.ch2_mode_str_none)
        self.ch2_mode_combobox.move(680 * self.scale, 33 * self.scale)
        self.ch2_mode_combobox.resize(self.ch2_mode_combobox.sizeHint())
        self.ch2_mode_combobox.activated[str].connect(self.ch2_mode_choice)

        # set overwrite threshold to W
        self.nameLabel_ch2_thresh = QLabel(self)
        self.nameLabel_ch2_thresh.setText('Overwrite threshold to W:')
        self.nameLabel_ch2_thresh.resize(self.nameLabel_ch2_thresh.sizeHint())
        self.nameLabel_ch2_thresh.move(840 * self.scale, 15 * self.scale)
        self.ch2_thresh = QLineEdit(self)
        self.ch2_thresh.setText(str(self.params.ch2_thresh_default))
        self.ch2_thresh.move(1000 * self.scale, 15 * self.scale)
        self.ch2_thresh.resize(30, 20)

        # change usage of ch2
        self.nameLabel_ch2_usage_label = QLabel(self)
        self.nameLabel_ch2_usage_label.setText('Ch2 usage:')
        self.nameLabel_ch2_usage_label.move(840 * self.scale, 60 * self.scale)
        self.ch2_usage_combobox = QtWidgets.QComboBox(self)
        self.ch2_usage_combobox.addItem(self.ch2_usage_str_dontshowCh2)
        self.ch2_usage_combobox.addItem(self.ch2_usage_str_showCh2)
        self.ch2_usage_combobox.addItem(self.ch2_usage_str_overwrite)
        self.ch2_usage_combobox.move(910 * self.scale, 60 * self.scale)
        self.ch2_usage_combobox.resize(self.ch2_usage_combobox.sizeHint())
        self.ch2_usage_combobox.activated[str].connect(self.ch2_usage_choice)
        self.ch2_usage_combobox_setup()

        self.ylim_label_eeg = QLabel(self)
        self.ylim_label_eeg.setText('eeg y-max:')
        self.ylim_label_eeg.move(40 * self.scale, 40 * self.scale)
        self.ylim_label_eeg.resize(self.ylim_label_eeg.sizeHint())
        self.ylim_value_eeg_box = QLineEdit(self)
        self.ylim_value_eeg_box.move(45 * self.scale, 60 * self.scale)
        self.ylim_value_eeg_box.resize(40, 20)

        self.ylim_slider_eeg = QSlider(Qt.Vertical, self)
        self.ylim_slider_eeg.move(10 * self.scale, 40 * self.scale)
        self.ylim_slider_eeg.resize(20, 80)
        self.ylim_slider_eeg.setMinimum(0)
        self.ylim_slider_eeg.setMaximum(100)
        self.ylim_slider_eeg.setTickPosition(QSlider.TicksBelow)
        self.ylim_slider_eeg.setTickInterval(1)
        self.ylim_slider_eeg.valueChanged.connect(self.ylim_change_eeg)

        self.ylim_label_ch2 = QLabel(self)
        self.ylim_label_ch2.setText('ch2 y-max:')
        self.ylim_label_ch2.move(160 * self.scale, 40 * self.scale)
        self.ylim_label_ch2.resize(self.ylim_label_ch2.sizeHint())
        self.ylim_value_ch2_box = QLineEdit(self)
        self.ylim_value_ch2_box.move(165 * self.scale, 60 * self.scale)
        self.ylim_value_ch2_box.resize(40, 20)

        self.ylim_slider_ch2 = QSlider(Qt.Vertical, self)
        self.ylim_slider_ch2.move(130 * self.scale, 40 * self.scale)
        self.ylim_slider_ch2.resize(20, 40)
        self.ylim_slider_ch2.setMinimum(0)
        self.ylim_slider_ch2.setMaximum(100)
        self.ylim_slider_ch2.setTickPosition(QSlider.TicksBelow)
        self.ylim_slider_ch2.setTickInterval(1)
        self.ylim_slider_ch2.valueChanged.connect(self.ylim_change_ch2)

        self.ch2_thresh_slider = QSlider(Qt.Horizontal, self)
        self.ch2_thresh_slider.move(1035 * self.scale, 20 * self.scale)
        self.ch2_thresh_slider.resize(190, 20)
        self.ch2_thresh_slider.setMinimum(-8)
        self.ch2_thresh_slider.setMaximum(16)
        self.ch2_thresh_slider.setValue(4)
        self.ch2_thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.ch2_thresh_slider.setTickInterval(1)
        self.ch2_thresh_slider.valueChanged.connect(self.ch2_thresh_change)

        self.label_graph_ch2 = QLabel(self)
        self.label_graph_ch2.setFont(QtGui.QFont('SansSerif', 24))
        self.label_graph_ch2.setText('Epoch# : Prediction')
        self.label_graph_ch2.resize(self.label_graph_ch2.sizeHint())
        self.label_graph_ch2.move(500 * self.scale, 50 * self.scale)

        self.label_graph_eeg = QLabel(self)
        self.label_graph_eeg.setFont(QtGui.QFont('SansSerif', 20))
        self.label_graph_eeg.setText('EEG')
        self.label_graph_eeg.move(5 * self.scale, 205 * self.scale)

        self.label_graph_ch2 = QLabel(self)
        self.label_graph_ch2.setFont(QtGui.QFont('SansSerif', 20))
        self.label_graph_ch2.setText('Ch2')
        self.label_graph_ch2.move(5 * self.scale, 405 * self.scale)

        self.font = QtGui.QFont()
        self.font.setPointSize(18)
        self.font.setBold(True)
        self.font.setWeight(75)

        self.listOfPredictionResults = []
        self.listOfGraphs = []
        self.listOfGraphs.append([])
        self.listOfGraphs.append([])

        for graphID in range(self.graphNum):

            self.listOfPredictionResults.append(PredictionResultLabel(self))
            predXLoc = (graphID * 300) + 125
            predYLoc = 90
            self.listOfPredictionResults[graphID].move(predXLoc * self.scale, predYLoc * self.scale)

            xLoc = (graphID * 300) + 50
            for chanID in range(2):
                yLoc = (chanID * 200) + 120
                self.listOfGraphs[chanID].append(DynamicGraphCanvas(self, width = 3 * self.scale, height = 2 * self.scale, dpi=100))
                self.listOfGraphs[chanID][graphID].move(xLoc * self.scale, yLoc * self.scale)

        self.setWindowTitle('Sleep stage classifier')
        xSize = self.graphNum * 310
        ySize = 550
        self.resize(xSize * self.scale, ySize * self.scale)
        self.show()
        self.activateWindow()
        statusbar = self.statusBar()
        self.readFromDaq = False
        try:
            if len(self.args) > 5:
                print('Too many arquments for running app.py.')
                quit()
            self.sleepTime = float(self.args[4]) if len(self.args) > 4 else self.defaultSleepTime
            self.offsetWindowID = int(self.args[3]) if len(self.args) > 3 else 0
            if len(self.args) > 1:
                optionID = self.args[1]
                if optionID == 'm':
                    classifierID = selectClassifierID(self.params.finalClassifierDir, self.classifier_type)
                    self.inputFileID = self.args[2] if len(self.args) > 2 else self.randomlySelectInputFileID()
                    print('demo mode: reading inputFileID=', self.inputFileID)
                elif optionID == 'o':
                    classifierID = selectClassifierID(self.params.finalClassifierDir, self.classifier_type)
                    self.inputFileID = ''
                else:
                    classifierID = optionID
                    self.inputFileID = ''
                self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID, self.inputFileID, self.offsetWindowID)
            else:   # Neither classifierID nor inputFileID are specified.
                self.readFromDaq = True
                classifierID = selectClassifierID(self.params.finalClassifierDir, self.classifier_type)
                print('Data is read from DAQ. classifier ID is randomly selected.')
                self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID)
            self.client.hasGUI = True
            self.ylim_value_eeg_box.setText(str(self.client.ylim_max_eeg))
            self.ylim_value_ch2_box.setText(str(self.client.ylim_max_ch2))
            self.ylim_slider_eeg.setValue(self.client.ylim_max_eeg * 10)
            self.ylim_slider_ch2.setValue(self.client.ylim_max_ch2 * 10)

        except Exception as e:
            print('Exception in self.client = ...')
            statusbar.showMessage(str(e))
            raise e

    def randomlySelectInputFileID(self):
        eegFiles = getAllEEGFiles(self.params)
        return getFileIDFromEEGFile(eegFiles[np.random.randint(len(eegFiles))])

if __name__ == '__main__':
    args = sys.argv
    app = QtWidgets.QApplication(args)
    host, port = '127.0.0.1', 50007
    mainapp = RemApplication(host, port, args)
    mainapp.activateWindow()
    sys.exit(app.exec_())
