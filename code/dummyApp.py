#!/Users/ssg/.pyenv/shims/python
# -*- coding: utf-8 -*-
import sys
import threading
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QMainWindow
from PyQt4.QtGui import QLineEdit
from PyQt4.QtGui import QLabel
from dummyReadDaqServer import ReadDAQServer
from classifierClient import ClassifierClient
from drawGraph import DynamicGraphCanvas
from lampContainer import LampContainer

class RemApplication(QMainWindow):
    """
    GUIアプリ

    以下の二つのクラスを管理しています
    - ClassifierClient (in classifierClient.py)
    - ReadDAQServer (in readDaqServer.py)

    threadingモジュールを用いて上記のインスタンスの
    メソッドをバックグラウンドで実行しています
    """

    def __init__(self, host, port):

        # set parameters
        self.label_recordWaves = 'Record waves'
        self.label_notRecordWaves = 'Do not record waves'
        # self.current_label_for_record_waves = self.label_recordWaves
        # self.recordWaves = False
        self.current_label_for_record_waves = self.label_notRecordWaves
        self.recordWaves = True

        super(RemApplication, self).__init__()
        self.initUI()

        self.t = threading.Thread(target=self.run, daemon=True)

    def start_single_client(self):
        def to_f(inpt):
            try:
                return float(inpt)

            except Exception:
                return None

        eeg_std = to_f(self.eeg_std.text())
        emg_std = to_f(self.emg_std.text())

        statusbar = self.statusBar()

        try:
            self.client = ClassifierClient(host, port, 'single')

        except Exception as e:
            statusbar.showMessage(str(e))
            raise e

        try:
            self.server = ReadDAQServer(host, port, self.recordWaves, eeg_std=eeg_std,
                                        emg_std=emg_std, channelOpt='single')

        except Exception as e:
            statusbar.showMessage(str(e))
            raise e

        self.t.start()
        self.client.setGraph1chan(self.graph1chan)
        self.client.setGraph2chan(self.graph2chan)
        self.client.setPredictionLamp(self.predictionLamp)
        message = 'successfully started ! eeg_std is {}, and emg_std is {}'
        statusbar.showMessage(message.format(eeg_std, emg_std))   

    def start_double_client(self):
        def to_f(inpt):
            try:
                return float(inpt)

            except Exception:
                return None

        eeg_std = to_f(self.eeg_std.text())
        emg_std = to_f(self.emg_std.text())

        statusbar = self.statusBar()

        try:
            self.client = ClassifierClient(host, port, 'double')

        except Exception as e:
            statusbar.showMessage(str(e))
            raise e

        try:
            self.server = ReadDAQServer(host, port, self.recordWaves, eeg_std=eeg_std,
                                        emg_std=emg_std, channelOpt='double')

        except Exception as e:
            statusbar.showMessage(str(e))
            raise e

        self.t.start()
        self.client.setGraph1chan(self.graph1chan)
        self.client.setGraph2chan(self.graph2chan)
        self.client.setPredictionLamp(self.predictionLamp)
        message = 'successfully started ! eeg_std is {}, and emg_std is {}'
        statusbar.showMessage(message.format(eeg_std, emg_std))

    def run(self):
        print('start client')
        self.client.run()
        print('start server')
        self.server.run()

    def toggleWaveRecord(self):
        if self.recordWaves:
            self.recordWaves = False
            self.current_label_for_record_waves = self.label_recordWaves
        else:
            self.recordWaves = True
            self.current_label_for_record_waves = self.label_notRecordWaves
        self.waveRecordToggleButton.setText(self.current_label_for_record_waves)        
        self.waveRecordToggleButton.update()
        self.waveRecordToggleButton.resize(self.waveRecordToggleButton.sizeHint())
        self.show()

    def initUI(self):
        clientButton1chan = QtGui.QPushButton('Start (EEG only)', self)
        clientButton1chan.clicked.connect(self.start_single_client)
        clientButton1chan.resize(clientButton1chan.sizeHint())
        clientButton1chan.move(5, 10)

        clientButton2chan = QtGui.QPushButton('Start (EEG + EMG)', self)
        clientButton2chan.clicked.connect(self.start_double_client)
        clientButton2chan.resize(clientButton2chan.sizeHint())
        clientButton2chan.move(145, 10)

        self.nameLabel = QLabel(self)
        self.nameLabel.setText('EEG std:')
        self.nameLabel.move(10, 40)
        self.eeg_std = QLineEdit(self)
        self.eeg_std.move(80, 40)
        self.eeg_std.resize(140, 20)

        self.nameLabel = QLabel(self)
        self.nameLabel.setText('EMG std:')
        self.nameLabel.move(240, 40)
        self.emg_std = QLineEdit(self)
        self.emg_std.move(300, 40)
        self.emg_std.resize(140, 20)

        self.waveRecordToggleButton = QtGui.QPushButton(self.current_label_for_record_waves, self)
        self.waveRecordToggleButton.clicked.connect(self.toggleWaveRecord)
        self.waveRecordToggleButton.resize(self.waveRecordToggleButton.sizeHint())
        self.waveRecordToggleButton.move(10, 60)

        quitButton = QtGui.QPushButton('Quit Application', self)
        quitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        quitButton.resize(quitButton.sizeHint())
        quitButton.move(180, 60)

        self.predictionLamp = LampContainer(self)
        self.predictionLamp.move(180,110)

        self.graph1chan = DynamicGraphCanvas(self, width=4, height=2, dpi=100)
        self.graph1chan.move(20, 140)
        self.graph2chan = DynamicGraphCanvas(self, width=4, height=2, dpi=100)
        self.graph2chan.move(20, 350)

        self.setWindowTitle('Sleep Staging')
        self.resize(600, 800)
        self.show()
        self.activateWindow()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    host, port = '127.0.0.1', 50007
    mainapp = RemApplication(host, port)
    mainapp.activateWindow()
    sys.exit(app.exec_())
    # app.exec_()

