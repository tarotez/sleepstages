from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

class PredictionResultLabel(QtWidgets.QLabel):

    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self,parent)
        self.labelContent = '-'
        self.stageColors = [Qt.black, Qt.red, Qt.darkCyan]
        self.stageCode = 0

        self.setText(self.labelContent)
        # label2 = QtWidgets.QLabel(widget)
        # label3 = QtWidgets.QLabel(widget)

        # vbox = QtGui.QHBoxLayout(widget)
        # for w in [resLabel, label2, label3]
        # for w in [self.resLabel]:
            # vbox.addWidget(w)
        # widget.setLayout(vbox)

    def setChoice(self, segmentID, choice, choiceLabel):
        if choice == 'Wake':
            self.stageCode = 2
        elif choice == 'REM':
            self.stageCode = 1
        elif choice == 'NREM':
            self.stageCode = 0
        else:
            self.stageCode = 0
        # self.setLabel('Epoch ' + str(segmentID + 1) + ' : ' + choiceLabel + ' ', self.stageCode)
        self.setLabel(str(segmentID + 1) + ' : ' + choiceLabel + ' ', self.stageCode)

    def setLabel(self, labelContent, stageCode):
        self.labelContent = labelContent
        self.stageCode = stageCode
        # print('in setLabel, labelContent = ' + self.labelContent)
        self.setText(self.labelContent)
        self.setFont(QtGui.QFont('SansSerif', 24, QtGui.QFont.Bold))
        self.resize(self.sizeHint())
        p = self.palette()
        p.setColor(self.foregroundRole(), self.stageColors[self.stageCode])
        self.setPalette(p)
        self.update()

    def getLabel(self):
        return self.labelContent

    def getStageCode(self):
        return self.stageCode
