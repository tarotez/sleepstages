import sys
from PyQt5 import QtWidgets
from app import RemApplication

if __name__ == '__main__':
    args = [sys.argv[0]] + ['o'] + sys.argv[2:]
    app = QtWidgets.QApplication(args)
    host, port = '127.0.0.1', 50007
    mainapp = RemApplication(host, port, args)
    mainapp.activateWindow()
    sys.exit(app.exec_())
