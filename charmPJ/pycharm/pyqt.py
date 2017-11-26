
from PyQt5 import QtWidgets

class mywindow(QtWidgets.QWidget):
    def __init__(self):
        super(mywindow, self).__init__()
        QtWidgets.QMessageBox.information(self, "Pyqt", "information")


import sys,os
import platform
import importlib
"""
if not "C:\\Program Files (x86)\\Calibre2\\app\\DLLs" in sys.path:
    sys.path.insert(0, "C:\\Program Files (x86)\\Calibre2\\app\\DLLs")  ###zhushi
print(platform.python_version())
p, err = importlib.import_module("progress_indicator"), ''
print p,err
"""
while (sys.path):
    sys.path.pop()

if not "C:\\Program Files (x86)\\Calibre2\\app\\DLLs" in sys.path:
    sys.path.insert(0, u"C:\\Program Files (x86)\\Calibre2\\app\\DLLs")  ###zhushi
print(sys.path)
import PyQt5.Qt
print PyQt5.Qt.__file__
print os.path
app = QtWidgets.QApplication(sys.argv)
windows = mywindow()
label = QtWidgets.QLabel(windows)
label.setText("hello world")
bt = QtWidgets.QPushButton(windows)
bt.setText("bt")
windows.show()
sys.exit(app.exec_())
