import sys,os
while (sys.path):
    sys.path.pop()
if not "C:\\Python27_32\\Lib" in sys.path:
    sys.path.insert(0, "C:\\Python27_32\\Lib")  ###zhushi
if not "C:\\Program Files (x86)\\Calibre2\\app\\DLLs" in sys.path:
    sys.path.insert(0, u"C:\\Program Files (x86)\\Calibre2\\app\\DLLs")  ###zhushi
print(sys.modules)
import PyQt5.Qt
reload(PyQt5.Qt)
print PyQt5.Qt.__file__
print os.path
s = "1"
s.find("u")