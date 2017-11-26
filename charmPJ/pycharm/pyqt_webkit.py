from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import *
import sys,os

class PythonJS(QObject):
    def alert(self, msg):
        self.emit(pyqtSignal('contentChanged(const QString &)'), msg)

    def message(self):
        return "pyClick!!"

app = QApplication([])
view = QWebEngineView()
view.setHtml("""
           <script>function message() { return "Clicked!"; }</script>
           <script type"text/javascript" src="http://localhost:8080/123.js"></script>
           <h1>QtWebKit + Python sample program</h1>
           <input type="button" value="Click JavaScript!" 
                  onClick="alert('[javascript] ' + message1());"/>
           <input type="button" value="Click Python!" 
                  onClick="self.alert('[python] ' +
                                        python.message())"/>
           <br />
           <iframe src="http://www.baidu.com/"
                   width="750" height="500"
                   scrolling="no"
                   frameborder="0"
                   align="center"></iframe>
        """)
file_object = open("123.js")
try:
     all_the_text = file_object.read( )
finally:
     file_object.close( )
view.page().runJavaScript(all_the_text)
view.show()
app.exec_()

