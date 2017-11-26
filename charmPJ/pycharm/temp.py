from WindPy import w
import datetime
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import *
from PyQt5.QtGui import QStandardItemModel,QStandardItem
import myjr
import threading
from time import ctime,sleep
import pymysql.cursors
import os
"""
w.start();
print(w.isconnected())
data=w.wsq("600000.SH,000001.SZ","rt_last,rt_last_vol")

while(True):
    print(data)
"""

class MainWindow(QWidget, myjr.Ui_Form):
    def __init__(self, datatypes ,parent=None):
        super(QWidget, self).__init__(parent)
        self.setupUi(self)
        self.t1 = QTimer()
        self.pushButton.clicked.connect(self.ok_bt_onclick)
        self.t1.timeout.connect(self.mshow1)
        self.flag = 0
        self.orderName = list()
        w.start(waitTime=30)
        print(w.isconnected())

        self.initmodel()

        self.rt_susp_flag = {
            '0':u"正常",
            '1': u"停1h",
            '2': u"停2h",
            '3': u"停半天",
            '4': u"停下午",
            '5': u"停半小时",
            '6': u"临时停牌",
            '9': u"停牌一天",
        }
        self.rt_trade_status = {
            '0': u"无状态/状态未知",
            '1': u"正常交易中",
            '2': u"休市中/暂停交易",
            '3': u"已收盘/当日交易结束",
            '5': u"暂停交易（深交所临时停牌）",
            '4': u"集合竞价中",
            '8': u"盘前交易 PreMarket",
            '9': u"盘后交易 AfterMarket",
            '10': u"期权波动性中断",
            '11': u"可恢复交易的熔断",
            '12': u"不可恢复交易的熔断",
        }
        self.ErrorCode = {
            -40520001: u"未知错误",
            -40520002: u"内部错误",
            -40520003: u"系统错误",
            -40520004: u"登录失败",
            -40520005: u"无权限",
            -40520006: u"用户取消",
            -40520007: u"无数据",
            -40520008: u"超时错误",
            -40520009: u"本地WBOX错误",
            -40520010: u"需要内容不存在",
            -40520011: u"需要服务器不存在",
            -40520012: u"引用不存在",
            -40520013: u"其他地方登录错误",
            -40520014: u"未登录使用WIM工具，故无法登录",
            -40520015: u"连续登录失败次数过多",
            -40521001: u"IO操作错误",
            -40521002: u"后台服务器不可用",
            -40521003: u"网络连接失败",
            -40521004: u"请求发送失败",
            -40521005: u"数据接收失败",
            -40521006: u"网络错误",
            -40521007: u"服务器拒绝请求",
            -40521008: u"错误的应答",
            -40521009: u"数据解码失败",
            -40521010: u"网络超时",
            -40521011: u"频繁访问",
            -40522001: u"无合法会话",
            -40522002: u"非法数据服务",
            -40522003: u"非法请求",
            -40522004: u"万得代码语法错误",
            -40522005: u"不支持的万得代码",
            -40522006: u"指标语法错误",
            -40522007: u"不支持的指标",
            -40522008: u"指标参数语法错误",
            -40522009: u"不支持的指标参数",
            -40522010: u"日期与时间语法错误",
            -40522011: u"不支持的日期与时间",
            -40522012: u"不支持的请求参数",
            -40522013: u"数组下标越界",
            -40522014: u"重复的WQID",
            -40522015: u"请求无相应权限",
            -40522016: u"不支持的数据类型",
            -40522017: u"数据提取量超限",
        }
        assert(type(datatypes) == str)



        self.datatypes = datatypes;
        #create threading
        self.mutex = threading.Lock()
        ##
        self.ok_bt_onclick()  # init model
        ##
        threads = []
        t1 = threading.Thread(target=self.refreshdata)
        threads.append(t1)
        for t in threads:
            t.setDaemon(True)
            t.start()
        # 连接MySQL数据库
        self.connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='try',
                                     charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        self.cursor = self.connection.cursor()
        ##
        sleep(3)
        self.t1.start(100)

    def initmodel(self):
        self.model = QStandardItemModel()
        self.tableView.setModel(self.model)
        self.tableView.setColumnWidth(0, 140)

    def mshow1(self):

        #print(data)
        data = self.begin()
        if(not data):
            return

        for i in range(data.Data.__len__()): #列
            for j in range(data.Codes.__len__()): #行
                self.showtable(data,i,j)


    def showtable(self,data,i,j,offi = 0,offj = 0):
        if (data.Fields[i] == "rt_susp_flag".upper() and str(int(data.Data[i][j]))[-1] != '0' and self.model.item(j + offj, i + offi)
            and self.model.item(j + offj, i + offi).text().find(u"正常") != -1):
            QMessageBox.information(self, "!", u"%s停了" % self.model.verticalHeaderItem(j).text(), QMessageBox.Yes)
        if (self.model.item(j + offj, i + offi)):
            self.model.item(j + offj, i + offi).setText("%.4f" % data.Data[i][j])
            if (data.Fields[i] == "rt_susp_flag".upper()):
                self.model.item(j + offj, i + offi).setText("%s%d" % (self.rt_susp_flag[str(int(data.Data[i][j]))[-1]], data.Data[i][j]))
            elif (self.model.horizontalHeaderItem(i).text() == "rt_trade_status"):
                self.model.item(j + offj, i + offi).setText("%s%d" % (self.rt_trade_status[str(int(data.Data[i][j]))], data.Data[i][j]))
        else:
            self.model.setItem(j + offj, i + offi, QStandardItem("%.4f" % data.Data[i][j]))

    def mshow2(self):
        data = self.begin()
        if (not data):
            return
        for i in range(data.Data.__len__()): #列
            for j in range(0,data.Codes.__len__(),2): #行
                self.showtable(data,i,j,1,-j/2)
                self.showtable(data,i,j + 1,data.Fields.__len__() + 2,-(j + 1)/2)

    def begin(self):
        if (self.mutex.acquire(1)):
            data = self.data
            self.mutex.release()
        else:
            QMessageBox.information(self, "!", u"不应该呀", QMessageBox.Yes)
            return None
        if(data.Times.__len__()):
            self.label.setText(str(data.Times[0]).split(".")[0])
        if(data.Data.__len__() == 0):
            return None
        return data

    def refreshdataOnce(self):
        data = self.mwsq(self.order, self.datatypes)
        if (data.ErrorCode != 0):
            print(data.ErrorCode)
            print(self.ErrorCode[data.ErrorCode])
            QMessageBox.information(self, "ErrorCode", u"%s" % self.ErrorCode[data.ErrorCode], QMessageBox.Yes)
            os._exit()
        if (self.mutex.acquire(5)):
            self.data = data
            self.mutex.release()
        else:
            QMessageBox.information(self, "!", u"不应该呀,获取互斥锁失败", QMessageBox.Yes)
            return

    def refreshdata(self):
        while(True):
            data = self.mwsq(self.order, self.datatypes)
            if (data.ErrorCode != 0):
                print(data.ErrorCode)
                print(self.ErrorCode[data.ErrorCode])
                QMessageBox.information(self, "ErrorCode", u"%s" % self.ErrorCode[data.ErrorCode], QMessageBox.Yes)
                os._exit()
            if(self.mutex.acquire(5)):
                self.data = data
                self.mutex.release()
                if(not self.flag == 0):
                    continue
                i = 0;
                for var in data.Codes:
                    sql = "SHOW TABLES LIKE '%s'" % var
                    self.cursor.execute(sql)
                    if(self.cursor.fetchall().__len__() == 0):
                        sql = sql = """
                        CREATE TABLE `%s`
                        (
                        `id` INT(11) NOT NULL AUTO_INCREMENT,
                        `time` DATETIME NOT NULL,
                        `rt_last` DOUBLE NOT NULL,
                        `rt_last_vol` DOUBLE NOT NULL,
                        PRIMARY KEY (`id`)
                        )
                        """ % (var)
                        self.cursor.execute(sql)
                    try:
                        mindex = data.Fields.index("rt_trade_status".upper())
                        if(mindex != -1 and data.Data[mindex][i] == 1):
                            try:
                                sql = "INSERT INTO `%s` (`time`, `rt_last` ,`rt_last_vol`) VALUES (\'%s\',%f,%f)" % (
                                    var,datetime.datetime.now(), data.Data[0][i], data.Data[1][i])
                                self.cursor.execute(sql)
                            except:
                                QMessageBox.information(self, "!", u"网络错误", QMessageBox.Yes)
                    except:
                        pass
                    i += 1
                self.connection.commit()
            else:
                QMessageBox.information(self, "!", u"不应该呀,获取互斥锁失败" , QMessageBox.Yes)
                return

    def mwsq(self,order,datatypes):
        for tempvar in range(3):
            try:
                if (order.__len__() * datatypes.split(",").__len__() <= 100):
                    data = w.wsq(order, datatypes)
                    if(data.ErrorCode != 0):
                        return data
                    else:
                        continue

                incre = int(100 / datatypes.split(",").__len__())
                data = w.wsq(order[0:incre], datatypes)
                for var in range(incre, order.__len__(), incre):
                    tempdata = w.wsq(order[var:(var + incre) if (var + incre <= order.__len__()) else order.__len__()],
                                     datatypes)
                    data.Codes.extend(tempdata.Codes)
                    for i in range(data.Data.__len__()):
                        data.Data[i].extend(tempdata.Data[i])
                    if (tempdata.ErrorCode != 0):
                        QMessageBox.information(self, "!", u"网络错误:ErrorCode=%d" % tempdata.ErrorCode, QMessageBox.Yes)
                        return self.mwsq(order,datatypes)
                    if (tempdata.Times[0] - data.Times[0] > datetime.timedelta(seconds=15)):
                        QMessageBox.information(self, "!", u"网络延迟过大", QMessageBox.Yes)
                        return self.mwsq(order,datatypes)
                return data
            except:
                # QMessageBox.information(self, "!", u"未知网络错误", QMessageBox.Yes)
                print(u"未知网络错误")
                return self.mwsq(order,datatypes)

    def ok_bt_onclick(self):
        if(self.comboBox.currentIndex() == 0):
            order = self.textEdit.toPlainText().__str__()
            self.order = order.split("\n")
            self.datatypes = "rt_last,rt_last_vol,rt_susp_flag"
            self.model.clear()
            self.model.setHorizontalHeaderLabels(self.datatypes.split(","))
            self.model.setVerticalHeaderLabels(self.order)
            self.flag = 0
            self.t1.stop()
            self.t1.disconnect()
            self.t1.timeout.connect(self.mshow1)
            self.refreshdataOnce()

            self.t1.start(100)
        elif(self.comboBox.currentIndex() == 1):
            self.orderName = list()
            self.order = list()
            order = self.textEdit.toPlainText().__str__()
            order = order.strip("\n").split("\n")
            self.datatypes = "rt_last,rt_last_vol,rt_susp_flag"
            templist = list()
            templist.append("code")
            templist.extend(self.datatypes.split(","))
            templist.append("code")
            templist.extend(self.datatypes.split(","))
            self.model.clear()
            self.model.setHorizontalHeaderLabels(templist)
            try:
                for var in order:
                    self.order.append(var.split(",")[1])
                    self.order.append(var.split(",")[2])
                    self.orderName.append(var.split(",")[0])
                self.model.setVerticalHeaderLabels(self.orderName)
                for var in range(0,self.order.__len__(),2):
                    self.model.setItem(var / 2, 0, QStandardItem("%s" % self.order[var]))
                    self.model.setItem(var / 2, self.datatypes.split(",").__len__() + 1, QStandardItem("%s" % self.order[var + 1]))
            except:
                QMessageBox.information(self, "!", u"数据格式错误", QMessageBox.Yes)
                os._exit(0)

            self.flag = 1
            self.t1.stop()
            self.t1.disconnect()
            self.t1.timeout.connect(self.mshow2)
            self.refreshdataOnce()
            self.t1.start(100)



app = QApplication([])
window = MainWindow("rt_last,rt_last_vol,rt_susp_flag")

window.show()

app.exec_()
