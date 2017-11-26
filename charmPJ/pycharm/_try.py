from WindPy import *
import datetime, os
import pymysql.cursors, numpy


class mtry():
    def __init__(self):
        self.connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='try',
                                          charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        self.cursor = self.connection.cursor()
        w.start()

    def mpwsi(self, codes,
              fields="open,close,high,low,volume,amount,change,pctchange,bias_bias,dmi_pdi,expma_expma,kdj_k,macd_diff,rsi_rsi",
              beginTime=datetime.datetime.now() - datetime.timedelta(days=30),
              endTime=datetime.datetime.now()):
        if (isinstance(codes, str)):
            codes = codes.split(",")
            if (codes.__len__() != 1):
                pass  # messagebox
                return
        if (isinstance(fields, str)):
            fields = fields.split(",")

            sql = "SHOW TABLES LIKE '%s'" % codes[0]
            self.cursor.execute(sql)
        if (self.cursor.fetchall().__len__() == 0):
            sql = sql = """
                                CREATE TABLE `%s`
                                (
                                `time` DATETIME NOT NULL,
                                `open` FLOAT NOT NULL,
                                `close` FLOAT NOT NULL,
                                `high` FLOAT NOT NULL,
                                `low` FLOAT NOT NULL,
                                `volume` FLOAT NOT NULL,
                                `amount` FLOAT NOT NULL,
                                `change` FLOAT NOT NULL,
                                `pctchange` FLOAT NOT NULL,
                                `bias_bias` FLOAT NOT NULL,
                                `dmi_pdi` FLOAT NOT NULL,
                                `expma_expma` FLOAT NOT NULL,
                                `kdj_k` FLOAT NOT NULL,
                                `macd_diff` FLOAT NOT NULL,
                                `rsi_rsi` FLOAT NOT NULL,
                                PRIMARY KEY (`time`)
                                )
                                """ % (codes[0])
            self.cursor.execute(sql)
        try:
            while (1):
                data = w.wsi(codes, fields, beginTime, endTime)
                if (data.ErrorCode != 0):
                    pass  # messagebox
                else:
                    break
        except:
            print(u"未知网络错误")
            pass  # messagebox
            os._exit(0)

        for i in range(data.Times.__len__()):
            sql = "SELECT * from `%s` WHERE `time` = %s" % (codes[0], data.Times[i].strftime("'%Y-%m-%d %H:%M:%S'"))
            self.cursor.execute(sql)
            if self.cursor.fetchall().__len__() == 0:
                sql = "INSERT INTO `{0}` (`time`, `open` ,`close`,`high`,`low`,`volume`,`amount`,`change`" \
                      ",`pctchange`,`bias_bias`,`dmi_pdi`,`expma_expma`,`kdj_k`,`macd_diff`,`rsi_rsi`) VALUES " \
                      "(\'{1}\',{2[0]},{2[1]},{2[2]},{2[3]},{2[4]},{2[5]},{2[6]},{2[7]},{2[8]},{2[9]},{2[10]},{2[11]},{2[12]},{2[13]})".format(
                    codes[0], data.Times[i], numpy.array(data.Data)[:, i].tolist())
                self.cursor.execute(sql.replace("nan", "0"))
            else:
                print(i)
                print(self.cursor.fetchall().__len__(), self.cursor.fetchone())
        self.connection.commit()
        return data

    def msqlwsi(self, codes,
                fields="open,close,high,low,volume,amount,change,pctchange,bias_bias,dmi_pdi,expma_expma,kdj_k,macd_diff,rsi_rsi",
                beginTime=datetime.datetime.now() - datetime.timedelta(days=30),
                endTime=datetime.datetime.now()):
        sql = "SELECT {0} FROM `{1}` WHERE `time` >= \'{2}\' AND `time` <= \'{3}\'".format(fields, codes, beginTime,
                                                                                           endTime)
        self.cursor.execute(sql)
        data = list()
        for var in self.cursor.fetchall():
            data.append(list(var.values()))
        return data


def main():
    """w.start()
    data = w.wsi("600000.SH", "open,high,low,close,volume,amt,chg,pct_chg,BIAS,DMI,EXPMA,KDJ,MACD,RSI",
                 "2017-11-21 09:00:00", "2017-11-21 22:14:38")
    print(data)"""
    mt = mtry()
    data = mt.msqlwsi("600000.SH", "open,close,high,low,volume", datetime.datetime.now() - datetime.timedelta(days=1))
    data = numpy.array(data)
    print(data)
    print(data[0][0])


if __name__ == '__main__':
    main()
