import pymysql.cursors
from WindPy import w
import datetime

# 连接MySQL数据库
connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='try',
                             charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
cursor = connection.cursor()
w.start()
print(w.isconnected())

data = w.wsq("000001.SZ,USDX.FX", "rt_last,rt_last_vol")
#sql = "INSERT INTO `data` (`time`, `rt_last` ,`rt_last_vol`) VALUES (\'%s\',%f,%f)" % (datetime.datetime.now(),data.Data[0][0],data.Data[1][0])
#sql = "SELECT * FROM `data` WHERE `time` > \'%s\'" % (datetime.datetime.now() + datetime.timedelta(seconds=-3600))
#sql = "SHOW TABLES LIKE 'data'"
sql = """
                        CREATE TABLE `abc`
                        (
                        `id` INT(11) NOT NULL AUTO_INCREMENT,
                        `time` DATETIME NOT NULL,
                        `rt_last` DOUBLE NOT NULL,
                        `rt_last_vol` DOUBLE NOT NULL,
                        PRIMARY KEY (`id`)
                        )
                        """
cursor.execute(sql)
print(cursor.fetchall())
connection.commit()






connection.close()