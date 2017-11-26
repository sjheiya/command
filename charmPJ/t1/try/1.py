import threading,progressbar,time,pymysql,datetime

def t(v):
    while(1):
        print(v)

def main():
    progress = progressbar.ProgressBar(max_value=100)
    for i in range(100):
        progress.update(i)
        time.sleep(0.01)
    progress.finish()

    print("123")
    pass



    """t1 = threading.Thread(target=t,args=("3"))
    t1.setDaemon(True)
    t1.start()

    while(1):
        print("2")"""
def trydb():
    connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='try',
                                      charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    cursor = connection.cursor()
    cursor.execute("""SELECT DATE_FORMAT(`time`,"%%Y-%%m-%%d" ) as `time`,count(*) as `acount` From `%s` 
                    WHERE `time` >= '%s' AND `time` <= '%s' GROUP BY DATE_FORMAT(`time`,"%%Y-%%m-%%d" ) ORDER BY `time`""" % (
                        "000001.SZ", datetime.datetime.now() - datetime.timedelta(days=29), datetime.datetime.now()))
    t = cursor.fetchall()
    print(t)

def changeint(bool):
    bool[0] += 1

if __name__ == '__main__':
    var = 1
    changeint([var])
    print(var)
    pass