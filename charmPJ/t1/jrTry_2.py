import tensorflow as tf
import os
import progressbar
import time

import tensorflow as tf
import numpy, pymysql.cursors, random
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from WindPy import *
import datetime, threading
from PyQt5.QtWidgets import QMessageBox


class mtry():
    def __init__(self):
        self.fields = "open,close,high,low,volume,amt,chg,pct_chg,BIAS,DMI,EXPMA,KDJ,MACD,RSI"
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
        self.connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='try',
                                          charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        self.cursor = self.connection.cursor()
        w.start()

    def findtable(self, table):
        result = self.runsql("SHOW TABLES LIKE '%s'" % table)
        if (result.__len__() == 0):
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
                                            """ % (table)
            self.cursor.execute(sql)
            return False
        result = self.runsql("SELECT * FROM `%s`" % table)
        if result.__len__() == 0:
            return False
        return True

    def mpwsi(self, codes, fields="open,close,high,low,volume,amt,chg,pct_chg,BIAS,DMI,EXPMA,KDJ,MACD,RSI",
              beginTime=datetime.datetime.now() - datetime.timedelta(days=30),
              endTime=datetime.datetime.now(), data=None):
        if (isinstance(codes, str)):
            codes = codes.split(",")
            if (codes.__len__() != 1):
                raise InterruptedError("!", u"mpwsi codes只接受一个")
                return
        if (isinstance(fields, str)):
            fields = fields.split(",")

        self.findtable(codes[0])

        try:
            while (1):
                if data != None:
                    break
                data = w.wsi(codes[0], fields, beginTime, endTime)
                # print(data)
                if (data.ErrorCode != 0):
                    raise InterruptedError("ErrorCode", u"%s" % self.ErrorCode[data.ErrorCode])
                else:
                    break
        except:
            print(u"未知网络错误")
            raise InterruptedError("!", u"未知网络错误")
            os._exit(0)

        for i in range(data.Times.__len__()):
            if data.Times[i] < beginTime:
                continue
            if data.Times[i] > endTime:
                break
            sql = "SELECT * from `%s` WHERE `time` = %s" % (codes[0], data.Times[i].strftime("'%Y-%m-%d %H:%M:%S'"))
            self.cursor.execute(sql)
            if self.cursor.fetchall().__len__() == 0:
                sql = "INSERT INTO `{0}` (`time`, `open` ,`close`,`high`,`low`,`volume`,`amount`,`change`" \
                      ",`pctchange`,`bias_bias`,`dmi_pdi`,`expma_expma`,`kdj_k`,`macd_diff`,`rsi_rsi`) VALUES " \
                      "(\'{1}\',{2[0]},{2[1]},{2[2]},{2[3]},{2[4]},{2[5]},{2[6]},{2[7]},{2[8]},{2[9]},{2[10]},{2[11]},{2[12]},{2[13]})".format(
                    codes[0], data.Times[i], numpy.array(data.Data)[:, i].tolist())
                self.cursor.execute(sql.replace("nan", "-1"))
                """                if i % 10 == 0:
                    print("+", i)"""
            else:
                """                if i % 10 == 0:
                    print(i)
                    print(self.cursor.fetchall().__len__(), self.cursor.fetchone())"""
        self.connection.commit()
        return data

    def banchinsert(self, table, data, begintime, endtime):
        if(begintime >= endtime):
            return
        sql = """INSERT INTO `{0}` (`time`, `open` ,`close`,`high`,`low`,`volume`,`amount`,`change`,`pctchange`,`bias_bias`,`dmi_pdi`,`expma_expma`,`kdj_k`,`macd_diff`,`rsi_rsi`)
              VALUES""".format(table)
        for i in range(data.Times.__len__()):
            if data.Times[i] <= begintime:
                continue
            if data.Times[i] >= endtime:
                break
            sql += """('{0}',{1[0]},{1[1]},{1[2]},{1[3]},{1[4]},{1[5]},{1[6]},{1[7]},{1[8]},{1[9]},{1[10]},{1[11]},{1[12]},{1[13]}),""".format(
                data.Times[i], numpy.array(data.Data)[:, i].tolist()).replace("nan","-1")
        if sql.endswith("VALUES"):
            return
        sql = sql[:-1] + ";"
        self.cursor.execute(sql)
        self.connection.commit()

    def msqlwsi(self, codes, ok = [True],
                fields="open,close,high,low,volume,amount,change,pctchange,bias_bias,dmi_pdi,expma_expma,kdj_k,macd_diff,rsi_rsi",
                beginTime=datetime.datetime.now() - datetime.timedelta(days=30),
                endTime=datetime.datetime.now()):
        if isinstance(fields, list):
            fields = str(fields)[1:-1].replace("'", "")
        sql = "SELECT {0} FROM `{1}` WHERE `time` >= \'{2}\' AND `time` <= \'{3}\'".format(fields, codes, beginTime,
                                                                                           endTime)
        try:
            self.cursor.execute(sql)
        except:
            ok[0] = False
            return None
        t = self.cursor.fetchall()
        data = list()
        for var in t:
            data.append(list(var.values()))

        return data

    def runsql(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def pre2inputs(self, codes, fields="open,close,high,low,volume",
                   begin=datetime.datetime(datetime.datetime.now().year, datetime.datetime.now().month,
                                           datetime.datetime.now().day) - datetime.timedelta(days=29),
                   end=datetime.datetime(datetime.datetime.now().year, datetime.datetime.now().month,
                                         datetime.datetime.now().day)):
        if (isinstance(codes, str)):
            codes = codes.split(",")
        if (isinstance(fields, str)):
            fields = fields.split(",")

        self.inputscodes = codes
        self.inputsfields = fields
        self.inputsbegin = begin
        self.inputsend = end

    def read_and_decode(self,filename_queue):
        # 输入文件名队列
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(  # 解析example
            serialized_example,
            # 必须写明features里面的key的名称
            features={
                'train/kxian': tf.FixedLenFeature([5 * 242 * 5], tf.float32),  # 图片是string类型
                'train/label': tf.FixedLenFeature([2], tf.int64),  # 标记是int64类型
            })
        '''
        # 对于BytesList，要重新进行解码，把string类型的0维Tensor变成uint8类型的一维Tensor   
        image = tf.decode_raw(features['image_raw'], tf.uint8)    
        image.set_shape([mnist.IMAGE_PIXELS])   
        # Tensor("input/DecodeRaw:0", shape=(784,), dtype=uint8)   
        # image张量的形状为：Tensor("input/sub:0", shape=(784,), dtype=float32)   
        image = tf.cast(image, tf.float32) * (1./ 255) - 0.5   
        # 把标记从uint8类型转换为int32类型  
        # label张量的形状为Tensor("input/Cast_1:0", shape=(), dtype=int32)   '''
        image = tf.cast(features['train/kxian'], tf.float32)
        label = tf.cast(features['train/label'], tf.int64)
        return image, label

    def inputs(self,train, batch_size, num_epochs):
        # 输入参数:
        # train: 选择输入训练数据/验证数据
        # batch_size: 训练的每一批有多少个样本
        # num_epochs: 过几遍数据，设置为0/None 表示永远训练下去
        """
        返回结果：A tuple (images, labels)
        * images: 类型float, 形状[batch_size, mnist.IMAGE_PIXELS]，范围[-0.5, 0.5].
        * labels： 类型int32，形状[batch_size]，范围 [0, mnist.NUM_CLASSES]
        注意tf.train.QueueRunner 必须用tf.train.start_queue_runners()来启动线程
        """
        cwd = os.getcwd()
        mdirectory = cwd + "\\mytrain\\"
        if not num_epochs: num_epochs = None
        # 获取文件路径，即/tmp/data/train.tfrecords, /tmp/data/validation.records
        TRAIN_FILE = 'mtrain.tfrecords'
        VALIDATION_FILE = 'validation.records'
        filename = os.path.join(mdirectory,
                                TRAIN_FILE if train else VALIDATION_FILE)
        with tf.name_scope('input'):
            # tf.train.string_input_producer 返回一个QueueRunner，里面有一个FIFOQueue
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=num_epochs)  # 如果样本量很大，可以分成若干文件，把文件名列表传入
        image, label = self.read_and_decode(filename_queue)
        # 随机化example，并把它们规整成batch_size 大小
        # tf.train.shuffle_batch 生成了RandomShuffleQueue，并开启两个线程
        images, sparse_labels = tf.train.shuffle_batch(
            [image[::10], label], batch_size=batch_size if train else 5000, num_threads=2,
            capacity=3000 + 3 * batch_size,
            min_after_dequeue=1000)  # 留下一部分队列，来保证每次有足够的数据做随机打乱
        return images, sparse_labels

    def getdata2db(self):
        progress = progressbar.ProgressBar(max_value=self.inputscodes.__len__())
        for i in range(self.inputscodes.__len__()):
            progress.update(i)
            for var1 in range(4):
                data = w.wsi(self.inputscodes[i], self.fields, self.inputsbegin, self.inputsend)
                if data.ErrorCode == 0:
                    break
                if var1 == 3:
                    raise ConnectionError(u"网络错误")
            if self.findtable(self.inputscodes[i]):
                t = self.runsql(
                    """SELECT DATE_FORMAT(`time`,"%%Y-%%m-%%d" ) as `time`,count(*) as `acount` From `%s` 
                    WHERE `time` >= '%s' AND `time` <= '%s' GROUP BY DATE_FORMAT(`time`,"%%Y-%%m-%%d" ) ORDER BY `time`""" % (
                        self.inputscodes[i], self.inputsbegin, self.inputsend))
                for var2 in t:
                    if var2["acount"] != 242:
                        self.mpwsi(self.inputscodes[i], self.fields,
                                   datetime.datetime.strptime(var2["time"], "%Y-%m-%d"),
                                   datetime.datetime.strptime(var2["time"], "%Y-%m-%d") + datetime.timedelta(days=1),
                                   data)
                self.banchinsert(self.inputscodes[i], data,
                                 datetime.datetime.strptime(t[-1]["time"], "%Y-%m-%d") + datetime.timedelta(days=1),
                                 self.inputsend)
            else:
                self.banchinsert(self.inputscodes[i], data,self.inputsbegin,self.inputsend)

    """    def getdata2stack(self, mutex=threading.Lock()):
        mrand = range(self.inputscodes.__len__())
        mrand = random.shuffle(mrand)
        i = 0
        self.datastck = list()
        while len(self.datastck) < 100:
            if (mutex.acquire()):
                self.datastck.append(w.wsi(self.inputscodes[mrand[i]], self.inputsfields, self, endTime))"""

    def generatetfrecords(self,filename):
        writer = tf.python_io.TFRecordWriter(filename)
        progress = progressbar.ProgressBar(max_value=self.inputscodes.__len__())
        mrand = list(range(self.inputscodes.__len__()))
        random.shuffle(mrand)
        for i in range(self.inputscodes.__len__()):
            progress.update(i)
            i = mrand[i]
            j = random.randint(0, 29 - 14)
            begintime = self.inputsbegin + datetime.timedelta(days=j)
            endtime = self.inputsend
            temp = list()
            while (begintime <= endtime - datetime.timedelta(days=1)):
                ok = [True]
                p = self.msqlwsi(self.inputscodes[i], ok, self.inputsfields, begintime,
                                 begintime + timedelta(days=1))
                if not ok[0]:#表不存在
                    break
                if isinstance(p, list) and p.__len__() == 242:
                    if p == [[-1.0] * 5] * 242:
                        break
                    temp.extend(p)
                elif isinstance(p, list) and p.__len__() == 0:
                    begintime += datetime.timedelta(days=1)
                    continue
                else:
                    raise InterruptedError(u"不应该呀")

                if temp.__len__() == 242 * 6:
                    """t = int(temp[-1-242][1] > temp[-1][1])
                    temp = numpy.array(temp)
                    temp = numpy.reshape(temp, (-1,)).tolist()


                    temp = temp[:5 * 242 * 5]

                    feature = {"train/kxian": tf.train.Feature(float_list=tf.train.FloatList(value=temp)),
                               "train/label": tf.train.Feature(int64_list=tf.train.Int64List(value=[t, 1 - t]))}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    break"""
                    t = int(temp[-1 - 242][1] > temp[-1][1])
                    temp = numpy.array(temp)
                    temp = numpy.reshape(temp, (-1,)).tolist()

                    temp = temp[:5 * 242 * 5]

                    feature = {"train/kxian": tf.train.Feature(float_list=tf.train.FloatList(value=temp)),
                               "train/label": tf.train.Feature(int64_list=tf.train.Int64List(value=[t, 1 - t]))}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    break
                begintime += datetime.timedelta(days=1)


def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    # 最多占gpu资源的70%
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # 开始不会给tensorflow全部gpu资源 而是按需增加
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # 首先导入数据，看一下数据的形式
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # print(mnist.train.images.shape)
    mytry = mtry()
    f = open("codes.txt", "r")
    codes = f.readline()
    imax = codes.split(",").__len__() - 1
    jmax = 29 - 14
    mytry.pre2inputs(codes)
    batch_size = 128
    kxian, labels = mytry.inputs(train=True, batch_size=batch_size, num_epochs=0)

    lr = 1e-3
    # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
    #batch_size = tf.placeholder(tf.int32)  # 注意类型必须为 tf.int32
    # batch_size = 128

    # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
    input_size = 5
    # 时序持续长度为28，即每做一次预测，需要先输入28行
    timestep_size = 121
    # 每个隐含层的节点数
    hidden_size = 256
    # LSTM layer 的层数
    layer_num = 16
    # 最后输出分类类别数量，如果是回归预测的话应该是 1
    class_num = 2

    _X = kxian
    y = tf.cast(labels,tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    # 把784个点的字符信息还原成 28 * 28 的图片
    # 下面几个步骤是实现 RNN / LSTM 的关键
    ####################################################################
    # **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
    # X = tf.reshape(_X, [-1, 28, 28])
    weights = {
        'wd1': tf.Variable(tf.random_normal([input_size * timestep_size, hidden_size * timestep_size])),
    }
    biases = {
        'bd1': tf.Variable(tf.random_normal([hidden_size * timestep_size])),
    }
    fc1 = tf.reshape(_X, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    X = tf.reshape(fc1, [-1, timestep_size, hidden_size])
    # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

    # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

    # **步骤5：用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, batch_size, hidden_size],
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [batch_size, hidden_size]
    # outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
    # h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    # *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
    # 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
    # **步骤6：方法二，按时间步展开计算
    outputs = list()
    state = init_state
    with tf.variable_scope('RNN'):
        with progressbar.ProgressBar(max_value=timestep_size) as progress:
            for timestep in range(timestep_size):
                progress.update(timestep)
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                # 这里的state保存了每一层 LSTM 的状态
                (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
                outputs.append(cell_output)
    h_state = outputs[-1]
    # 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
    # 首先定义 softmax 的连接权重矩阵和偏置
    # out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
    # out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
    # 开始训练和测试
    print("@")
    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
    print("@")
    # 损失和评估函数
    cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
    print("@")
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    print("@")
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("@")
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    print("@")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print("@")
    progress = progressbar.ProgressBar(max_value=2000)
    for i in range(2000):
        if coord.should_stop():
            print("break")
            break
        progress.update(i)
        # batch = mnist.train.next_batch(_batch_size)
        if (i + 1) % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={keep_prob: 1.0})
            # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
            print("Iter%d, step %d, training accuracy %g" % ((i + 1) * batch_size, (i + 1), train_accuracy))
        sess.run(train_op, feed_dict={keep_prob: 0.5})

        # 计算测试数据的准确率
        # print("test accuracy %g"% sess.run(accuracy, feed_dict={
        #    _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))



if __name__ == '__main__':
    main()
    os._exit(0)
    mytry = mtry()
    f = open("codes.txt", "r")
    codes = f.readline()
    mytry.pre2inputs(codes)
    #mytry.getdata2db()
    cwd = os.getcwd()
    mytry.generatetfrecords(cwd + "\\mytrain\\mtrain.tfrecords")

    os._exit(0)
    """"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    q = tf.RandomShuffleQueue(capacity=1000, min_after_dequeue=100, dtypes=[tf.float32, tf.int16], shapes=[(2, 2), ()])


    def inputt():
        print(1)
        return [[[1, 2], [3, 4]], 5]


    enqueue_op = q.enqueue(inputt())
    qr = tf.train.QueueRunner(q, enqueue_ops=[enqueue_op] * 1)
    sess.run(tf.global_variables_initializer())
    enqueue_threads = qr.create_threads(sess, start=True)
    while (1):
        tt = sess.run(q.dequeue_many(2))
        tt = tt[0][:, ::2]
        tt = numpy.reshape(tt, (-1, 4))
        print(tt[0][::1])
    pass
