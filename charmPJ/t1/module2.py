import tensorflow as tf
import input_data
import os
import progressbar
import time

q = tf.FIFOQueue(1,"float")
counter = tf.Variable(0.0)
increment_op = tf.assign_add(counter,tf.constant(1.0))
enqueue_op = q.enqueue([counter])

qr = tf.train.QueueRunner(q,enqueue_ops = [increment_op,enqueue_op] * 1)
coord = tf.train.Coordinator()
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    enqueue_threads = qr.create_threads(sess,coord = coord, start = True)

    for i in range(10):
        print(sess.run(q.dequeue()))
    
    coord.request_stop()
    coord.join(enqueue_threads)
'''
class FLAGS:
    pass
FLAGS.directory = "tmp/data/"
FLAGS.learning_rate = 0.0001
FLAGS.num_epochs = 100
FLAGS.batch_size = 100

def main():   
    # 获取数据
    data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)

    
    # 将数据转换为tf.train.Example类型，并写入TFRecords文件  
    convert_to(data_sets.train, 'train')  
    convert_to(data_sets.validation, 'validation')  
    convert_to(data_sets.test, 'test')

def convert_to(data_set, name):   
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples # 55000个训练数据，5000个验证数据，10000个测试数据  
    if images.shape[0] != num_examples:     
        raise ValueError('Images size %d does not match label size %d.' %                      
                         (images.shape[0], num_examples))   
    '''
    rows = images.shape[1] # 28
    cols = images.shape[2] # 28   
    depth = images.shape[3] # 1，是黑白图像，所以是单通道
    '''#因为新版mnist已经将图片展平,即images.shap[1] == 784 ，所以手动定义图片行列数
    rows = 28
    cols = 28   
    depth = 1 #是黑白图像，所以是单通道

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')   
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    ###进度条
    widgets = [progressbar.Percentage(),
               ' ', progressbar.Bar(),
               ' ', progressbar.AdaptiveETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=num_examples).start()
    for index in range(num_examples):     
        image_raw = images[index].tostring()   
        # 写入协议缓冲区中，height、width、depth、label编码成int64类型，image_raw编码成二进制   
        # 类似json数据结构 
        example = tf.train.Example(features=tf.train.Features(feature={                                
            'height': _int64_feature(rows),                                
            'width': _int64_feature(cols),                                
            'depth': _int64_feature(depth),                                
            'label': tf.train.Feature(float_list=tf.train.FloatList(value=labels[index].tolist())),                                
            'image_raw': tf.train.Feature(float_list=tf.train.FloatList(value=images[index].tolist()))
            }))  
        writer.write(example.SerializeToString()) # 序列化为字符串  
        pbar.update(index)
    writer.close()
    pbar.finish()
        
def _int64_feature(value):   
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) 
def _bytes_feature(value):   
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_and_decode(filename_queue): 
    # 输入文件名队列  
    reader = tf.TFRecordReader() 
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example( # 解析example      
        serialized_example,      
        # 必须写明features里面的key的名称     
        features={        
            'image_raw': tf.FixedLenFeature([784], tf.float32), # 图片是string类型
            'label': tf.FixedLenFeature([10], tf.float32),  # 标记是int64类型
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
    image = tf.cast(features['image_raw'], tf.float32)
    label = tf.cast(features['label'], tf.float32)
    return image, label

def inputs(train, batch_size, num_epochs):
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
    if not num_epochs: num_epochs = None
    # 获取文件路径，即/tmp/data/train.tfrecords, /tmp/data/validation.records
    TRAIN_FILE = 'train.tfrecords'
    VALIDATION_FILE = 'validation.records'
    filename = os.path.join(FLAGS.directory,
                            TRAIN_FILE if train else VALIDATION_FILE)
    with tf.name_scope('input'):
        # tf.train.string_input_producer 返回一个QueueRunner，里面有一个FIFOQueue
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs) # 如果样本量很大，可以分成若干文件，把文件名列表传入
    image, label = read_and_decode(filename_queue)
    # 随机化example，并把它们规整成batch_size 大小
    # tf.train.shuffle_batch 生成了RandomShuffleQueue，并开启两个线程
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size if train else 5000, num_threads=2,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)#留下一部分队列，来保证每次有足够的数据做随机打乱
    return images, sparse_labels

def inputs2(train, batch_size, num_epochs):
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
    if not num_epochs: num_epochs = None
    # 获取文件路径，即/tmp/data/train.tfrecords, /tmp/data/validation.records
    TRAIN_FILE = 'train.tfrecords'
    VALIDATION_FILE = 'validation.records'
    filename = os.path.join(FLAGS.directory,
                            TRAIN_FILE if train else VALIDATION_FILE)
    with tf.name_scope('input'):
        # tf.train.string_input_producer 返回一个QueueRunner，里面有一个FIFOQueue
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs) # 如果样本量很大，可以分成若干文件，把文件名列表传入
    image, label = read_and_decode(filename_queue)
    # 随机化example，并把它们规整成batch_size 大小
    # tf.train.shuffle_batch 生成了RandomShuffleQueue，并开启两个线程
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size if train else 5000, num_threads=2,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000) # 留下一部分队列，来保证每次有足够的数据做随机打乱
    return images, sparse_labels

def run_training():
    with tf.Graph().as_default():
        # 输入images 和labels
        images, labels = inputs(train=True, batch_size=FLAGS.batch_size,num_epochs=1)
        #测试数据
        data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)
        '''# 构建一个从推理模型来预测数据的图
        logits = mnist.inference(images,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2)
        loss = mnist.loss(logits, labels) # 定义损失函数
        # Add to the Graph operations that train the model.
        train_op = mnist.training(loss, FLAGS.learning_rate)'''

        #x = tf.placeholder("float", [None, 784])
        x = images
        W = tf.Variable(tf.zeros([784,10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(x,W) + b)
        #y_ = tf.placeholder("float", [None,10])
        y_ = labels
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        y1 = tf.nn.softmax(tf.matmul(data_sets.test.images,W) + b)
        correct_prediction = tf.equal(tf.argmax(y1,1), tf.argmax(data_sets.test.labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


        # 初始化参数，特别注意：string_input_producer 内部创建了一个epoch 计数变量，
        # 归入tf.GraphKeys.LOCAL_VARIABLES 集合中，必须单独用initialize_local_variables()初始化
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess = tf.InteractiveSession()
        sess.run(init_op)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #input()
        #sess.run(train_step)
        try:
            step = 0
            while not coord.should_stop(): # 进入永久循环
                start_time = time.time()
                #_, loss_value = sess.run([train_op, loss])
                sess.run(train_step)
                duration = time.time() - start_time
                # 每100 次训练输出一次结果
                if step % 100 == 0:
                    print (sess.run(accuracy))
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            coord.request_stop() # 通知其他线程关闭
        coord.join(threads)
        sess.close()
        


# 定义整个网络
def alex_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # 第一层卷积
    # 卷积
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # 下采样
    pool1 = maxpool2d('pool1', conv1, k=2)

    # 规范化
    norm1 = norm('norm1', pool1, lsize=4)
    # 第二层卷积
    # 卷积
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    # 最大池化（向下采样）
    pool2 = maxpool2d('pool2', conv2, k=2)
    # 规范化
    norm2 = norm('norm2', pool2, lsize=4)
    # 第三层卷积
    # 卷积
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    # 下采样
    #pool3 = maxpool2d('pool3', conv3, k=2)
    pool3 = conv3

    # 规范化
    norm3 = norm('norm3', pool3, lsize=4)
    # 第四层卷积
    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
    # 第五层卷积
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # 下采样
    pool5 = maxpool2d('pool5', conv5, k=2)

    # 规范化
    norm5 = norm('norm5', pool5, lsize=4)

    # 全连接层1
    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    #全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    # 输出层
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

# 定义网络的一参数
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
# 定义网络的参数
n_input = 784# 输入的维度(img shape: 28×28)
n_classes = 10 # 标记的维度 (0-9 digits)
dropout = 0.75 # Dropout 的概率，输出的可能性

# 定义卷积操作
def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name) # 使用relu 激活函数
# 定义池化层操作
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME', name=name)
# 规范化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0,beta=0.75, name=name)


def run_training_alex():
    with tf.Graph().as_default():
        # 输入images 和labels
        images, labels = inputs(train=True, batch_size=batch_size,num_epochs=5)
        #测试数据
        data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)

        #############
        # 定义所有的网络参数
        weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, 10]))
        }
        biases = {
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([n_classes]))
        }
        keep_prob = tf.placeholder(tf.float32) #dropout

        # 构建模型
        pred = alex_net(images, weights, biases, keep_prob)
        # 定义损失函数和优化器

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # 评估函数
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        ###############test
        pred1 = alex_net(data_sets.test.images, weights, biases, keep_prob)
        correct_pred1 = tf.equal(tf.argmax(pred1, 1), tf.argmax(data_sets.test.labels, 1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))

        # 初始化参数，特别注意：string_input_producer 内部创建了一个epoch 计数变量，
        # 归入tf.GraphKeys.LOCAL_VARIABLES 集合中，必须单独用initialize_local_variables()初始化
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init_op)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #进度条
        widgets = [progressbar.Percentage(),
               ' ', progressbar.Bar(),
               ' ', progressbar.AdaptiveETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=data_sets.train.num_examples * 5).start()
        try:
            step = 0
            while not coord.should_stop(): # 进入永久循环
                start_time = time.time()
                #_, loss_value = sess.run([train_op, loss])
                sess.run(optimizer, feed_dict={keep_prob: dropout})
                duration = time.time() - start_time
                # 每100 次训练输出一次结果
                if step % display_step == 0:
                    # 计算损失值和准确度，输出
                    loss, acc = sess.run([cost, accuracy], feed_dict={keep_prob: 1.})
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
                step += 1
                pbar.update(step)
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            coord.request_stop() # 通知其他线程关闭
        print("Testing Accuracy:", sess.run(accuracy1, feed_dict={keep_prob: 1.}))
        coord.join(threads)
        pbar.finish()
        sess.close()

#main()
#run_training_alex()
run_training()





