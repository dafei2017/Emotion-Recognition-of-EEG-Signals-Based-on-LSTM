#encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
tf.reset_default_graph()

# Hyper Parameters
learning_rate = 0.001    # 学习率
n_steps = 12            # LSTM 展开步数（时序持续长度）
n_inputs = 21504           # 输入节点数，由每段信号*通道数得来：672*32
n_hiddens = 64         # 隐层节点数
n_layers = 2            # LSTM layer 层数
n_classes = 2          # 输出节点数（分类数目）
# 加载训练数据
trainData=[]
trainLabel=[]
segNum = 12  #信号分割的段数，原始信号为8064维，分割成12段，则每段为672
thld = 5    #评价值阈值，小于5记为类别0，反之记为1，评价值范围为（1，9）
train_identity = 24  #用于模型训练的人物个数
for i in range(train_identity):  #读取数据集中的前24个人数据作为训练集
    datName = 's%02d.dat'%(i+1)
    x = pickle.load(open(datName, 'rb'),encoding='iso-8859-1')  #读取.dat文件
    data = x['data']   #读取数据，数据格式为视频数*通道数*数据维度
    labelss = x['labels'] #读取label
    segStep = int(data.shape[2]/segNum)  #分成的每段数据维度

    for j in range(data.shape[0]):   #把每段视频作为一个训练样本，对数据进行分段，得到的每个训练样本n_inputs维度为：通道数*segStep
        labelData = labelss[j][0]  # (valence,arousal,dominance,liking)
        oneHot = []       #按照tensorflow方式分配训练标签为（0,1）或者为（1,0）
        if labelData < 5:
            oneHot.append(1)
            oneHot.append(0)
        else:
            oneHot.append(0)
            oneHot.append(1)
        trainLabel.append(numpy.array(oneHot))

        segDatas = []   #对数据进行分段
        for m in range(segNum):
            segData = []
            for k in range(32):
                segData.append(data[j][k][(segStep*m):(segStep*m+segStep)])
            segData = numpy.array(segData)
            segData = segData.flatten()
            segDatas.append(segData)
        segDatas = numpy.array(segDatas)
        segDatas = segDatas.flatten()
        trainData.append(segDatas)

test_identity=8  #用于测试的人物数量

testData = []   #测试数据
testLabel = []   #测试标签
segNum = 12     #数据分割段数

for i in range(train_identity,train_identity+test_identity):   #读取数据集中的后8个人数据作为训练集
    datName = 's%02d.dat' % (i + 1)
    x = pickle.load(open(datName, 'rb'),encoding='iso-8859-1')
    data = x['data']
    labelss = x['labels']
    segStep = int(data.shape[2] / segNum)

    for j in range(data.shape[0]):
        labelData = labelss[j][0]  # (valence,arousal,dominance,liking)
        oneHot = []
        if labelData < 5:
            oneHot.append(1)
            oneHot.append(0)
        else:
            oneHot.append(0)
            oneHot.append(1)
        testLabel.append(numpy.array(oneHot))

        segDatas = []
        for m in range(segNum):
            segData = []
            for k in range(32):
                segData.append(data[j][k][(segStep * m):(segStep * m + segStep)])
            segData = numpy.array(segData)
            segData = segData.flatten()
            segDatas.append(segData)
        segDatas = numpy.array(segDatas)
        segDatas = segDatas.flatten()
        testData.append(segDatas)

test_x = numpy.array(testData)  #将数据转为numpy型
test_y = numpy.array(testLabel)

# 建立tensor placeholder
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, n_steps * n_inputs], name='x_input')     # 输入
    y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')               # 输出
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')           # 保持多少不被 dropout
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')       # 批大小

# weights and biases
with tf.name_scope('weights'):
    Weights = tf.Variable(tf.truncated_normal([n_hiddens, n_classes],stddev=0.1), dtype=tf.float32, name='W')
    tf.summary.histogram('output_layer_weights', Weights)
with tf.name_scope('biases'):
    biases = tf.Variable(tf.random_normal([n_classes]), name='b')
    tf.summary.histogram('output_layer_biases', biases)

# 定义LSTM结构
def RNN_LSTM(x, Weights, biases):
    # RNN 输入 reshape
    x = tf.reshape(x, [-1, n_steps, n_inputs])
    # 定义 LSTM cell
    # cell 中的 dropout
    def attn_cell():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
        with tf.name_scope('lstm_dropout'):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # attn_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # 实现多层 LSTM
    # [attn_cell() for _ in range(n_layers)]
    enc_cells = []
    for i in range(0, n_layers):
        enc_cells.append(attn_cell())
    with tf.name_scope('lstm_cells_layers'):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)
    # 全零初始化 state
    _init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn 运行网络
    outputs, states = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=_init_state, dtype=tf.float32, time_major=False)
   
    # 输出
    #return tf.matmul(outputs[:,-1,:], Weights) + biases
    return tf.nn.softmax(tf.matmul(outputs[:,-1,:], Weights) + biases)

with tf.name_scope('output_layer'):
    pred = RNN_LSTM(x, Weights, biases)
    tf.summary.histogram('outputs', pred)
# 定义损失
with tf.name_scope('loss'):
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices=[1]))#计算SoftMax损失
    tf.summary.scalar('loss', cost)
# optimizer
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)#定义Adam优化器

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope('accuracy'):
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(pred, axis=1))[1]#定义准确率计算
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all() #用于写入到tensorboard中

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) #初始化设置

num_samples = 960  #训练集的样本数量，人数*视频数：24*40

training_steps = 10000   #模型训练迭代次数
with tf.Session() as sess:
    sess.run(init)   #初始化
    train_writer = tf.summary.FileWriter("./logs/lstm_arousal/train",sess.graph)   #将训练过程写入到summary中，用于tensorboard显示
    test_writer = tf.summary.FileWriter("./logs/lstm_arousal/test",sess.graph)   #将测试过程写入到summary中，用于tensorboard显示
    trainData = numpy.array(trainData)   #转换为numpy数组
    trainLabel = numpy.array(trainLabel)
    # training
    # step = 1
    _batch_size = 30   #设置批训练大小
    epoch_iterations = num_samples / _batch_size   #训练完一整遍数据集需要的迭代次数
    #开始训练
    for step in range(1,training_steps+1):
        #按批大小读取数据
        if step%epoch_iterations==0:
            start_index = 0
            end_index = _batch_size
        else:
            start_index = ((step - 1) % epoch_iterations) * _batch_size
            end_index = (step % epoch_iterations) * _batch_size
        batch_x = trainData[int(start_index):int(end_index)]
        batch_y = trainLabel[int(start_index):int(end_index)]

        sess.run(train_op, feed_dict={x:batch_x, y:batch_y, keep_prob:0.5, batch_size:_batch_size})#进行迭代
        if step % 100 == 0:  #每一百次进行结果输出，并保存到Summary中
            loss = sess.run(cost, feed_dict={x:trainData, y:trainLabel, keep_prob:1.0, batch_size:trainData.shape[0]})
            acc = sess.run(accuracy, feed_dict={x:trainData, y:trainLabel, keep_prob:1.0, batch_size:trainData.shape[0]})
            testAcc = sess.run(accuracy, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]})
            print('Iter: %d' % step, '| train loss: %.6f' % loss, '| train accuracy: %.6f' % acc,
                  '| test accuracy:%.6f' % testAcc)
            train_result = sess.run(merged, feed_dict={x:trainData, y:trainLabel, keep_prob:1.0, batch_size:trainData.shape[0]})
            test_result = sess.run(merged, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]})
            train_writer.add_summary(train_result,step+1)
            test_writer.add_summary(test_result,step+1)

    print("Optimization Finished!")
    # prediction
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]}))
