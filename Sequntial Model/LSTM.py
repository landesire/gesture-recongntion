#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : for sylvia lynne



from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import math
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]
def data_prepare():
    f = open("S_feature_all.txt")  # skip the header

    data = np.loadtxt(f, delimiter=',')
    label_list = []

    # data_min,data_max=data.min(),data.max()
    # data=(data-data_min)/(data_max-data_min)
    print(len(data))
    print(data[0])
    print(len(data[0]))
    curret_feature=[]
    list_data = data.tolist()
    #put data in time series format
    #改变特征，让每一帧的1-2-3维joint合并，然后再十帧一起排列
    for index in range(len(data)):
        current_single_joint=list_data[index][0:480]
        current_joint_pair=list_data[index][480:1800]
        current_triple_joint=list_data[index][1800:6600]
        for frame_index in range(10):
            single_joint_feature=current_single_joint[48*frame_index:48*(frame_index+1)]
            joint_pair_feature=current_joint_pair[132*frame_index:132*(frame_index+1)]
            triple_joint_feature=current_triple_joint[480*frame_index:480*(frame_index+1)]
            #print(single_joint_feature)
            #print(triple_joint_feature)
            curret_feature=curret_feature+single_joint_feature+joint_pair_feature+triple_joint_feature
        data[index]=curret_feature
        #print("after change")
        #print(data[index][150])
        curret_feature.clear()
    print(data)
    print(data[0])
    print(len(data[0]))


    #生成标注类别1-10类
    for i in range(len(data)):
        label = (i /800)
        label_list.append(label)
        #print(label)
    y_label = np.array(label_list)
    y_label = y_label.reshape(-1, 1)
    # print y_train2
    print(len(y_label))
    all_data = np.hstack((y_label, data))
    print("this is all data")
    print(all_data[255][125])
    print("all data shape"+str(all_data.shape))



    shuffled_indices = np.random.permutation(len(all_data))
    test_set_size = int(len(all_data) * 0.2)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_data = all_data[train_indices]
    test_data = all_data[test_indices]
    print("train data lenth" + str(len(train_data)))
    print("test data lenth" + str(len(test_data)))

    train_feature = train_data[:, 1:len(train_data[0])]
    train_label = train_data[:, 0].astype(np.int)
    train_label_size = train_label.shape[0]
    train_label = train_label.reshape(train_label_size, 1)
    train_label = convert_to_one_hot(train_label, 10)

    test_feature = test_data[:, 1:len(test_data[0])]
    test_label = test_data[:, 0].astype(np.int)
    test_label_size = test_label.shape[0]
    test_label = test_label.reshape(test_label_size, 1)
    test_label = convert_to_one_hot(test_label, 10)
    print("train data feature number " + str(train_feature.shape))
    print("train data feature number " + str(train_label.shape))
    print(test_label)
    # print train_feature
    print("this is label")

    print("begins2")
    return (train_feature,train_label,test_feature,test_label)

'''
对于每个sample.我们将每一帧看作一个序列，10帧为10个序列，每个序列660维，由1-2-3个joints合并而成
'''


# def RNN(x, weights, biases):
#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, timesteps, n_input)
#     # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
#
#     # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
#     x = tf.unstack(x, timesteps, 1)
#
#     # Define a lstm cell with tensorflow
#     lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#
#     # Get lstm cell output
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#     # Linear activation, using rnn inner loop last output
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']

def LSTM(train_feature,train_label,test_feature,test_label):
    # Training Parameters
    learning_rate = 0.008
    training_steps = 60000
    batch_size = 400
    display_step = 200

    # Network Parameters
    num_input = 660  # single-joint feature+ two-joint feature + triple-joint feature
    timesteps = 10  # timesteps  (frame的个数)
    num_hidden = 128  # hidden layer num of features
    num_classes = 10  # 10个动作
    momentum = 0.7

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    keep_prob = tf.placeholder("float")
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes],mean=0.0,stddev=math.sqrt(2.0/(num_input+num_classes))))
    }
    biases = {
        'out': tf.Variable(tf.constant(0.0))
    }
    #将输入数据转换为10*660的矩阵形式，而不是6600的一行
    x = tf.unstack(X, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    #一共有10个outputs,每一个对应10个序列的输出,outputs[-1]代表最后一个output,可以认为是rnn中间层输出结果，也是1个等长的向量
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    acc_count = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        # RNN的测试集同样要将660*10转成10个660的向量作为输入
        test_data = test_feature.reshape((-1, timesteps, num_input))
        test_label = test_label
        for step in range(1, training_steps + 1):
            offset = (step * batch_size) % (train_label.shape[0] - batch_size)
            batch_feature = train_feature[offset:(offset + batch_size), :]
            batch_label = train_label[offset:(offset + batch_size), :]
            # 每一个batch批次数据需要转换成训练/测试数据那样的输入格式
            batch_feature = batch_feature.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            #print("current step" + str(step))
            #print(batch_feature.shape)
            sess.run(train_op, feed_dict={X: batch_feature, Y: batch_label})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy

                loss, acc, acc_sum= sess.run([loss_op, accuracy,acc_count], feed_dict={X: batch_feature,
                                                                     Y: batch_label})
                test_acc, test_acc_count = sess.run([accuracy, acc_count],feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc) + ", Training AccSum= " +
                      "{:.3f}".format(acc_sum) + ", Test Accuracy= " + "{:.3f}".format(
                    test_acc) + ", Test Sum= " + "{:.3f}".format(test_acc_count))

        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images

        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

if __name__ == '__main__':
    (train_feature, train_label, test_feature, test_label)=data_prepare()
    LSTM(train_feature,train_label,test_feature,test_label)