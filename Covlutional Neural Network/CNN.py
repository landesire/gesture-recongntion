#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/9 16:52
# @Author  : Bindingdai && sylvia lynne
# @Site    : 
# @File    : CNN.py
# @Software: PyCharm Community Edition


from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import  math
# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


f = open("S_feature_all.txt")  # skip the header
data = np.loadtxt(f, delimiter=',')
# print data
label_list = []
print(len(data))
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min)
print(data[0])

for i in range(len(data)):
    label = (i / 800)
    label_list.append(label)
y_train2 = np.array(label_list)
y_train2 = y_train2.reshape(-1, 1)
# print y_train2
print(len(y_train2))
all_data = np.hstack((y_train2, data))
print("this is all data")
print(all_data[255][0])
print("all data shape" + str(all_data.shape))

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
print("train data feature number " + str(train_data.shape))
print(test_label)
# print train_feature
print("this is label")

print("begins2")

# Training Parameters
learning_rate = 0.001
num_steps = 10000
batch_size = 800
display_step = 10

# Network Parameters
num_input = 6600 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 10, 660, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print("thi is conv1")
    print(conv1)
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print("thi is conv2")
    print(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.sigmoid(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32],stddev=math.sqrt(2.0/(num_input+32)))),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([5*24*32, 128],stddev=math.sqrt(2.0/(32+128)))),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([128, num_classes],stddev=math.sqrt(2.0/(num_input+32))))
}

biases = {
    'bc1': tf.Variable(tf.zeros([32])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bd1': tf.Variable(tf.zeros([128])),
    'out': tf.Variable(tf.zeros([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        offset = (step * batch_size) % (train_label.shape[0] - batch_size)
        batch_feature = train_feature[offset:(offset + batch_size), :]
        batch_label = train_label[offset:(offset + batch_size), :]
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_feature, Y: batch_label, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_feature,
                                                                 Y: batch_label,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")


    # Calculate accuracy for test set
    print("Testing Accuracy:",
        sess.run(accuracy, feed_dict={X: test_feature,
                                      Y: test_label,
                                      keep_prob: 1.0}))