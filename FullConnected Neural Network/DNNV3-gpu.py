# Author: bindingdai & sylvia lynne

import numpy as np
import math
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
log_device_placement = True
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def data_prepare():
    f = open("/cephfs/group/omg-qqv-video-mining/bindingdai/feature_all.txt")  # skip the header
    data = np.loadtxt(f, delimiter=',')
    # print data
    label_list = []
    print(len(data))

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
    return (train_feature,train_label,test_feature,test_label)

#deep neural network
def DNN(train_feature,train_label,test_feature,test_label):
    # Parameters
    learning_rate = 0.007
    num_steps = 100000
    momentum = 0.7
    batch_size = 400
    display_step = 100

    # Network Parameters
    n_hidden_1 = 60  # 1st layer number of neurons
    n_hidden_2 = 60  # 2nd layer number of neurons
    n_hidden_3 = 32  # 2nd layer number of neurons
    num_input = 6600  # data input (feature length)
    num_classes = 10  #  total classes (0-9 class)

    # tf Graph input
    with tf.device('/gpu:0'):
        X = tf.placeholder("float", [None, num_input])
        Y = tf.placeholder("float", [None, num_classes])
        keep_prob=tf.placeholder("float")

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1],mean=0.0,stddev=math.sqrt(2.0/(num_input+n_hidden_1)))),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],mean=0.0,stddev=math.sqrt(2.0/(n_hidden_2+n_hidden_1)))),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=math.sqrt(2.0 / (n_hidden_2 + n_hidden_3)))),
             'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes],mean=0.0,stddev=math.sqrt(2.0/(num_classes+n_hidden_3))))
        }
        biases = {
            'b1': tf.Variable(tf.constant(0.0)),
            'b2': tf.Variable(tf.constant(0.0)),
            'b3': tf.Variable(tf.constant(0.0)),
            'out': tf.Variable(tf.constant(0.0))
        }

        # first layer
        layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
        # layer_1 = tf.Print(layer_1, [layer_1], summarize=60)
        layer_1 = tf.nn.sigmoid(layer_1)
        # second layer

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)

        # third layer
        layer_3 = tf.add(tf.matmul(layer_1, weights['h3']), biases['b3'])
        layer_3 = tf.nn.dropout(layer_3, keep_prob)

        out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
        prediction = tf.nn.softmax(out_layer)
        # prediction = tf.Print(prediction, [prediction], summarize=10)
        logits = out_layer
        # weights_print=tf.Print(logits,[weights['out']],summarize=64)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
        #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=0.01)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        predict_label = tf.argmax(prediction, 1)
        # predict_label=tf.Print(predict_label,[predict_label],summarize=1000)
        correct_pred = tf.equal(predict_label, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc_count = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:

            # Run the initializer
            sess.run(init)

            for step in range(1, num_steps + 1):
                # full batch train
                # batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop)
                offset = (step * batch_size) % (train_label.shape[0] - batch_size)
                batch_feature = train_feature[offset:(offset + batch_size), :]
                batch_label = train_label[offset:(offset + batch_size), :]
                sess.run(train_op, feed_dict={X: batch_feature, Y: batch_label, keep_prob: 0.5})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc, acc_sum = sess.run([loss_op, accuracy, acc_count],
                                                  feed_dict={X: train_feature, Y: train_label, keep_prob: 1.0})
                    test_acc, test_acc_count = sess.run([accuracy, acc_count],
                                                        feed_dict={X: test_feature, Y: test_label, keep_prob: 1.0})
                    print("Step " + str(step) + ", Minibatch Loss= " +
                          "{:.4f}".format(loss) + ", Training Accuracy= " +
                          "{:.3f}".format(acc) + ", Training AccSum= " +
                          "{:.3f}".format(acc_sum) + ", Test Accuracy= " + "{:.3f}".format(
                        test_acc) + ", Test Sum= " + "{:.3f}".format(test_acc_count))

            print("Optimization Finished!")

            # Calculate accuracy for test features
            print("Testing Accuracy:",
                  sess.run([accuracy, acc_count], feed_dict={X: test_feature, Y: test_label, keep_prob: 1.0}))


if __name__ == '__main__':
    (train_feature, train_label, test_feature, test_label)=data_prepare()
    DNN(train_feature,train_label,test_feature,test_label)