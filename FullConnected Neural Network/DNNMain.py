# Author: bindingdai & sylvia lynne

import numpy as np
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

#compare to mnist data
#from tensorflow.examples.tutorials.mnist import input_data
#print("begins1")
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#print(mnist.train.labels.shape)
#print(mnist.train.labels)

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def data_prepare():
    f = open("S_feature_all.txt")  # skip the header
    data = np.loadtxt(f, delimiter=',')
    # print data
    label_list = []
    print(len(data))
    data_min,data_max=data.min(),data.max()
    data=(data-data_min)/(data_max-data_min)
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
    print("train data feature number " + str(train_data.shape))
    print(test_label)
    # print train_feature
    print("this is label")

    print("begins2")
    return (train_feature,train_label,test_feature,test_label)

#artifical neural network
def DNN(train_feature,train_label,test_feature,test_label):
    # Parameters
    learning_rate = 0.008
    num_steps = 60000
    momentum = 0.7
    batch_size = 400
    display_step = 100

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons
    num_input = 6600  # data input (feature length)
    num_classes = 10  #  total classes (0-9 class)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    beta_regul = tf.placeholder(tf.float32)

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
        'out': tf.Variable(tf.truncated_normal([num_classes]))
    }

    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_2_train = tf.nn.dropout(layer_2, 1.0)
    out_layer_train= tf.matmul(layer_2_train, weights['out']) + biases['out']
    out_layer_test = tf.matmul(layer_2, weights['out']) + biases['out']


    logits_train = out_layer_train
    logits_test=out_layer_test
    prediction_train=tf.nn.softmax(logits_train)
    prediction_test = tf.nn.softmax(logits_test)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_train, labels=Y)+beta_regul * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])+tf.nn.l2_loss(weights['out'])))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred_train= tf.equal(tf.argmax(prediction_train, 1), tf.argmax(Y, 1))
    accuracy_train = tf.reduce_mean(tf.cast(correct_pred_train, tf.float32))
    acc_count_train= tf.reduce_sum(tf.cast(correct_pred_train, tf.float32))

    correct_pred_test = tf.equal(tf.argmax(prediction_test, 1), tf.argmax(Y, 1))
    accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))
    acc_count_test = tf.reduce_sum(tf.cast(correct_pred_test, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps + 1):
            # full batch train
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            offset = (step * batch_size) % (train_label.shape[0] - batch_size)
            batch_feature = train_feature[offset:(offset + batch_size), :]
            batch_label = train_label[offset:(offset + batch_size), :]
            sess.run(train_op, feed_dict={X: batch_feature, Y: batch_label,beta_regul:1e-3})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc, acc_sum = sess.run([loss_op, accuracy_train, acc_count_train], feed_dict={X: train_feature,Y: train_label,beta_regul:0.0})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc) + ", Training AccSum= " +
                      "{:.3f}".format(acc_sum))

        print("Optimization Finished!")

        # Calculate accuracy for test features
        print("Testing Accuracy:",
              sess.run([accuracy_test, acc_count_test], feed_dict={X: test_feature, Y: test_label,beta_regul:1e-5}))


if __name__ == '__main__':
    (train_feature, train_label, test_feature, test_label)=data_prepare()
    DNN(train_feature,train_label,test_feature,test_label)