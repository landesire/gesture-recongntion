
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

#artifical neural network
def ANN(train_feature,train_label,test_feature,test_label):
    # Parameters
    learning_rate = 0.01
    num_steps = 60000
    momentum = 0.7
    batch_size = 30
    display_step = 100
    keep_prob=0.95

    # Network Parameters
    n_hidden_1 = 64  # 1st layer number of neurons
    num_input = 6600  # data input (feature length)
    num_classes = 10  #  total classes (0-9 class)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    keep_prob = tf.placeholder("float")

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    layer_1 = tf.add(tf.identity(X))
    layer_1=tf.nn.dropout(layer_1,keep_prob)*keep_prob
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

    logits = out_layer
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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

        for step in range(1, num_steps + 1):
            # full batch train
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            offset = (step * batch_size) % (train_label.shape[0] - batch_size)
            batch_feature = train_feature[offset:(offset + batch_size), :]
            batch_label = train_label[offset:(offset + batch_size), :]
            sess.run(train_op, feed_dict={X: batch_feature, Y: batch_label})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc, acc_sum = sess.run([loss_op, accuracy, acc_count], feed_dict={X: train_feature,Y: train_label})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc) + ", Training AccSum= " +
                      "{:.3f}".format(acc_sum))

        print("Optimization Finished!")

        # Calculate accuracy for test features
        print("Testing Accuracy:",
              sess.run([accuracy, acc_count], feed_dict={X: test_feature, Y: test_label}))


if __name__ == '__main__':
    (train_feature, train_label, test_feature, test_label)=data_prepare()
    ANN(train_feature,train_label,test_feature,test_label)