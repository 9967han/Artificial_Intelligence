import tensorflow as tf
import numpy as np
import os
def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def cnn(X):
    # 첫번쨰 convolutional layer
    Kernel1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 32], stddev=0.1))
    Bias1 = tf.Variable(tf.constant(0.1, shape=[32]))
    Conv1 = tf.nn.conv2d(X, Kernel1, strides=[1, 1, 1, 1], padding='SAME') + Bias1
    Activation1 = tf.nn.relu(Conv1)

    # 첫번째 pooling layer
    Pool1 = tf.nn.max_pool(Activation1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

    print(Pool1)

    # 두번째 convolutional layer
    Kernel2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 16], stddev=0.1))
    Bias2 = tf.Variable(tf.constant(0.1, shape=[16]))
    Conv2 = tf.nn.conv2d(Pool1, Kernel2, strides=[1, 1, 1, 1], padding='SAME') + Bias2
    Activation2 = tf.nn.relu(Conv2)

    # 두번쨰 pooling layer
    Pool2 = tf.nn.max_pool(Activation2, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')
    Pool2 = tf.reshape(Pool2, [-1, 12*8*16])


    W3 = tf.Variable(tf.random_normal([12*8*16, 128], stddev=0.01))
    fc1 = tf.nn.relu(tf.matmul(Pool2, W3))

    W4 = tf.Variable(tf.random_normal([128,64], stddev = 0.01))
    Bias4 = tf.Variable(tf.random_normal(shape=[64], stddev=0.01))
    fc2 = tf.nn.relu(tf.matmul(fc1, W4) + Bias4)

    W5 = tf.Variable(tf.random_normal([64,1], stddev = 0.01))
    Bias5 = tf.Variable(tf.random_normal(shape=[1], stddev=0.01))
    fc3 = tf.nn.sigmoid(tf.matmul(fc2, W5) + Bias5)
    return fc3
