import tensorflow as tf
def nn(X):
    W1 = tf.Variable(tf.random_normal([24, 10], stddev=0.01))
    B1 = tf.Variable(tf.random_normal(shape=[10], stddev=0.01))
    L1 = tf.nn.relu(tf.matmul(X, W1) + B1)

    W2 = tf.Variable(tf.random_normal([10, 1], stddev=0.01))
    B2 = tf.Variable(tf.random_normal(shape=[1], stddev=0.01))
    model = tf.nn.sigmoid(tf.matmul(L1, W2) + B2)
    return model
