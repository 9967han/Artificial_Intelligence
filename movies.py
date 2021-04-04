import tensorflow as tf
import numpy as np
from PIL import Image
from cnn import *
from nn import *
import os
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.19
session = tf.Session(config=config)
y_train = []

genre = ['Animation', 'Comedy', 'Family', 'Drama', 'Romance', 'Action', 'Adventure',
          'Thriller', 'Horror', 'History', 'Crime', 'Fantasy', 'Science Fiction', 'War',
          'Music', 'Mystery', 'Documentary', 'Western', 'Foreign', 'TV Movie'] # number of genre: 20

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def test_next_batch(num, featuredata, image, cnt):
    idx = np.arange(0, len(featuredata))  #998
    idx = idx[cnt:cnt+num]
    feature_sequence = [featuredata[i] for i in idx]
    image_sequence = [image[i] for i in idx]
    return np.asarray(feature_sequence), np.asarray(image_sequence)

feature = np.zeros(shape=[8588, 24], dtype=np.float32)
label = np.zeros(shape=[8588, 1], dtype=np.float32)

count = 0
with open('movie_train.csv', newline='', encoding='utf-8') as csvfile:
    movies = csv.reader(csvfile, delimiter=',')
    for row in movies: #type of row: list
        if count == 4180:
            continue
        #0: series
        #1: budget
        #2: popularity
        #3: revenue
        #4~23: genre

        #feature0: series
        if row[1] == "":
            feature[count, 0] = 0.0
        else:
            feature[count, 0] = 1.0

        #feature1: budget
        feature[count, 1] = float(row[2])/300000000.0

        #feature2: popularity
        feature[count, 2] = float(row[10])/547.4883

        #feature3: revenue
        feature[count, 3] = float(row[15])/2068223624.0

        #feature4~23: genre
        n = 5
        while True:
            indexNum = genre.index(row[3].split("'")[n])+1
            symbolNum = row[3].split("'")[n+1][-1]
            feature[count, indexNum] = 1.0
            if symbolNum == ']':
                break
            n=n+6

        #label0: vote_average
        label[count, 0] = (float(row[22])-1.7)/7.2
        count = count+1

feature_test = np.zeros(shape=[999, 24], dtype=np.float32)
count = 0

with open('movie_test.csv', newline='', encoding='utf-8') as csvfile:
    movies = csv.reader(csvfile, delimiter=',')
    for row in movies: #type of row: list
        #0: series
        #1: budget
        #2: popularity
        #3: revenue
        #4~23: genre

        #feature0: series
        if row[1] == "":
            feature_test[count, 0] = 0.0
        else:
            feature_test[count, 0] = 1.0

        #feature1: budget
        feature_test[count, 1] = float(row[2])/300000000.0

        #feature2: popularity
        feature_test[count, 2] = float(row[10])/547.4883

        #feature3: revenue
        feature_test[count, 3] = float(row[15])/2068223624.0

        #feature4~23: genre
        n = 5
        while True:
            indexNum = genre.index(row[3].split("'")[n])+1
            symbolNum = row[3].split("'")[n+1][-1]
            feature_test[count, indexNum] = 1.0
            if symbolNum == ']':
                break
            n=n+6
        count = count + 1

Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

nn_X = tf.placeholder(shape=[None, 24], dtype=tf.float32)
nn_output = nn(nn_X)
loss_nn = tf.reduce_mean((Y-nn_output)*(Y-nn_output))
optimize_nn = tf.train.AdamOptimizer(0.00001).minimize(loss_nn)

cnn_X = tf.placeholder(shape=[None, 300, 200, 3], dtype=tf.float32)
cnn_output = cnn(cnn_X)
loss_cnn = tf.reduce_mean((Y-cnn_output)*(Y-cnn_output))
optimize_cnn = tf.train.AdamOptimizer(0.00001).minimize(loss_cnn)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 10
total_batch = int(8588/ batch_size)
test_total_batch = int(999/ batch_size)

for epoch in range(20):
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, feature, label)
        _,loss_nn_val = sess.run((optimize_nn, loss_nn),feed_dict={nn_X: batch_xs, Y: batch_ys})
    print('Epoch:', '%04d' % (epoch + 1), 'loss_nn: ', loss_nn_val**0.5)
print('Neural Network Training Done!')

trainImage = np.load('movie_train_image.npy')  # (8588, 300, 200, 3), (300, 200, 3) 짜리 8588개
testImage = np.load('movie_test_image.npy')  # (999, 300, 200, 3), (300, 200, 3)짜리 1000개

for epoch in range(20):
    for i in range(total_batch): #total_batch: 858
        batch_xs, batch_ys = next_batch(batch_size, trainImage, label)
        _,loss_cnn_val, cnnval = sess.run((optimize_cnn, loss_cnn, cnn_output), feed_dict={cnn_X: batch_xs, Y: batch_ys})
    print('Epoch:', '%04d' % (epoch + 1), 'loss_cnn: ', loss_cnn_val**0.5)
print('Convolution Neural Network Training Done!')

f = open("result.txt", 'w')
for i in range(test_total_batch+1): #test_total_batch: 100
    batch_xs, batch_xs2 = test_next_batch(batch_size, feature_test, testImage, i*10)
    nn,cnn = sess.run((nn_output, cnn_output), feed_dict={nn_X: batch_xs, cnn_X: batch_xs2})
    result = 1.7+(nn*10*(5/6) + cnn*10*(1/6))*0.83
    result = result.tolist()

    for j in range(len(result)):
        string = 'data#' + str(i*10+j) + ': '+ str(round(result[j][0], 1))
        print(string)
    for j in range(len(result)):
        string = 'data#' + str(i * 10 + j) + ': ' + str(round(result[j][0], 1)) + '\n'
        f.write(string)
f.close()


