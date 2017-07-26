''' A modified mnist model from
    https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks.ipynb
'''

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import numpy as np
import os
from sample_gen import dataload, show
import tensorflow as tf

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
tensor_path = os.path.join(dir, 'tensors/')

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

height = 50
width = 50
channels = 23
n_inputs = height * width * channels

conv1_fmaps = 10
conv1_ksize = 5
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 20
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 2

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
    print(X.get_shape())
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    #print(X_reshaped.get_shape())
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
print('conv1: ', conv1.get_shape())

conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")
print('conv2: ', conv2.get_shape())

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    print('pool3: ', pool3.get_shape())
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 25 * 25])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    print('fc1: ', fc1.get_shape())

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    print('logits: ', logits.get_shape())
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
    #print(Y_proba.get_shape())

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

n_epochs = 4
test_path = os.path.join(dir, 'test_set/')

def test():
    preds = []
    true = []
    with tf.Session() as sess:
        init.run()
        saver.restore(sess, './my_mnist_model')
        for X_test, y_test in dataload(test_path):
            Z = Y_proba.eval(feed_dict= {X: X_test})
            for i in range(len(y_test)):
                true.append( y_test[i])
                preds.append( np.argmax(Z[i], axis = 0) )
                print( Z[i], y_test[i] )
    print( len(preds),
    1- ((np.abs( np.array(preds) - np.array(true) ).sum())/len(preds)) )

def train():
    with tf.Session() as sess:
        init.run()
        saver.restore(sess, './my_mnist_model')
        for epoch in range(n_epochs):
            i = 0
            for X_batch, y_batch in dataload(tensor_path):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict = {X: X_batch, y: y_batch} )
            print(epoch, acc_train)
            save_path = saver.save(sess, "./my_mnist_model")
train()
test()
