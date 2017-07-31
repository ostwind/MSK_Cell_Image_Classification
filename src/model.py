''' A modified mnist model from
    https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks.ipynb
'''
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

import numpy as np
import os
from sample_gen import inputs
import tensorflow as tf
from functools import partial
import pickle

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original_imgs_path = os.path.join(dir, 'original/') # cut first array from /real_original/
train_path = os.path.join(dir, 'tensors/')
test_path = os.path.join(dir, 'test_set/')

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
reset_graph()

height = 50
width = 50
channels = 23
n_inputs = height * width * channels

conv1_fmaps = 10
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 20
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 1000
n_outputs = 3

with tf.name_scope('input'):
    images_batch, labels_batch = inputs(train_path)
print(images_batch.get_shape(), labels_batch.get_shape())

with tf.name_scope('batch_norm'):
    training = tf.placeholder_with_default(False, shape=(), name='training')
    batch_norm_momentum = 0.9
    batch_norm_layer = partial(tf.layers.batch_normalization, training=training,
    momentum=batch_norm_momentum)
    dropout_rate = 0.5

with tf.name_scope('conv1'):
    conv1 = tf.layers.conv2d(images_batch, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad, name="conv1")
    #print('conv1: ', conv1.get_shape())
    bn1 = tf.nn.elu(batch_norm_layer(conv1))
    hidden1_drop = tf.layers.dropout(bn1, dropout_rate, training=training)

with tf.name_scope('conv2'):
    conv2 = tf.layers.conv2d(hidden1_drop, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad, name="conv2")
    #print('conv2: ', conv2.get_shape())
    bn2 = tf.nn.elu(batch_norm_layer(conv2))
    hidden2_drop = tf.layers.dropout(bn2, dropout_rate, training=training)

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(hidden2_drop, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    print('pool3: ', pool3.get_shape())
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 25 * 25])
    print('pool_flat: ', pool3_flat.get_shape())

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, name="fc1")#, activation = tf.nn.elu)
    bn3 = tf.nn.elu(batch_norm_layer(fc1))
    #print('fc1: ', fc1.get_shape())

with tf.name_scope("output"):
    logits_before_bn = tf.layers.dense(bn3, n_outputs, name="output")
    logits = batch_norm_layer(logits_before_bn)
    #print('logits: ', logits.get_shape())
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope('loss'): # once I mv this section from 'train', loss appears as loss_1 on tensorboard
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_batch)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    # extra_update_ops for updating population statistics for batch norm
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels_batch, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

loss_record = tf.summary.scalar('loss', loss)
file_writer = tf.summary.FileWriter(logdir + 'loss', tf.get_default_graph())

acc_record = tf.summary.scalar('accuracy', accuracy)

write_op = tf.summary.merge_all() # put into session.run!
train_acc_writer = tf.summary.FileWriter(logdir + 'train', tf.get_default_graph())
#test_acc_writer = tf.summary.FileWriter(logdir + 'test', tf.get_default_graph())

def test():
    mislabeled_samples = set()
    preds = []
    true = []
    with tf.Session() as sess:
        saver.restore(sess, './saved_model')
        for X_test, y_test, f_name in dataload(train = False):
            Z = Y_proba.eval(feed_dict= {X: X_test})
            for i in range(len(y_test)):
                true.append( y_test[i])
                preds.append( np.argmax(Z[i], axis = 0) )
                if y_test[i] != np.argmax(Z[i], axis = 0):
                    mislabeled_samples.add(f_name[i])
            #print( Z[:10], y_test[:10].transpose() )
    print( len(preds), 1- ((np.abs( np.array(preds) - np.array(true) ).sum())/len(preds)) )

    with open('mislabeled.p', 'wb') as fp:
        pickle.dump(list(mislabeled_samples), fp)

def train():
    with tf.Session() as sess:
        init.run()
        #saver.restore(sess, './saved_model')
        n_epochs = 4
        tf.train.start_queue_runners(sess=sess)
        step = 0
        for epoch in range(n_epochs):
            for i in range(107000//64):

                #X_batch, y_batch = inputs( train_path )
                #acc_train = accuracy.eval( )
                #print(acc_train)
                #if i % 1000 == 0:
                #    print(acc_train)
                #acc_test = accuracy.eval(feed_dict = {X: X_test, y: y_test} )

                if step % 200 == 0: # 8 updates per epoch
                    print(Y_proba.eval(), labels_batch.eval())
                    summary_str = loss_record.eval()
                    #if i % 30 == 0:
                    #    print(epoch, step, acc_train)
                    file_writer.add_summary(summary_str, step)

                    acc_train = acc_record.eval( )
                    train_acc_writer.add_summary(acc_train, step)

                    # X_test, y_test, f = g.__next__() # python 3 's next
                    # acc_test = acc_record.eval(feed_dict = {
                    # training: False, X: X_test, y: y_test})
                    # test_acc_writer.add_summary(acc_test, step)
                    # test_acc_writer.flush()

                _, loss_val = sess.run(
                [training_op, write_op], feed_dict={training: True} ) #[training_op, write_op],
                #feed_dict={training:True, X: X_batch, y: y_batch})
                step += 1

            save_path = saver.save(sess, "./saved_model")
    file_writer.close()
train()
#test()
