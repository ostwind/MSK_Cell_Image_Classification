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

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
reset_graph()

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original_imgs_path = os.path.join(dir, 'original/') # cut first array from /real_original/
train_path = os.path.join(dir, 'tensors/')
test_path = os.path.join(dir, 'test_set/')

with tf.name_scope('input'):
    train_batch, train_labels_batch, train_fnames = inputs(train_path)
    test_batch, test_labels_batch, test_fnames = inputs(test_path)
    training = tf.placeholder_with_default(True, shape=(), name='training')

    input_batch, input_labels_batch, input_fnames = tf.cond(training,
    lambda: (train_batch, train_labels_batch, train_fnames),
    lambda:(test_batch, test_labels_batch, test_fnames))
    # check tf.cond doc to make sure lambda: True, lambda: False
    #print(train_batch.get_shape(), train_labels_batch.get_shape())

with tf.name_scope('batch_norm'):
    #data = tf.cond(am_testing, lambda:test_q, lambda:train_q)
    batch_norm_momentum = 0.9
    batch_norm_layer = partial(tf.layers.batch_normalization, training=training,
    momentum=batch_norm_momentum)
    dropout_rate = 0.5

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

with tf.name_scope('conv1'):
    conv1 = tf.layers.conv2d(input_batch, filters=conv1_fmaps, kernel_size=conv1_ksize,
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
    #print('pool3: ', pool3.get_shape())
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 25 * 25])
    #print('pool_flat: ', pool3_flat.get_shape())

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, name="fc1")#, activation = tf.nn.elu)
    bn3 = tf.nn.elu(batch_norm_layer(fc1))
    #print('fc1: ', fc1.get_shape())

with tf.name_scope("output"):
    logits_before_bn = tf.layers.dense(bn3, n_outputs, name="output")
    logits = batch_norm_layer(logits_before_bn)
    #print('logits: ', logits.get_shape())
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

    preds  = tf.cast( tf.argmax(logits, 1), tf.int32 )
    mislabeled = tf.not_equal( preds, input_labels_batch )
    mislabeled_filenames = tf.cast( tf.boolean_mask( input_fnames, mislabeled ), tf.string)

with tf.name_scope('loss'): # once I mv this section from 'train', loss appears as loss_1 on tensorboard
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_labels_batch)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    # extra_update_ops for updating population statistics for batch norm
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, input_labels_batch, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

loss_record = tf.summary.scalar('loss', loss)
file_writer = tf.summary.FileWriter(logdir + 'loss', tf.get_default_graph())

acc_record = tf.summary.scalar('accuracy', accuracy)
train_acc_writer = tf.summary.FileWriter(logdir + 'train', tf.get_default_graph())
test_acc_writer = tf.summary.FileWriter(logdir + 'test', tf.get_default_graph())

misclassified_record = tf.summary.text('misclassifieds', mislabeled_filenames)
misclassified_writer = tf.summary.FileWriter(logdir + 'misclassified', tf.get_default_graph())

write_op = tf.summary.merge_all() # put into session.run!

def train():
    with tf.Session() as sess:
        init.run()
        #saver.restore(sess, './saved_model')
        n_epochs = 4
        tf.train.start_queue_runners(sess=sess)
        step = 0
        for epoch in range(n_epochs):
            for i in range(107000//64):
                if step % 200 == 0: # 8 updates per epoch
                    #if i % 30 == 0:
                    #    print(epoch, step, acc_train)
                    summary_str = loss_record.eval()
                    file_writer.add_summary(summary_str, step)

                    acc_train = acc_record.eval(feed_dict = {training: True}  )
                    train_acc_writer.add_summary(acc_train, step)

                    acc_test = acc_record.eval(feed_dict = { training: False} )
                    test_acc_writer.add_summary(acc_test, step)

                    misclass = misclassified_record.eval(feed_dict = { training: False} )
                    misclassified_writer.add_summary(misclass, step)

                _, loss_val = sess.run(
                [training_op, write_op], feed_dict={training: True} )

                step += 1

            save_path = saver.save(sess, "./saved_model")
    file_writer.close()
train()

#def test():


#    with tf.name_scope('validation'):
#        test_image, test_label = inputs(training = False)

#    mislabeled_samples = set()
    #with tf.Session() as sess:
    #    saver.restore(sess, './saved_model')
#    with open('mislabeled.p', 'wb') as fp:
#        pickle.dump(list(mislabeled_samples), fp)