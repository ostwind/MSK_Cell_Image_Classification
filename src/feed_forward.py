'''
'''

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

import numpy as np
import os
from sample_feed import inputs
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
    # two separate queues
    train_batch, train_labels_batch, train_fnames = inputs(train_path)
    test_batch, test_labels_batch, test_fnames = inputs(test_path, batch_size = 256)
    training = tf.placeholder_with_default(True, shape=(), name='training')

    # batch to feed depends on training / not training (relevant for BN + DO)
    input_batch, input_labels_batch, input_fnames = tf.cond(training,
    lambda: (train_batch, train_labels_batch, train_fnames),
    lambda:(test_batch, test_labels_batch, test_fnames))

with tf.name_scope('batch_norm'):
    batch_norm_momentum = 0.9
    batch_norm_layer = partial(tf.layers.batch_normalization, training=training,
    momentum=batch_norm_momentum)
    dropout_rate = 0.5

def conv(input, fmaps, training, name, ksize = 3, stride = 1, pad = "SAME" ):
    with tf.name_scope(name):
        conv = tf.layers.conv2d(input, filters=fmaps, kernel_size=ksize,
                                 strides=stride, padding=pad, name=name)
        #print(name, conv.get_shape())
        bn = tf.nn.elu(batch_norm_layer(conv))
        hidden_drop = tf.layers.dropout(bn, dropout_rate, training=training)
    return hidden_drop

def pool(input, pool_fmaps, name, flatten = False,
ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = "VALID"):
    with tf.name_scope(name):
        pool = tf.nn.max_pool(input, ksize= ksize, strides=strides, padding= padding)
        #print('pool3: ', pool3.get_shape())
        if flatten: # pool_fmaps = pool5_fmaps = conv4_fmaps = 40
            pool = tf.reshape(pool, shape=[-1, pool_fmaps * 12 * 12])
    return pool

conv1_fmaps = 10
conv2_fmaps = 20
pool3_fmaps = conv2_fmaps

conv4_fmaps = 40
pool5_fmaps = conv4_fmaps

hidden1_drop = conv(input_batch, conv1_fmaps, training, 'conv1',
ksize = 5)
hidden2_drop = conv(hidden1_drop, conv2_fmaps, training, 'conv2',)
pool3_flat = pool(hidden2_drop, pool3_fmaps, 'pool3',)

hidden4_drop = conv(pool3_flat, conv4_fmaps, training, 'conv4',)
pool5_flat = pool(hidden4_drop, pool5_fmaps, 'pool5',
flatten = True )

n_fc1 = 200
num_classes = 6

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool5_flat, n_fc1, name="fc1")#, activation = tf.nn.elu)
    bn3 = tf.nn.elu(batch_norm_layer(fc1))
    #print('fc1: ', fc1.get_shape())

with tf.name_scope("output"):
    logits_before_bn = tf.layers.dense(bn3, num_classes, name="output")
    logits = batch_norm_layer(logits_before_bn)
    #print('logits: ', logits.get_shape())
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
    preds  = tf.cast( tf.argmax(logits, 1), tf.int32 )
    mislabeled = tf.not_equal( preds, input_labels_batch )
    mislabeled_filenames = tf.cast( tf.boolean_mask( input_fnames, mislabeled ), tf.string)

    # f_names | label | class prob 1 | ... | class prob 6 |
    # creating this tensor in tf is super awk
    class_proba_list = [ input_fnames, tf.as_string(input_labels_batch), tf.as_string(mislabeled)  ]
    #print(input_fnames.get_shape(), tf.as_string(input_labels_batch).get_shape())

    Y_proba_str = tf.as_string(Y_proba)

    class_proba_list += tf.unstack(Y_proba_str, axis = 1)
    class_proba = tf.stack(class_proba_list)

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

    # Compute a per-batch confusion matrix
    batch_confusion = tf.confusion_matrix(input_labels_batch, preds,
                                             num_classes=num_classes,
                                             name='batch_confusion')
    # Create an accumulator variable to hold the counts
    confusion = tf.Variable( tf.zeros( [num_classes, num_classes],
                                      dtype=tf.int32 ),
                                      name='confusion' )
    # Create the update op for doing a "+=" accumulation on the batch
    confusion_update = confusion.assign( confusion + batch_confusion )
    test_op = tf.group(confusion_update)

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

loss_record = tf.summary.scalar('loss', loss)
file_writer = tf.summary.FileWriter(logdir + 'loss', tf.get_default_graph())

acc_record = tf.summary.scalar('accuracy', accuracy)
train_acc_writer = tf.summary.FileWriter(logdir + 'train', tf.get_default_graph())
test_acc_writer = tf.summary.FileWriter(logdir + 'test', tf.get_default_graph())

os.makedirs(logdir + 'tensor_logs')

write_op = tf.summary.merge_all() # put into session.run!

def record(sess, step, epoch, n_epochs):
    if step % 200 == 0: # 8 updates per epoch
        #if i % 30 == 0:
        #    print(epoch, step, acc_train)
        l, acc_train = sess.run([ loss_record, acc_record ], feed_dict = {training: True} )
        file_writer.add_summary(l, step)
        train_acc_writer.add_summary(acc_train, step)

        acc_test = acc_record.eval(feed_dict = { training: False} )
        test_acc_writer.add_summary(acc_test, step)

    if epoch == n_epochs -1 and step % 200 == 0:
        # test_op updates confusion matrix: sess -> test_op -> confusion_update -> assign
        _, test_batch_stats = sess.run( [test_op, class_proba],
                               feed_dict={training: False})

        np.save('%stensor_logs/test_batch_step_%s' %(logdir, step), test_batch_stats)

    if epoch == n_epochs - 1 and step % 400 == 0:
       print(confusion.eval())

training_set_size = len([
name for name in os.listdir(train_path) if os.path.isfile(train_path + name)])

def train( restore = False):
    n_epochs = 4
    with tf.Session() as sess:
        init.run()

        if restore:
            saver.restore(sess, './saved_model')

        tf.train.start_queue_runners(sess=sess)
        step = 0
        for epoch in range(n_epochs):
            for i in range(training_set_size // 64):
                record(sess, step, epoch, n_epochs)
                _, loss_val = sess.run(
                [training_op, write_op], feed_dict={training: True} )
                step += 1

            save_path = saver.save(sess, "./saved_model")
    file_writer.close()

train()
