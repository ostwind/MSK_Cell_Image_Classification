from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

import numpy as np
import tensorflow as tf
from sample_feed import inputs
from functools import partial
import os
import math
from encoder import encoder
from decoder import decoder

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original_imgs_path = os.path.join(dir, 'original/') # cut first array from /real_original/
labeled_path = os.path.join(dir, 'labeled/')
unlabeled = os.path.join(dir, 'unlabeled/')
test_path = os.path.join(dir, 'test_set/')

with tf.name_scope('input'):
    # two separate queues
    labeled_batch, labeled_labels_batch, labeled_fnames = inputs(labeled_path)
    test_batch, test_labels_batch, test_fnames = inputs(test_path)
    unlabeled_batch, unlabeled_labels_batch, unlabeled_fnames = inputs(unlabeled)

    labeled = tf.placeholder_with_default(False, shape = (), name = 'labeled_bool')
    training = tf.placeholder_with_default(True, shape=(), name = 'train_bool')

    input_batch, input_labels_batch, input_fnames = tf.cond(labeled,
    lambda: tf.cond( training, lambda: (labeled_batch, labeled_labels_batch, labeled_fnames),
    lambda: (test_batch, test_labels_batch, test_fnames) ),
    lambda: (unlabeled_batch, unlabeled_labels_batch, unlabeled_fnames))

num_classes = 5

training_set_size = len([
name for name in os.listdir(labeled_path) if os.path.isfile(labeled_path + name)])

class Ladder:
    def __init__(self, n_epochs, encoder_layer_dims, decoder_layer_dims, noise_std):
        self.n_epochs = n_epochs
        self.batch_size = tf.shape(input_batch)
        self.optimizer = tf.train.AdamOptimizer()

        self.encoder = encoder(encoder_layer_dims,
        activation_types = ['elu', 'elu', 'elu', 'softmax'],
        train_batch_norms = [1,1,1,1], noise_std = noise_std, batch_size = self.batch_size)

        self.decoder = decoder(decoder_layer_dims, 0)

        with tf.name_scope('noisy_unlabeled'):
            noisy_unlabeled_logits = self.encoder.forward_noise(input_batch)
            with tf.control_dependencies([noisy_unlabeled_logits]):
                self.tilde_zs = self.encoder.collect_stored_var(var_name = 'buffer_tilde_z')

        with tf.name_scope('clean_unlabeled'):
            clean_unlabeled_logits = self.encoder.forward_clean(input_batch)
            with tf.control_dependencies([clean_unlabeled_logits]):
                self.z_pre_layers = self.encoder.collect_stored_var(var_name = 'buffer_z_pre')
                self.z_layers = self.encoder.collect_stored_var(var_name = 'buffer_z')

        with tf.name_scope('eval'):
            clean_test_logits = self.encoder.forward_clean(input_batch)
            clean_test_logits = tf.reshape( clean_test_logits, shape = [-1, 2500*5] )

            correct_len_logits = tf.layers.dense(clean_test_logits, num_classes, name="output")
            preds = tf.cast(tf.argmax(correct_len_logits, 1), tf.int32)

            with tf.control_dependencies([preds]):
                batch_confusion = tf.confusion_matrix(input_labels_batch, preds,
                                                          num_classes=num_classes,
                                                          name='batch_confusion')
                self.confusion = tf.get_variable('confusion', shape = [num_classes, num_classes],
                dtype = tf.int32,
                initializer = tf.zeros_initializer())
                # Create the update op for doing a "+=" accumulation on the batch
                self.confusion_update = self.confusion.assign( self.confusion + batch_confusion )
                #self.confusion_update = tf.add(confusion, batch_confusion)

        with tf.name_scope('decoder_UnsupervisedLoss'):
            self.hat_z_array = self.decoder.pre_reconstruction(self.tilde_zs, noisy_unlabeled_logits)
            self.normed_hat_zs = self.decoder.reconstruction(self.hat_z_array, self.z_pre_layers)
            # denoising cost starts from top decode layer
            denoise_cost = [0.1, 0.1, 10, 1000]
            #most import to denoise bottom decode layer according to abi [1000, 10, 0.1, 0.1 ]
            self.unsupervised_loss = 0

            assert len(self.z_layers) == len(self.normed_hat_zs)
            assert len(self.z_layers) == len(denoise_cost)

            for cost_lambda, z, hat_z in zip(
            denoise_cost, self.z_layers, self.normed_hat_zs ):
                # process the entire batch?
                self.unsupervised_loss += cost_lambda * tf.losses.mean_squared_error(hat_z, z)

        with tf.name_scope('noisy_labeled_SupervisedLoss'):#first step repeated
            noisy_labeled_logits = self.encoder.forward_noise(input_batch)
            pooled = tf.nn.max_pool(noisy_labeled_logits,
            ksize= [1,2,2,1], strides=[1,2,2,1], padding= 'VALID')
            fc1 = tf.layers.dense(pooled, num_classes, name="fc1", activation = tf.nn.elu)
            output = tf.reshape( fc1, shape = [64, -1] )


            with tf.control_dependencies([output]):
                #self.xentropy not updating
                self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=input_labels_batch)
                self.supervised_loss = tf.reduce_mean(self.xentropy)
                self.print_labels = self.xentropy
                #self.test_op = self.optimizer.minimize(self.supervised_loss)

        with tf.name_scope('total_loss'):
            #with tf.control_dependencies([]):
            self.loss = self.supervised_loss + self.unsupervised_loss
            self.test_op = self.optimizer.minimize(self.loss)

    def launch(self):
        self.init = tf.global_variables_initializer()
        # multi-threads needed because  multi queues cause  tf to lock up
        # https://stackoverflow.com/questions/35414009/multiple-queues-causing-tf-to-lock-up
        with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                             intra_op_parallelism_threads=4)) as sess:
            self.init.run()
            tf.train.start_queue_runners(sess = sess)

            step = 0
            for epoch in range(self.n_epochs):
                for i in range(training_set_size // 64):

                    # _, s_loss = sess.run(
                    # [self.test_op, self.supervised_loss],
                    # feed_dict = {labeled: True, training: True})

                    sess.run([self.tilde_zs, self.z_pre_layers, self.z_layers])

                    _, total_loss, s_loss = sess.run(
                    [self.test_op, self.loss, self.supervised_loss], feed_dict = {
                    labeled: True, training: True})

                    if step % 4 == 0:
                        print( step, s_loss, total_loss - s_loss, total_loss)

                    if step > 400 and step % 10:
                        sess.run(
                        [self.confusion_update], feed_dict = {labeled: True, training: False})
                        if step % 50 == 0:
                            print(self.confusion.eval())

                    step += 1

a_ladder = Ladder( 4, [10, 20, 40, num_classes], [num_classes, 40, 20, 10], 0.3)
a_ladder.launch()
