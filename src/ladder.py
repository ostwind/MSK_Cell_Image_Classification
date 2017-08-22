from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

import numpy as np
import tensorflow as tf
from functools import partial
import os
import math
from encoder import encoder
from decoder import decoder
from model_util import downsample, inputs

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
reset_graph()

class Ladder:
    def __init__(self, n_epochs, encoder_layer_dims, decoder_layer_dims,
    activation_types, denoise_costs, noise_std):
        self.n_epochs = n_epochs
        self.denoise_costs = denoise_costs

        self.batch_size = tf.shape(input_batch)
        self.optimizer = tf.train.AdamOptimizer()

        self.encoder = encoder(encoder_layer_dims,
        activation_types = activation_types,
        noise_std = noise_std, batch_size = self.batch_size)

        self.decoder = decoder(decoder_layer_dims)

        with tf.name_scope('noisy_labeled_SupervisedLoss'):
            # first pass is a bool that sets tf.variable_scope's reuse param
            noisy_labeled_logits = self.encoder.forward_noise(input_batch,
            labeled = True, first_pass = True)

            with tf.control_dependencies([noisy_labeled_logits]):
                self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=noisy_labeled_logits, labels=input_labels_batch)
                self.supervised_loss = tf.reduce_mean(self.xentropy)

        with tf.name_scope('noisy_unlabeled'):
            noisy_unlabeled_logits = self.encoder.forward_noise(
            input_batch, labeled = False)
            with tf.control_dependencies([noisy_unlabeled_logits]):
                self.tilde_zs = self.encoder.collect_stored_var(var_name = 'buffer_tilde_z')

        with tf.name_scope('clean_unlabeled'):
            clean_unlabeled_logits = self.encoder.forward_clean(input_batch,
            labeled = False )
            with tf.control_dependencies([clean_unlabeled_logits]):
                self.z_pre_layers = self.encoder.collect_stored_var(var_name = 'buffer_z_pre')
                self.z_layers = self.encoder.collect_stored_var(var_name = 'buffer_z')

        with tf.name_scope('decoder_UnsupervisedLoss'):
            self.hat_z_array = self.decoder.pre_reconstruction(self.tilde_zs, self.encoder.buffer_h)
            self.normed_hat_zs = self.decoder.reconstruction(self.hat_z_array, self.z_pre_layers)

            assert len(self.z_layers) == len(self.normed_hat_zs)
            assert len(self.z_layers) == len(self.denoise_costs)

            self.unsupervised_loss = 0
            for cost_lambda, z, hat_z in zip(
            self.denoise_costs, self.z_layers, self.normed_hat_zs ):
                # process the entire batch?
                self.unsupervised_loss += cost_lambda * tf.losses.mean_squared_error(hat_z, z)

        with tf.name_scope('total_loss'):
            self.loss = self.supervised_loss + self.unsupervised_loss
            self.test_op = self.optimizer.minimize(self.loss)

        with tf.name_scope('eval'):
            clean_test_logits = self.encoder.forward_clean(input_batch,
            labeled = True)
            self.preds = tf.cast(tf.argmax( clean_test_logits, 1), tf.int32)

            with tf.control_dependencies([self.preds]):
                self.batch_confusion = tf.confusion_matrix(input_labels_batch, self.preds,
                                                          num_classes=num_classes,
                                                          name='batch_confusion')
                self.confusion = tf.get_variable('confusion', shape = [num_classes, num_classes],
                dtype = tf.int32,
                initializer = tf.zeros_initializer())

                # Create the update op for doing a "+=" accumulation on the batch
                self.confusion_update = self.confusion.assign( self.confusion + self.batch_confusion )

    def _create_record(self):
        with tf.name_scope('records'):
            self.loss_record = tf.summary.scalar('loss', self.loss)
            self.loss_writer = tf.summary.FileWriter(logdir + 'loss', tf.get_default_graph())

            self.acc_record = tf.summary.scalar('accuracy', self.accuracy)
            self.test_acc_writer = tf.summary.FileWriter(logdir + 'test', tf.get_default_graph())

            os.makedirs(logdir + 'tensor_logs')
            self.write_op = tf.summary.merge_all() # put into session.run!

    # def _record(sess, step, epoch, n_epochs):
    #     if step % 100 = 0:
    #         acc_test = sess.run([self.acc_record, self.write_op],
    #         feed_dict = {labeled: True, training: False} )
    #         self.test_acc_writer.add_summary(acc_test, step)
    #         self.

    def launch(self, restore = True):
        print( 'steps per epoch: ', training_set_size // 64 )
        self.init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # multi-threads needed because  multi queues cause  tf to lock up
        # https://stackoverflow.com/questions/35414009/multiple-queues-causing-tf-to-lock-up
        with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                             intra_op_parallelism_threads=4)) as sess:
            self.init.run()
            #self._create_record()
            if restore and os.path.isfile('./saved_ladder') :
               saver.restore(sess, './saved_ladder')
            tf.train.start_queue_runners(sess = sess)
            step = 0

            for epoch in range(self.n_epochs):
                for i in range(training_set_size // 64):

                    s_loss = sess.run([self.supervised_loss])

                    sess.run([self.tilde_zs, self.z_pre_layers, self.z_layers],
                    feed_dict = {labeled: False, training: True})

                    _, total_loss = sess.run([self.test_op, self.loss])

                    if step % 20 == 0:
                        print( step, s_loss[0], total_loss - s_loss[0], total_loss)

                    if total_loss < 80 or step > 5100:
                        _, matrix = sess.run(
                        [self.confusion_update, self.confusion],
                        feed_dict = {labeled: True, training: False})
                        print(matrix)

                    if step % 200 == 0:
                        save_path = saver.save(sess, "./saved_ladder")

                    step += 1

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

    labeled = tf.placeholder_with_default(True, shape = (), name = 'labeled_bool')
    training = tf.placeholder_with_default(True, shape=(), name = 'train_bool')

    input_batch, input_labels_batch, input_fnames = tf.cond(labeled,
    lambda: tf.cond( training, lambda: (labeled_batch, labeled_labels_batch, labeled_fnames),
    lambda: (test_batch, test_labels_batch, test_fnames) ),
    lambda: (unlabeled_batch, unlabeled_labels_batch, unlabeled_fnames))

num_classes = 5

training_set_size = len([
name for name in os.listdir(labeled_path) if os.path.isfile(labeled_path + name)])

# do not let 2 layer dims be the same
encoder_dims = [20, 40, 80]#, 80]
decoder_dims = list(reversed(encoder_dims))
activation_types = ['elu', 'elu', 'softmax']
# denoising cost starts from top decode layer
denoise_cost = [0.1, 10, 1000]

print(encoder_dims, decoder_dims)
a_ladder = Ladder( 4, encoder_dims, decoder_dims,
activation_types, denoise_cost, 0.3)
a_ladder.launch()
