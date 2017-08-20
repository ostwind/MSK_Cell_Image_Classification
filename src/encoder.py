import numpy as np
import tensorflow as tf
from sample_feed import inputs
from functools import partial
from model_util import var_const_init, var_gauss_init, batch_normalize, conv
import os
import math

class encode_layer():
    def __init__(self, d_out, activation_type,
    train_bn_scaling, noise, batch_size):
        #self.data = data
        #self.d_in = d_in
        self.d_out = d_out
        self.activation_type = activation_type
        self.train_bn_scaling = train_bn_scaling
        self.noise = noise
        self.batch_size = batch_size

        # shift and scale vars after Batch Normalization
        # elu activation: only beta shift is computed
        # softmax: beta, gamma are trained
        self.bn_beta = var_const_init(0, [d_out], 'beta')
        if self.train_bn_scaling:
            self.bn_gamma = var_const_init(1, [d_out], 'gamma')

        # store z_pre, z to be used in calculation of reconstruction cost
        self.buffer_z = None
        self.buffer_z_pre = None
        # tilde_z will be used by decoder for reconstruction
        self.buffer_tilde_z = None

    def get(self, var_name_str):
        return eval('self.'+var_name_str)

    def _post_bn_shift_scale(self, x):
        t = x + self.bn_beta
        if self.train_bn_scaling:
            t = tf.multiply(t, self.bn_gamma)
        return t

    def _activation(self, h):
        if self.activation_type == 'elu':
            return tf.nn.elu(h)
        if self.activation_type == 'softmax':
            #with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            #    h = tf.layers.dense(h, 200, name = 'fc_end')
            #h = tf.nn.max_pool(h, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
            return h#tf.reshape( h, shape = [64, -1] ) #tf.nn.softmax(h)
        raise Exception('activation func %s not found ' %(self.activation_type))

    def forward_clean(self, h):
        # convolution version of linear layer op: W_l h_(l-1)
        self.buffer_z_pre = conv(h, filters = self.d_out)#, name = str(self.d_out))
        #print('z_pre: ', self.buffer_z_pre)

        # batchnorm w/ no shift or scale
        self.buffer_z = batch_normalize(self.buffer_z_pre)
        #print('z: ', self.buffer_z)

        # activation(gamma_l (elemwise mul) (z_l + beta_l))
        h_post = self._post_bn_shift_scale(self.buffer_z)
        h_post = self._activation( h_post )
        return h_post

    def forward_noise(self, tilde_h):
        # use bias or no? str(self.d_out)
        z_pre = conv(tilde_h, filters = self.d_out)#, name = str(self.d_out))

        z_pre_norm = batch_normalize(z_pre)
        noise = tf.random_normal( tf.shape(z_pre_norm), mean = 0, stddev=self.noise )
        self.buffer_tilde_z = z_pre_norm + noise

        z = self._post_bn_shift_scale(self.buffer_tilde_z)
        tilde_h_post = self._activation(z)
        return tilde_h_post

class encoder():
    def __init__(self, d_encoders, activation_types, train_batch_norms, noise_std, batch_size):
        self.noise = noise_std
        self.encoder_layers = []
        self.first_layer_shape = batch_size

        for i in range(len(d_encoders)):

            d_output = d_encoders[i]
            activation = activation_types[i]
            train_batch_norm = train_batch_norms[i]
            self.encoder_layers.append( encode_layer(
            d_output, activation, train_batch_norm, noise_std, batch_size) )

    def forward_clean(self, x):
        h = x
        for encoder_layer in self.encoder_layers:
            h = encoder_layer.forward_clean(h)
        return h

    def forward_noise(self, x):
        noise = tf.random_normal( tf.shape(x), mean = 0, stddev=self.noise )
        h = x + noise
        for encoder_layer in self.encoder_layers:
            h = encoder_layer.forward_noise(h)
            print('encoder noise forward: ', h.shape)
        return h

    def collect_stored_var(self, var_name, reverse = True ):
        layer_vars_collected = []
        for encoder_layer in self.encoder_layers:
            layer_vars_collected.append( encoder_layer.get(var_name) )
        if reverse:
            layer_vars_collected.reverse()
        return layer_vars_collected
