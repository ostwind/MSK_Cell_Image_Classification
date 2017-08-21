import numpy as np
import tensorflow as tf
from sample_feed import inputs
from functools import partial
from model_util import var_const_init, var_gauss_init, batch_normalize, conv
import os
import math

class encode_layer():
    def __init__(self, d_out, activation_type, noise, batch_size):
        self.d_out = d_out
        self.activation_type = activation_type
        self.noise = noise
        self.batch_size = batch_size

        # shift and scale vars after Batch Normalization
        # elu activation: only beta shift is computed
        # softmax: beta, gamma are trained
        self.bn_beta = var_const_init(0, [d_out], 'beta')
        if self.activation_type == 'softmax':
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
        if self.activation_type == 'softmax':
            t = tf.multiply(t, self.bn_gamma)
        return t

    def _activation(self, h):
        if self.activation_type == 'elu':
            return tf.nn.elu(h)
        if self.activation_type == 'softmax':
            # to maintain encode/decode layer dim symmetry, actual prediction
            # occurs in ladder.py
            return h
        raise Exception('activation func %s not found ' %(self.activation_type))

    def forward_clean(self, h, first_pass = False):
        # convolution version of linear layer op: W_l h_(l-1)
        with tf.variable_scope(str(self.d_out), reuse = not first_pass) as scope:
            print('clean', self.d_out, not first_pass)
            self.buffer_z_pre = conv(h, filters = self.d_out, name = str(self.d_out))

        # batchnorm w/ no shift or scale
        self.buffer_z = batch_normalize(self.buffer_z_pre)
        #print('z: ', self.buffer_z)

        # activation(gamma_l (elemwise mul) (z_l + beta_l))
        h_post = self._post_bn_shift_scale(self.buffer_z)
        h_post = self._activation( h_post )
        return h_post

    def forward_noise(self, tilde_h, first_pass = False):
        # use bias or no? str(self.d_out)
        with tf.variable_scope(str(self.d_out), reuse = not first_pass) as scope:
            print('noise', self.d_out, not first_pass)
            z_pre = conv(tilde_h, filters = self.d_out, name = str(self.d_out))

        z_pre_norm = batch_normalize(z_pre)
        noise = tf.random_normal( tf.shape(z_pre_norm), mean = 0, stddev=self.noise )
        self.buffer_tilde_z = z_pre_norm + noise

        z = self._post_bn_shift_scale(self.buffer_tilde_z)
        tilde_h_post = self._activation(z)
        return tilde_h_post

class encoder():
    def __init__(self, d_encoders, activation_types, noise_std, batch_size):
        self.noise = noise_std
        self.encoder_layers = []
        self.first_layer_shape = batch_size

        for i in range(len(d_encoders)):

            d_output = d_encoders[i]
            activation = activation_types[i]
            self.encoder_layers.append( encode_layer(
            d_output, activation, noise_std, batch_size) )

    def forward_clean(self, x, first_pass = False):
        h = x
        for encoder_layer in self.encoder_layers:
            h = encoder_layer.forward_clean(h, first_pass)
        return h

    def forward_noise(self, x, first_pass = False):
        noise = tf.random_normal( tf.shape(x), mean = 0, stddev=self.noise )
        h = x + noise
        for encoder_layer in self.encoder_layers:
            h = encoder_layer.forward_noise(h, first_pass)
            #print('encoder noise forward: ', h.shape)
        return h

    def collect_stored_var(self, var_name, reverse = True ):
        layer_vars_collected = []
        for encoder_layer in self.encoder_layers:
            layer_vars_collected.append( encoder_layer.get(var_name) )
        if reverse: # as decoder input
            layer_vars_collected.reverse()
        return layer_vars_collected
