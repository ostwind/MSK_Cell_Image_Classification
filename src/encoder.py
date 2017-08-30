import numpy as np
import tensorflow as tf
from sample_feed import inputs
from functools import partial
from model_util import var_const_init, var_gauss_init, batch_normalize, conv, downsample
import os
import math

class encode_layer():
    def __init__(self, d_out, activation_type, noise):
        assert activation_type in ['elu', 'softmax'], print(
        'activation func %s not found ' %activation_type )
        self.d_out = d_out
        self.activation_type = activation_type
        self.noise = noise

        # shift and scale vars after Batch Normalization
        # elu activation: only beta shift is computed
        # softmax: beta, gamma are trained
        self.bn_beta = var_const_init(0, [d_out], 'beta')
        #if self.activation_type == 'softmax':
        self.bn_gamma = var_const_init(1, [d_out], 'gamma')

        # store z_pre, z to be used in calculation of reconstruction cost
        self.buffer_z = None
        self.buffer_z_pre = None
        # tilde_z will be used by decoder for reconstruction
        self.buffer_tilde_z = None
        # h passed as first u into decoder
        self.buffer_h = None
    def get(self, var_name_str):
        return eval('self.'+var_name_str)

    def _post_bn_shift_scale(self, x):
        t = x + self.bn_beta
        #if self.activation_type == 'softmax':
        t = tf.multiply(t, self.bn_gamma)
        return t

    def _activation(self, h, first_pass = False):
        if self.activation_type == 'elu':
            return tf.nn.elu(h)
        if self.activation_type == 'softmax':
            # to maintain encode/decode layer dim symmetry, actual prediction
            # occurs in ladder.py
            # no technology to unpool (arg max pool don't work on cpu), so save for decoder
            self.buffer_h = h

            with tf.variable_scope('downsampling', reuse = not first_pass):
                pooled = tf.nn.max_pool(h,
                ksize= [1,2,2,1], strides=[1,2,2,1], padding= 'VALID')
                pooled = tf.reshape( pooled, shape = [64, 25*25*self.d_out] )
                #pooled = tf.layers.dense(pooled, 200, name = 'fc', activation = tf.nn.elu)
                # placing nonlinear activation on this layer is bad
                preds = tf.layers.dense( pooled , 5, name = 'preds')
            return preds

    def forward_clean(self, h, labeled, first_pass = False):
        # convolution version of linear layer op: W_l h_(l-1)
        with tf.variable_scope(str(self.d_out), reuse = not first_pass) as scope:
            #print('clean', self.d_out, not first_pass)
            self.buffer_z_pre = conv(h, filters = self.d_out, name = str(self.d_out))

        # batchnorm w/ no shift or scale
        self.buffer_z = batch_normalize(self.buffer_z_pre)
        #print('z: ', self.buffer_z)

        # unlabeled logits not used, skip ops after filling buffer
        if not labeled and self.activation_type == 'softmax':
            return tf.ones_like([0], dtype=tf.float32)
        # activation(gamma_l (elemwise mul) (z_l + beta_l))
        h_post = self._post_bn_shift_scale(self.buffer_z)
        h_post = self._activation( h_post, first_pass )
        return h_post

    def forward_noise(self, tilde_h, labeled, first_pass = False):
        with tf.variable_scope(str(self.d_out), reuse = not first_pass) as scope:
            # use bias or no? str(self.d_out)
            #print('noise', self.d_out, not first_pass)
            z_pre = conv(tilde_h, filters = self.d_out, name = str(self.d_out))

        z_pre_norm = batch_normalize(z_pre)
        noise = tf.random_normal( tf.shape(z_pre_norm), mean = 0, stddev=self.noise )
        self.buffer_tilde_z = z_pre_norm + noise

        if not labeled and self.activation_type == 'softmax':
            return tf.ones_like([0], dtype=tf.float32)

        z = self._post_bn_shift_scale(self.buffer_tilde_z)
        tilde_h_post = self._activation(z, first_pass)
        return tilde_h_post

class encoder():
    def __init__(self, d_encoders, activation_types, noise_std):
        self.noise = noise_std
        self.encoder_layers = []

        for i in range(len(d_encoders)):
            d_output = d_encoders[i]
            activation = activation_types[i]
            self.encoder_layers.append( encode_layer(
            d_output, activation, noise_std) )
        self.buffer_h = None

    def forward_clean(self, x, labeled, first_pass = False):
        h = x
        for encoder_layer in self.encoder_layers:
            h = encoder_layer.forward_clean(h, labeled, first_pass)
        return h

    def forward_noise(self, x, labeled, first_pass = False):
        noise = tf.random_normal( tf.shape(x), mean = 0, stddev=self.noise )
        h = x + noise
        for encoder_layer in self.encoder_layers:
            h = encoder_layer.forward_noise(h, labeled, first_pass)
            #print('encoder noise forward: ', h.shape)

        if labeled: # there is only 1 noisy labeled pass
            self.buffer_h = self.encoder_layers[-1].buffer_h
        return h

    def collect_stored_var(self, var_name, reverse = True ):
        layer_vars_collected = []
        for encoder_layer in self.encoder_layers:
            layer_vars_collected.append( encoder_layer.get(var_name) )
        if reverse: # as decoder input, appearance of tensors reversed
            layer_vars_collected.reverse()
        return layer_vars_collected
