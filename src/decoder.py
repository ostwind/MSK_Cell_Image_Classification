import numpy as np
import tensorflow as tf
from model_util import var_const_init, var_gauss_init, batch_normalize, conv

class decode_layer():
    def __init__(self, d_in, d_out ):
        self.d_out = d_out
        self.d_in = d_in
        # buffer for hat_z_l to be used for cost calculation
        self.buffer_hat_z = None

    def g(self, tilde_z, u): # denoising func
        # ok if reconstruction calc occurs w/ shape (64, 50, 50, d_in)?
        batch_size, h, w, c = tf.shape(input_batch)
        input_shape = np.zeros((batch_size, h, w, self.d_in), dtype=np.float32)

        a1 = var_const_init(0, input_shape,  'a1')
        a2 = var_const_init(1, input_shape, 'a2')
        a3 = var_const_init(0, input_shape, 'a3')
        a4 = var_const_init(0, input_shape, 'a4')
        a5 = var_const_init(0, input_shape, 'a5')

        a6 = var_const_init(0, input_shape, 'a6')
        a7 = var_const_init(1, input_shape, 'a7')
        a8 = var_const_init(0, input_shape, 'a8')
        a9 = var_const_init(0, input_shape, 'a9')
        a10 = var_const_init(0, input_shape, 'a10')

        #tf.sigmoid was used in paper, better computation + perf w/ elu
        mu = tf.multiply( a1, tf.nn.elu( tf.multiply(a2, u) + a3)   ) + \
             tf.multiply(a4, u) + a5
        v = tf.multiply( a6, tf.nn.elu( tf.multiply(a7, u) + a8)   ) + \
             tf.multiply(a9, u) + a10

        hat_z = tf.multiply( tilde_z - mu, v ) + mu
        return hat_z

    def denoise(self, tilde_z, u):
        # how does weight sharing occur in the decoder?
        self.buffer_hat_z = self.g(tilde_z, u )

        if self.d_out: # not the last decode layer, so compute u for next layer
            t = tf.layers.conv2d_transpose(self.buffer_hat_z,
            filters = self.d_out, kernel_size = 3,
            padding = 'SAME', #use_bias = False,
            name = 'decoder%s' %(self.d_in) )

            u_below = batch_normalize( t )
            return u_below
        else:
            return None

class decoder():
    def __init__(self, d_decoders):
        self.decode_layers = []
        for i in range(len(d_decoders)):
            d_in = d_decoders[ i  ]

            if i < len(d_decoders) - 1 :
                d_output = d_decoders[i + 1]
                self.decode_layers.append( decode_layer(d_in, d_output ) )

            else:
                self.decode_layers.append( decode_layer(d_in, None)  )

    def pre_reconstruction(self, tilde_zs, encoder_logit): #encoder_logit <- tilde_h^L
        assert len(tilde_zs) == len(self.decode_layers), print(
        'len of tilde zs == %s != %s == len of decoders' %(len(tilde_zs), len(self.decode_layers)))

        hat_z_array = []
        u = batch_normalize(encoder_logit)

        for i in range(len(self.decode_layers)):
            tilde_z = tilde_zs[i]
            decode_layer = self.decode_layers[i]

            u = decode_layer.denoise(tilde_z, u)
            hat_z_array.append(decode_layer.buffer_hat_z)

        return hat_z_array

    def reconstruction(self, hat_z_array, z_pre_layers):
        assert len(hat_z_array) == len(z_pre_layers), print('%s, %s' %(len(hat_z_arrays), len(z_pre_layers)))
        normed_hat_zs = []
        for i, (hat_z, z_pre) in enumerate(zip(hat_z_array, z_pre_layers)):

            # normalizing using mean and variance of clean target
            mean, variance = tf.nn.moments(z_pre, axes = [0])
            normed_hat_z = tf.nn.batch_normalization(hat_z, mean = mean, variance = variance,
            offset = None, scale = None, variance_epsilon = 10e-6 )

            normed_hat_zs.append(normed_hat_z)
        return normed_hat_zs
