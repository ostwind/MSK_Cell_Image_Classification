import numpy as np
import tensorflow as tf
from model_util import var_const_init, var_gauss_init, batch_normalize, conv

class decode_layer():
    def __init__(self, d_in, d_out):
        self.d_out = d_out
        self.d_in = d_in
        # buffer for hat_z_l to be used for cost calculation
        self.buffer_hat_z = None


    def g(self, tilde_z, u, u_size): # denoising func
        original_shape = tf.shape(u)
        input_shape = np.zeros((64, 50, 50, self.d_in), dtype=np.float32)
        #print(tilde_z.shape, u.shape)

        a1 = var_const_init(0, input_shape,  'a1')
        #print(a1.shape, u.shape)
        a2 = var_const_init(1, input_shape, 'a2')
        a3 = var_const_init(0, input_shape, 'a3')
        a4 = var_const_init(0, input_shape, 'a4')
        a5 = var_const_init(0, input_shape, 'a5')

        a6 = var_const_init(0, input_shape, 'a6')
        a7 = var_const_init(1, input_shape, 'a7')
        a8 = var_const_init(0, input_shape, 'a8')
        a9 = var_const_init(0, input_shape, 'a9')
        a10 = var_const_init(0, input_shape, 'a10')

        mu = tf.multiply( a1, tf.sigmoid( tf.multiply(a2, u) + a3)   ) + \
             tf.multiply(a4, u) + a5
        v = tf.multiply( a6, tf.sigmoid( tf.multiply(a7, u) + a8)   ) + \
             tf.multiply(a9, u) + a10

        hat_z = tf.multiply( tilde_z - mu, v ) + mu
        hat_z = tf.reshape(hat_z, original_shape)
        return hat_z

    def denoise(self, tilde_z, u, u_size):
        self.buffer_hat_z = self.g(tilde_z, u, u_size )

        if self.d_out:
            #print(self.d_out)
            t = tf.layers.conv2d_transpose(self.buffer_hat_z, filters = self.d_out, kernel_size = 3,
            padding = 'SAME'  )
            #conv( self.buffer_hat_z, filters = self.d_out ) #tf.layers.dense(h, self.d_out)
            u_below = batch_normalize( t )
            return u_below
        else:
            return None

class decoder():
    def __init__(self, d_decoders, input_dim):

        self.decode_layers = []
        for i in range(len(d_decoders)):
            d_in = d_decoders[ i  ]

            if i < len(d_decoders) - 1 :
                #print( i + 1, len(d_decoders))
                d_output = d_decoders[i + 1]
                self.decode_layers.append( decode_layer(d_in, d_output ) )
            else:
                self.decode_layers.append( decode_layer(d_in, None)  )

        #self.decode_layers.append( decode_layer( d_in, None  ) )

    def pre_reconstruction(self, tilde_zs, encoder_logit): #encoder_logit -> tilde_h^L
        assert len(tilde_zs) == len(self.decode_layers), print(
        'len of tilde zs == %s != %s == len of decoders' %(len(tilde_zs), len(self.decode_layers)))

        hat_z_array = []
        u = batch_normalize(encoder_logit)

        for i in range(len(self.decode_layers)):
            tilde_z = tilde_zs[i]
            decode_layer = self.decode_layers[i]
            u = decode_layer.denoise(tilde_z, u, u_size = 0)
            hat_z_array.append(decode_layer.buffer_hat_z)

        return hat_z_array

    def reconstruction(self, hat_z_array, z_pre_layers):
        assert len(hat_z_array) == len(z_pre_layers), print('%s, %s' %(len(hat_z_arrays), len(z_pre_layers)))
        normed_hat_zs = []
        for i, (hat_z, z_pre) in enumerate(zip(hat_z_array, z_pre_layers)):
            mean, variance = tf.nn.moments(z_pre, axes = [0])
            #mean_var = var_const_init(mean, [1, z_pre.shape[0]], 'corrupted_ZPre_mean')

            normed_hat_z = tf.nn.batch_normalization(hat_z, mean = mean, variance = variance,
            offset = None, scale = None, variance_epsilon = 10e-6 )
            normed_hat_zs.append(normed_hat_z)
            #should mean + var be variables?
            #noise = tf.random_normal( z_pre.shape[0], mean = 0, stddev=1 )
            #variance_var = var_const_init(variance, [1, z_pre.shape[0]], 'corrupted_ZPre_var')
            #variance_var += noise
        return normed_hat_zs
