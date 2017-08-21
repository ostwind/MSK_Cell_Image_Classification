import tensorflow as tf



def var_const_init(const, shape, name, validate_shape = True):
    return tf.Variable(const * tf.ones_like(shape, dtype=tf.float32), validate_shape = validate_shape, name = name)
def var_gauss_init(shape, name):
    return tf.Variable(tf.random_normal(shape, dtype=tf.float32), name=name) #/math.sqrt(shape[0])


# batch norm w/ no gamma scale or beta shift
def batch_normalize( tensor ):
    mean, variance = tf.nn.moments(tensor, axes = [0]) # axes = [0] for batch handling
    z = tf.nn.batch_normalization(tensor, mean = mean, variance = variance,
    offset = None, scale = None, variance_epsilon = 10e-6 )
    return z

def conv(input_tensor, filters,
kernel_size = 3, strides = 1, padding = "SAME", name = 'f' ):

    # use_bias <- False, paper asks for matrix multiplication Wh in the case of MLP
    return tf.layers.conv2d(
    input_tensor, filters = filters, kernel_size = kernel_size, strides = strides,
    padding = "SAME", name = name, use_bias = False)

def downsample(encoder_output_logit, num_classes):
    pooled = tf.nn.max_pool(encoder_output_logit,
    ksize= [1,2,2,1], strides=[1,2,2,1], padding= 'VALID')
    pooled = tf.reshape( pooled, shape = [64, 3125] )
    pooled = tf.layers.dense(pooled, 200, name="fc1", activation = tf.nn.elu)
    #reshape 64 may also occur here
    output = tf.layers.dense(pooled, num_classes, name="fc2", activation = tf.nn.elu)
    return output
