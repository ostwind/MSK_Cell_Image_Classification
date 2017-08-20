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
    return tf.layers.conv2d(
    input_tensor, filters = filters, kernel_size = kernel_size, strides = strides,
    padding = "SAME")#, name = name, reuse= True)
