''' common functions for the model pipeline such as:
    1. backpropgation variables
    2. batch normalize and conv layers
    3. 'objecting' input samples and queueing them into queue runners as batches
'''

import tensorflow as tf
import os
from random import shuffle

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
    padding = padding, name = name)

def downsample(encoder_output_logit, num_classes):
    pooled = tf.nn.max_pool(encoder_output_logit,
    ksize= [1,2,2,1], strides=[1,2,2,1], padding= 'VALID')
    pooled = tf.reshape( pooled, shape = [64, 30*30*200] )
    return output

''' SAMPLE QUEUE GENERATION AND DEQUEUE FOR TF ladder.py
    read_my_data <= inputs (form queues of ImageRecord objects)
'''

def read_my_data(filename_queue):

    class ImageRecord(object):
        def __init__(self):
            # Dimensions of the images in the dataset.
            self.height = 50
            self.width = 50
            self.depth = 32

    result = ImageRecord()
    label_bytes = 1
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    assert record_bytes == (50*50*32)+1

      # Read a record, getting filenames from the filename_queue.  No
      # header or footer in the binary, so we leave header_bytes
      # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

      # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

      # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

      # The remaining bytes after the label represent the image, which we reshape
      # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
    [result.depth, result.height, result.width])
      # Convert from [depth, height, width] to [height, width, depth].
    result.imagematrix = tf.transpose(depth_major, [1, 2, 0])
    return result

def _collect_paths(input_directory):
    string_tensor = []
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            if '.bin' in f:
                string_tensor.append(input_directory+f)
    return string_tensor

def inputs(input_directory, batch_size = 64, num_epochs = None):
    string_tensor = _collect_paths(input_directory)
    shuffle(string_tensor)

    # if num_epochs met, automatic halt occurs. If none, manual halt required
    filename_queue = tf.train.string_input_producer(
    string_tensor, num_epochs = num_epochs, shuffle = False)

    image_record = read_my_data(filename_queue)
    image_record.imagematrix = tf.cast(image_record.imagematrix, tf.float32)

    min_after_dequeue = 10000
    capacity = min_after_dequeue +  3 * batch_size

    example_batch, label_batch, fname_batch = tf.train.shuffle_batch(
    [image_record.imagematrix, image_record.label, image_record.key],
    batch_size = batch_size, capacity = capacity,
    min_after_dequeue = min_after_dequeue, allow_smaller_final_batch = True)

    label_batch = tf.reshape(label_batch, shape=[-1])
    return example_batch, label_batch, fname_batch
