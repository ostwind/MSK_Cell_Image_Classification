from PIL import Image, ImageDraw
import numpy as np
import os
import pandas as pd
from tifffile import TiffFile
import tensorflow as tf
import random
import glob

img_length = 50

def center(box):
    c_x = (box[2] + box[0])/2
    c_y = (box[3] + box[1])/2
    return float(c_x), float(c_y)

def box(center, h = img_length):
    x1 = int( center[0]- (h // 2) )
    y1 = int( center[1]- (h // 2) )
    x2 = int( center[0] + (h // 2) )
    y2 = int( center[1] + (h // 2) )
    return x1, y1, x2, y2

def enforce_size(region):
    return box(center(region), h = img_length)

def is_good_size(region, greater_than = img_length, less_than = img_length):
    x1, y1, x2, y2 = region
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return False
    return (x2 - x1 <= less_than) and (x2 - x1 >= greater_than) and (y2 - y1 <= less_than) and (y2 - y1 >= greater_than)

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def tensor(spot, r):
    if not is_good_size(r, greater_than = 25, less_than = 70):
        return 'bad dim'
    x1, y1, x2, y2 = enforce_size(r)
    if not is_good_size([x1, y1, x2, y2]):
        return 'border cell that cannot make 50x50 img'
    tensor = []
    for subdir, dirs, files in os.walk(original_imgs_path):
        files.sort()
        for f in files:
            # extract spot integer from filenames e.g. -filename-_spot_00X.tif
            if spot ==  int(f.split('_')[-1].split('.')[0] ):
                im = Image.open(original_imgs_path + f)
                cell_im = im.crop( (x1, y1, x2, y2) )
                cell_matrix = (np.array(cell_im))/255
                #print(cell_matrix)
                tensor.append(cell_matrix)
    tensor = np.array(tensor).astype(np.uint8).transpose()
    #tensor = normalize(tensor)
    return tensor

# creates a (N X M X D) tensor from region r = (x1, y1, x2, y2)
def load(read_from_path = ''):
    if read_from_path:
        return np.load(read_from_path)

def rotate(tensor): # dont rotate if there are enough samples
    if type(tensor) == str:
        return 'bad dim'

    rotations = [tensor]
    for i in range(3):
        rotations.append(np.rot90( rotations[-1], axes = (0,1)))
    return rotations

def save(region, spot, cell_id, label, output_dir ):
    tensors = rotate(tensor(spot, region))
    if type(tensors) == str:
        return 'bad dim'

    i = 0
    for t in tensors:
        if t.shape != (img_length, img_length, 23):
            print(t.shape)
            continue
        filename = '%s_%s_%s.bin' %(cell_id, label, i)
        rotated_tensor = np.insert( t, 0, int(label) )
        rotated_tensor = rotated_tensor.flatten()
        rotated_tensor.tofile(output_dir + filename)
        i += 1
        #        mo = Image.fromarray( (t[:,:,10] ))
        #        mo.save(output_dir + '%s_%s_%s.png' %(cell_id, label, i), 'PNG')


def gen_tensors(spot, input_directory, output_directory, test_directory, metadata):
    i = 0
    s = np.random.uniform(0, 1, len(metadata.index))
    for i, row in metadata.iterrows():
        xmin = metadata.ix[i,'XMin']
        ymin = metadata.ix[i,'YMin']
        xmax = metadata.ix[i,'XMax']
        ymax = metadata.ix[i,'YMax']
        cell_id = metadata.ix[i, 'Object Id']
        cd4_intensity = metadata.ix[i, 'Marker 2 Intensity']
        cd4_pos = metadata.ix[i, 'Marker 2 Positive']
        sox10_intensity = metadata.ix[i, 'Marker 8 Intensity']
        sox10_pos = metadata.ix[i, 'Marker 8 Positive']
        #cd8_pos = meta

        path_to_write = output_directory
        if s[i] < 0.3: # train/test split at 70/30
            path_to_write = test_directory

        label = 0
        if int(cd4_pos) == 1 and int(sox10_pos) == 0:
            label = 1 # cd4+
        elif int(cd4_pos) == 0 and int(sox10_pos) == 1:
            label = 0 # tumor cell
        elif int(cd4_pos) == 0 and int(sox10_pos) == 0:
            label = 2 # cd8+
        else:
            continue

        save( [xmin, ymin, xmax, ymax], int(spot),
        cell_id, label, path_to_write)

        i += 1
        if i % 2000 == 0:
            print('files done ', i, 'spot: ', spot)

def empty_dir(path):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original_imgs_path = os.path.join(dir, 'original/') # cut first array from /real_original/
train_path = os.path.join(dir, 'tensors/')
test_path = os.path.join(dir, 'test_set/')

def read_my_data(filename_queue):#, fname_queue_for_dequeue):

    class ImageRecord(object):
        def __init__(self):
            # Dimensions of the images in the dataset.
            self.height = 50
            self.width = 50
            self.depth = 23

    result = ImageRecord()
    #result.filename = fname_queue_for_dequeue.dequeue()
    #print(result.filename, type(result.filename))

    label_bytes = 1
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    assert record_bytes == 57500+1

      # Read a record, getting filenames from the filename_queue.  No
      # header or footer in the binary, so we leave header_bytes
      # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    #print(type(result.key), result.key)
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


from random import shuffle

def inputs(input_directory, batch_size = 64, num_epochs = None):

    string_tensor = _collect_paths(input_directory)
    shuffle(string_tensor)

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

if __name__ == '__main__':
    empty_dir(train_path)
    empty_dir(test_path)

    # generate spotX.csv from /data/cell_metadata.csv

    for i in [2, 3, 4, 5]:
        spot = pd.read_csv('spot%s.csv' %(i))
        modified_spot = spot.loc[(spot['Marker 8 Intensity'] < 12) & (spot['Marker 8 Intensity'] > 8)]
        #modified_spot5 = spot5.loc[(spot5['Marker 8 Intensity'] > 12) & (spot5['Marker 8 Intensity'] < 8)]
        #print(spot['Marker 8 Positive'].value_counts())
        gen_tensors(i, original_imgs_path, train_path, test_path, spot)
