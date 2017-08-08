''' contains sample data generation,
    - tensor <= rotate <= save <= gen_tensors

    data augmentation,
    - rotate by X deg

    and data feed mechanisms for feed_forward.py
'''
from PIL import Image, ImageDraw
import numpy as np
import os
import pandas as pd
from tifffile import TiffFile
import tensorflow as tf
import random
import glob
from random import shuffle
from multiprocessing.dummy import Pool
import itertools

img_length = 30

def tensor(spot, r):
    x1, y1, x2, y2 = r

    tensor = []
    for subdir, dirs, files in os.walk(original_imgs_path):
        files.sort()
        for f in files:
            # extract spot integer from filenames e.g. -filename-_spot_00X.tif
            if spot ==  int(f.split('_')[-1].split('.')[0] ):
                im = Image.open(original_imgs_path + f)
                cell_im = im.crop( (x1, y1, x2, y2) )
                cell_im = cell_im.convert(mode = 'F')
                cell_im = cell_im.resize((img_length, img_length), Image.ANTIALIAS)
                # approx normalization to ~0-10 range
                cell_matrix = (np.array(cell_im))/255
                tensor.append(cell_matrix)
    tensor = np.array(tensor).astype(np.uint8).transpose()
    #tensor = normalize(tensor)
    return tensor

def rotate(tensor): # dont rotate if there are enough samples
    rotations = [tensor]
    # each increment of cell_orientation corresponds to 90 deg clockwise turn
    # of spatial dimensions
    for cell_orientation in range(3):
        rotations.append(np.rot90( rotations[-1], axes = (0,1)))
    return rotations

def save(region, spot, cell_id, label, output_dir ):
    tensors = rotate(tensor(spot, region))

    cell_orientation = 0
    for t in tensors:
        assert t.shape == (img_length, img_length, 32), 'incorrect shape: %s' %(t.shape)

        filename = '%s_%s_%s.bin' %(cell_id, label, cell_orientation)
        rotated_tensor = np.insert( t, 0, int(label) )
        rotated_tensor = rotated_tensor.flatten()
        rotated_tensor.tofile(output_dir + filename)
        cell_orientation += 1

def _good_dim_ratio(region, min_ratio = 0.5):
    xmin, ymin, xmax, ymax = region
    length, width = ymax - ymin, xmax - xmin
    if length/width < min_ratio or width/length < min_ratio:
        return False
    return True

def gen_tensors(spot, metadata):#input_directory, output_directory, test_directory, metadata):
    input_directory = original_imgs_path
    output_directory = train_path
    test_directory = test_path

    i = 0

    s = np.random.uniform(0, 1, metadata.shape[0])
    #print(metadata.shape)#input_directory, output_directory)
    for index, row in metadata.iterrows():
        xmin = metadata.ix[index,'XMin']
        ymin = metadata.ix[index,'YMin']
        xmax = metadata.ix[index,'XMax']
        ymax = metadata.ix[index,'YMax']
        # check dimension ratios are not too rectangular, but square
        if not _good_dim_ratio( [xmin, ymin, xmax, ymax] ):
            continue

        cell_id = metadata.ix[index, 'Object Id']
        cd4 = metadata.ix[index, 'Dye 3 Positive']
        cd8 = metadata.ix[index, 'Dye 4 Positive']
        cd20 = metadata.ix[index, 'Dye 6 Positive']

        path_to_write = output_directory
        if s[i] < 0.3: # train/test split at 70/30
            path_to_write = test_directory
        # convert from binary indicator vec to dec for tensorflow class input
        label = str(cd4)+ str(cd8) + str(cd20)
        label = int(label, 2)

        save( [xmin, ymin, xmax, ymax], int(spot),
        str(spot) + str(cell_id), label, path_to_write)

        i += 1
        if i % 2000 == 0:
            print('files done ', i, 'spot: ', spot)

''' SAMPLE QUEUE GENERATION AND DEQUEUE FOR TF model.py
    read_my_data <= inputs (form queues of ImageRecord objects)
'''

def read_my_data(filename_queue):

    class ImageRecord(object):
        def __init__(self):
            # Dimensions of the images in the dataset.
            self.height = 30
            self.width = 30
            self.depth = 32

    result = ImageRecord()
    label_bytes = 1
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    assert record_bytes == (30*30*32)+1

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

def empty_dir(path):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original_imgs_path = os.path.join(dir, 'original/') # cut first array from /real_original/
train_path = os.path.join(dir, 'tensors/')
test_path = os.path.join(dir, 'test_set/')

if __name__ == '__main__':
    #empty_dir(train_path)
    #empty_dir(test_path)
    # generate spotX.csv from /data/cell_metadata.csv
    df = pd.read_csv('../data/cell_metadata.csv')

    # # creating intermediate dataframes to be read later
    # for i in range(3, 12):
    #     spot = df[ df['Image Location'].str.contains('Spot%s' %(i))]
    #     spot.to_csv('spot%s.csv' %(i))

    spots = [5, 6, 7, 8, 9, 10, 11]
    metadata_list = []
    for i in range(5, 12):
         metadata_list.append(df[ df['Image Location'].str.contains('Spot%s' %(i))] )

    pool = Pool()
    results = pool.starmap( gen_tensors, zip(spots, metadata_list))
    pool.close()
    pool.join()

        #gen_tensors(i, original_imgs_path, train_path, test_path, spot_metadata)

# TODO: test normalizing
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

# def center(box):
#     c_x = (box[2] + box[0])/2
#     c_y = (box[3] + box[1])/2
#     return float(c_x), float(c_y)
#
# def box(center, h = img_length):
#     x1 = int( center[0]- (h // 2) )
#     y1 = int( center[1]- (h // 2) )
#     x2 = int( center[0] + (h // 2) )
#     y2 = int( center[1] + (h // 2) )
#     return x1, y1, x2, y2
#
# # crop this size w/ no rescaling
# def enforce_size(region):
#     return box(center(region), h = img_length)

# def is_good_size(region, greater_than = img_length, less_than = img_length):
#     x1, y1, x2, y2 = region
#     if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
#         return False
#     return (x2 - x1 <= less_than) and (x2 - x1 >= greater_than) and (y2 - y1 <= less_than) and (y2 - y1 >= greater_than)
