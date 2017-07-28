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
        #print(t[:,:,17].shape, t[:,:,17].dtype)
#        mo = Image.fromarray( (t[:,:,10] ))
#        mo.save(output_dir + '%s_%s_%s.png' %(cell_id, label, i), 'PNG')
        t.dump(output_dir + '%s_%s_%s.dat' %(cell_id, label, i))
        i += 1

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
            print(i, spot)

def update_cache(cache, sample_num, batch_size):
    next_batch = cache[-batch_size:]
    del cache[-batch_size:]
    tensor_batch, label_batch, file_name_batch = zip(*next_batch)
    label_batch = np.concatenate(label_batch, axis = 0)
    return tensor_batch, label_batch, file_name_batch

def dataload(train):
    input_directory = test_path
    if train:
        input_directory = train_path

    cache = []
    batch_size = 500
    sample_num = 0
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            if '.dat' in f:
                tf_tensor = ( load(read_from_path = input_directory + f) )
                cache.append( (tf_tensor, np.array( [ int(f.split('_')[1]) ]), f.split('_')[0] ) )
                sample_num += 1
                if sample_num == 5000:
                    random.shuffle(cache)
                    while sample_num != 0:
                        yield update_cache(cache, sample_num, batch_size)
                        sample_num -= batch_size

    random.shuffle(cache)
    while sample_num > batch_size: # yield remainder
        yield update_cache(cache,sample_num, batch_size)
        sample_num -= batch_size

def empty_dir(path):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original_imgs_path = os.path.join(dir, 'original/') # cut first array from /real_original/
train_path = os.path.join(dir, 'tensors/')
test_path = os.path.join(dir, 'test_set/')

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
