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
import random
import glob
from random import shuffle
from multiprocessing.dummy import Pool
import itertools
import tensorflow as tf

# needed to make multiprocess work on Ubuntu OS
# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
os.system("taskset -p 0xff %d" % os.getpid())

img_length = 50

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original_imgs_path = os.path.join(dir, 'original/') # cut first array from /real_original/
labeled = os.path.join(dir, 'labeled/')
unlabeled = os.path.join(dir, 'unlabeled/')
test_set = os.path.join(dir, 'test_set/')
tif_files = glob.glob(original_imgs_path + '*')

def tensor(spot, r):
    spot_specific_tif_files = [ f for f in tif_files if spot ==  int(f.split('_')[-1].split('.')[0] )]

    # r is a 4-tuple (x1, y1, x2, y2)
    # check if img is out of bounds
    x1, y1, x2, y2 = r
    tensor = []
    for f in spot_specific_tif_files:
        # extract spot integer from filenames e.g. -filename-_spot_00X.tif
        im = Image.open(f)
        cell_im = im.crop( (x1, y1, x2, y2) )
        cell_im = cell_im.convert(mode = 'F')
        cell_im = cell_im.resize( (img_length, img_length), Image.ANTIALIAS )
        tensor.append( np.array(cell_im) )

    tensor = np.array(tensor).transpose()
    #computes (x - mean) / adjusted_stddev
    #tensor = tf.image.per_image_standardization(tensor)
    tensor = (tensor - np.mean(tensor)) / np.std(tensor)
    # if not convert to uint8, file sizes are huge
    tensor = np.array(tensor).astype(np.uint8)
    return tensor

def rotate(tensor): # dont rotate if there are enough samples
    rotations = [tensor]
    # each increment of cell_orientation corresponds to 90 deg clockwise turn
    # of spatial dimensions
    for cell_orientation in range(3):
        rotations.append(np.rot90( rotations[-1], axes = (0,1)))
    return rotations

def save(region, spot, cell_id, label, output_dir ):
    original_cell = tensor(spot, region)
    # cell box might be out of bounds
    if len(original_cell) == 1:
        return

    tensors = rotate(original_cell)
    cell_orientation = 0
    for t in tensors:
        #print(type(t), t.shape)
        assert t.shape == (img_length, img_length, 32), 'incorrect shape: %s' %(t.shape)

        filename = '%s_%s_%s.bin' %(cell_id, label, cell_orientation)
        # label removed, dim restored in sample_feed.read_my_data
        rotated_tensor = np.insert( t, 0, int(label) )
        rotated_tensor = rotated_tensor.flatten()
        rotated_tensor.tofile(output_dir + filename)
        cell_orientation += 1

def empty_dir(path):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)

class dataset():
    def __init__(self, csv_path, spot):
        self.df = pd.read_csv(csv_path)
        self.image_directory = original_imgs_path

        self.unlabeled_directory = unlabeled
        self.labeled_directory = labeled
        self.test_directory = test_set

        self._filter_by_spot( [spot])#[3, 4, 5, 6, 7, 8, 9, 10, 11] )
        self._filter_by_dim_ratio()

        self._label()
        self._train_test_split()
        self.write_matrix()

    def _filter_by_spot(self, spots):
        spots_to_strs = ['Spot' + str(s) for s in spots]
        # regex '|' to try to match each of the substrings in the words in df['Image Location']
        self.df = self.df[ self.df['Image Location'].str.contains( '|'.join(spots_to_strs)  ) ]

    def _filter_by_dim_ratio(self, ratio_threshold = 0.5):
        # eliminate cells too rectangular
        self.df['length'], self.df['width'] = (self.df.YMax - self.df.YMin,
                                         self.df.XMax - self.df.XMin)

        # filter cells too large in either dimension
        self.df['good_len'] = self.df.length < img_length
        self.df['good_width'] = self.df.width < img_length
        self.df = self.df[ (self.df.good_len == True) & (self.df.good_width == True) ]

        self.df.ratio1_good = ( self.df.length / self.df.width > ratio_threshold )
        self.df.ratio2_good = ( self.df.width / self.df.length > ratio_threshold )
        self.df = self.df[ (self.df.ratio1_good == True) & (self.df.ratio2_good == True) ]

    def _label(self):
        ''' binary label: CD3 1/0 | CD4 1/0 | CD8 1/0 | CD20 1/0
        '''
        self.df['label'] = self.df['Dye 2 Positive'].astype(str)
        self.df['label'] += (self.df['Dye 3 Positive'].astype(str) +
                         self.df['Dye 4 Positive'].astype(str) +
                         self.df['Dye 6 Positive'].astype(str) )

        def _binary_to_dec(binary_string):
            valid_labels = {'0000':0, '0001':1, '1010':2, '1100':3, '1110':4, }
            if binary_string in valid_labels.keys():
                return valid_labels[binary_string]
            return 5

        self.df['label_dec'] = self.df.label.apply(_binary_to_dec )
        self.df = self.df[ self.df['label_dec'] != 5]

    def _train_test_split(self):
        self.df['proba'] = np.random.uniform(0, 1, self.df.shape[0])

        #self.df['bad_unlabeled'] = self.df['label_dec'] == 5
        #self.df['labeled'] = self.df['label_dec'] != 5

        #70/30 train/test split, of the labeled data
        #self.df['test_set'] =  (self.df.proba < 0.3) & self.df['labeled']
        #self.df['train_set'] =  ~self.df['test_set'] & self.df['labeled']
        #self.df = self.df[ self.df.test_set | self.df.train_set  ]

        # 30/10/60 train/test/unlabeled split
        self.df['train_set'] =  (self.df.proba < 0.3) #& self.df['labeled']
        self.df['unlabeled'] = (self.df.proba > 0.4) #& self.df['labeled']
        self.df['test_set'] =  ~self.df['train_set'] & ~self.df['unlabeled'] #& self.df['labeled']
        #self.df = self.df[ self.df.test_set | self.df.train_set  ]

    def _write(self, row):
        path_to_write = self.unlabeled_directory
        if row['test_set']:
            path_to_write = self.test_directory
        if row['train_set']:
            path_to_write = self.labeled_directory

        save( [row['XMin'], row['YMin'], row['XMax'], row['YMax'] ],
        int(row['spot']), row['spot'] + str(row['Object Id']),
        row['label_dec'], path_to_write)
        return ' written'

    def write_matrix(self):
        self.df['spot'] = self.df['Image Location'].str.split('_').str[-1]
        self.df['spot'] = self.df['spot'].str.extract('([0-9]+)', expand = False)
        self.df['written'] = self.df.apply(lambda row: self._write(row), axis = 1)

if __name__ == '__main__':
    empty_dir(labeled)
    empty_dir(test_set)
    empty_dir(unlabeled)
    # generate spotX.csv from /data/cell_metadata.csv

    path = '../data/cell_metadata.csv'

    np.random.seed(42)

    spots = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    def f(s):
        dataset(path, spot = s)

    pool = Pool()
    results = pool.map( f, spots)
    pool.close()
    pool.join()
