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

img_length = 40

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original_imgs_path = os.path.join(dir, 'original/') # cut first array from /real_original/
train_path = os.path.join(dir, 'tensors/')
test_path = os.path.join(dir, 'test_set/')

tif_files = glob.glob(original_imgs_path + '*')

def tensor(spot, r):
    x1, y1, x2, y2 = r
    tensor = []
    for f in tif_files:
        # extract spot integer from filenames e.g. -filename-_spot_00X.tif
        if spot ==  int(f.split('_')[-1].split('.')[0] ):
            im = Image.open(f)
            cell_im = im.crop( (x1, y1, x2, y2) )
            cell_im = cell_im.convert(mode = 'F')
            cell_im = cell_im.resize((img_length, img_length), Image.ANTIALIAS)
            # approx normalization to ~0-10 range
            cell_matrix = (np.array(cell_im))/255
            tensor.append(cell_matrix)
    tensor = np.array(tensor).astype(np.uint8).transpose()

    #tensor = normalize(tensor)
    #print(tensor[:,:,0], tensor.shape)
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
    def __init__(self, df, spot):
        self.df = df#pd.read_csv(csv_path)
        self.image_directory = original_imgs_path
        self.train_directory = train_path
        self.test_directory = test_path

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
        self.df.length, self.df.width = (self.df.YMax - self.df.YMin,
                                         self.df.XMax - self.df.XMin)
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

        self.df['label_binary'] = self.df.label.apply(_binary_to_dec )

    def _train_test_split(self):
        self.df['proba'] = np.random.uniform(0, 1, self.df.shape[0])
        self.df['test_set'] = self.df.proba < 0.1

        # training sample cannot be test nor be biologically impossible
        # but ambiguous/impossible samples may appear in test w/ label 5
        self.df['train_set'] =  ~self.df['test_set'] & (self.df['label_binary'] != 5)
        self.df = self.df[ self.df.test_set | self.df.train_set  ]

    def _write(self, row):
        path_to_write = self.train_directory
        if row['test_set']:
            path_to_write = self.test_directory

        save( [row['XMin'], row['YMin'], row['XMax'], row['YMax'] ],
        int(row['spot']), row['spot'] + str(row['Object Id']),
        row['label_binary'], path_to_write)
        return ' written'

    def write_matrix(self):
        self.df['spot'] = self.df['Image Location'].str.split('_').str[-1]
        self.df['spot'] = self.df['spot'].str.extract('([0-9]+)', expand = False)
        self.df['written'] = self.df.apply(lambda row: self._write(row), axis = 1)

if __name__ == '__main__':
    empty_dir(train_path)
    empty_dir(test_path)
    # generate spotX.csv from /data/cell_metadata.csv
    df = pd.read_csv('../data/cell_metadata.csv')
    np.random.seed(42)

    spots = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    def f(s):
        dataset(df, spot = s)

    pool = Pool()
    results = pool.map( f, spots)
    pool.close()
    pool.join()

    # for spot in [6, 8,9,10,11]:
    #     spot_metadata = df[ df['Image Location'].str.contains('Spot%s' %(spot))]
    #     gen_tensors(spot, spot_metadata)

    # # creating intermediate dataframes to be read later
    # for i in range(3, 12):
    #     spot = df[ df['Image Location'].str.contains('Spot%s' %(i))]
    #     spot.to_csv('spot%s.csv' %(i))


    #multi-processing option
    # spots = [6,8,9,10,11] #[3, 4, 5, 6, 7, 8, 9, 10, 11]
    # metadata_list = []
    # for i in spots:
    #      metadata_list.append(df[ df['Image Location'].str.contains('Spot%s' %(i))] )
    #
    # pool = Pool()
    # results = pool.starmap( gen_tensors, zip(spots, metadata_list))
    # pool.close()
    # pool.join()

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(32):
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
