from PIL import Image, ImageDraw
import numpy as np
import os
import pandas as pd
from tifffile import TiffFile
import tensorflow as tf
#from preprocessing import center, box

def center(box):
    c_x = (box[2] + box[0])/2
    c_y = (box[3] + box[1])/2
    return float(c_x), float(c_y)

def box(center, h = 50):
    x1 = max( 0, int( center[0]- (h // 2) ) )
    y1 = max( 0, int( center[1]- (h // 2) ) )
    x2 = int( center[0] + (h // 2) )
    y2 = int( center[1] + (h // 2) )
    return x1, y1, x2, y2

def enforce_size(region):
    return box(center(region), h = 50)

def is_good_size(region, greater_than = 50, less_than = 50):
    return (region[2] - region[0] <= less_than) and (region[3] - region[1] >= greater_than)

# creates a (N X M X D) tensor from region r = (x1, y1, x2, y2)
def tensor( r, read_from_path = ''):
    if read_from_path:
        return np.load(read_from_path)

    if not is_good_size(r, greater_than = 25, less_than = 70):
        return 'bad dim'

    x1, y1, x2, y2 = enforce_size(r)
    if not is_good_size([x1, y1, x2, y2]):
        return 'border cell that cannot make 50x50 img'

    tensor = []
    for subdir, dirs, files in os.walk(original_imgs_path):
        for f in files:
            #print(f)
            with TiffFile(str(original_imgs_path) + f) as tif:
                matrix = tif[0].asarray().astype(np.float32)
                matrix = matrix/np.linalg.norm(matrix)
                tensor.append(matrix[ x1:x2, y1:y2  ])


    tensor = np.array(tensor)
    #print(tensor.shape)
    return tensor

def rotate(tensor):
    if type(tensor) == str:
        return 'bad dim'

    rotations = [tensor]
    for i in range(3):
        rotations.append(np.rot90( rotations[-1], axes = (1,2)))
    return rotations

def save(region, cell_id, label, output_dir):
    tensors = rotate(tensor(region))
    if type(tensors) == str:
        return 'bad dim'

    i = 0
    for t in tensors:
        #im = t[5]
        t.dump(output_dir + '%s_%s_%s.dat' %(cell_id, label, i))
        i += 1

def gen_tensors(input_directory, output_directory, metadata):
    i = 0
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

        #print(input_directory, output_directory)
        save( [xmin, ymin, xmax, ymax], cell_id, sox10_pos, output_directory)
        i += 1
        if i % 500 == 0:
            print(i)

def dataload(input_directory):
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            tf_tensor = ( tensor('', read_from_path = input_directory + f) )
            #print(tf_tensor.shape)
            tf_tensor = tf_tensor.transpose( )
            yield (tf_tensor, np.array( [ int(f.split('_')[1]) ] ).reshape(-1,) )

def testload(input_directory):
    tensors = []
    labels = []
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            tf_tensor = ( tensor('', read_from_path = input_directory + f) )
            #print(tf_tensor.shape)
            tensors.append( tf_tensor.transpose( ) )
            labels.append( np.array( [ int(f.split('_')[1]) ] ).reshape(-1,) )
            #yield (tf_tensor, np.array( [ int(f.split('_')[1]) ] ).reshape(-1,) )
    return np.array(tensors), np.array(labels)

def show(matrix):
    #print(matrix)
    #im = Image.fromarray(np.int8(matrix/255 ) )
    im = Image.fromarray(matrix)
    im.show()

if __name__ == '__main__':
    dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
    original_imgs_path = os.path.join(dir, 'original/')
    tensor_path = os.path.join(dir, 'tensors/')
    #tensor([200, 200, 250, 250])


    spot5 = pd.read_csv('spot5.csv')
    modified_spot5 = spot5.loc[(spot5['Marker 8 Intensity'] < 12) & (spot5['Marker 8 Intensity'] > 8)]

    gen_tensors(original_imgs_path, tensor_path, modified_spot5)

#     #test_path = os.path.join(dir, 'test_set/')
#     #print(testload(test_path))
#
#     i = 0
#     for l, label in dataload(tensor_path):
#         #print(l, l.shape, label)
#         #print(l.dtype)
#         show(l[4])
#         i += 1
#
#         if i == 3:
#             break
#         #if l.shape != (50, 50, 23):
#             #print(l.shape)
