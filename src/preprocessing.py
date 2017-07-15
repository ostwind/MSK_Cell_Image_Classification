''' In the data folder, intermediate files generated from this work flow are:
    ../original == colorized()  ==> ../colored == grid_crop() ==> ../cropped
    takes original GE greyscales and generates GUI ready images
'''

from PIL import ImageTk, Image
from tifffile import TiffFile, imsave, imread
import numpy as np
import os
from scipy import misc

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original = os.path.join(dir, 'original')
colored = os.path.join(dir, 'colored')
cropped = os.path.join(dir, 'cropped/')
print('Final output saved to %s' %(cropped) )

def colorized(file_paths = original, output_path = colored):
    ''' given a list of tif file paths, normalize underlying matrices wrt
        pixel intensity and experimental distribution of markers, then fuse
        multiple greyscale into a (8int, 8int, 8int) RGB additive format
    '''
    matrix_stack = []

    for path in file_paths:
        matrix_stack.append( np.array(Image.open(path)) )

    weights = [256-150, 256-150, 256]
    RGB = np.zeros((matrix_stack[0].shape[0], matrix_stack[0].shape[1], 3), "uint8")

    for i in range(len(matrix_stack)):
        RGB[:,:,i] = matrix_stack[i]/weights[i]

    img = Image.fromarray(RGB)
    img.resize((840,480), Image.ANTIALIAS)
    img.save( os.path.join(output_path, 'rgb_AA.png'), 'PNG')

filenames = ['MELANA_AFRemoved_pyr16_spot_004.tif',
'CD4_AFRemoved_pyr16_spot_004.tif',
'S001_mono_dapi_reg_pyr16_spot_004.tif']

file_paths = []
for f in filenames:
    file_paths.append( os.path.join(original, f) )

colorized( file_paths, output_path = colored )
print('colorized!')

def yield_coord(image, shift = [0, 0], stride = [400, 200], axis = 'x'):
    height, width = (4400, 7000)
    if axis =='x':
        for x in range(4000, width, stride[0]):
            yield x + shift[0] , x+ shift[0] +stride[0]
    else:
        for y in range(0, height, stride[1]):
            yield y + shift[1], y+ shift[1] +stride[1]

def grid_crop(input_directory = colored, output_directory = cropped):
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            image = Image.open(f)
            #print(f)
            if '.png' in f:
                for y1, y2 in yield_coord( f, axis = 'y'):
                    for x1, x2 in yield_coord( f):
                        small_img = image.crop((x1, y1, x2, y2))
                        save_name = '%s_%s_%s_%s_%s.png' %(f.split('_')[0], x1, x2, y1, y2)
                        small_img.save(output_directory + save_name)

grid_crop()
