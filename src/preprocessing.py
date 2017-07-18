''' In the data folder, intermediate files generated from this work flow are:
    ../original == colorized()  ==> ../colored == grid_crop() ==> ../cropped
    takes original GE greyscales and generates GUI ready images
'''

from PIL import ImageTk, Image, ImageEnhance, ImageFilter
from tifffile import TiffFile, imsave, imread
import numpy as np
import os
from scipy import misc

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original = os.path.join(dir, 'original/')
colored = os.path.join(dir, 'colored/')
cropped = os.path.join(dir, 'cropped/')
print('Final output saved to %s' %(colored) )

def zoom(input_directory = os.path.join(dir, 'display/'),
    output_directory = os.path.join(dir, 'zoom/')):
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            img = Image.open(input_directory + f)
            img = img.transform((500, 200), Image.EXTENT, (125, 50, 250, 100 ))
            img.save(output_directory + f)
zoom()

def colorized(file_paths, output_path = colored):
    ''' given a list of tif file paths, normalize underlying matrices wrt
        pixel intensity and experimental distribution of markers, then fuse
        multiple greyscale into a (8int, 8int, 8int) RGB additive format
    '''
    matrix_stack = []

    for path in file_paths:
        print(path)
        matrix_stack.append( np.array(Image.open(path)) )

    weights = [256-150, 256-150, 256]
    RGB = np.zeros((matrix_stack[0].shape[0], matrix_stack[0].shape[1], 3), "uint8")

    for i in range(len(matrix_stack)):
        RGB[:,:,i] = matrix_stack[i]/weights[i]

    img = Image.fromarray(RGB)

    img.resize((7082,4424), Image.ANTIALIAS)
    img_name = file_paths[0].split('.')[0].split('/')[-1]
    img.save( os.path.join(output_path, img_name +'.png'), 'PNG' )

def yield_rgb_triplet(input_directory = cropped):
    all_file_names = []
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            all_file_names.append(f)

    while all_file_names:
        key = '_'.join( all_file_names[0].split('_')[2:] )
        triplet = [ f for f in all_file_names if key in f ]
        rgb_order = ['SOX10', 'CD4', 'S001']
        rgb_triplet = [ f for i in range(3) for f in triplet if rgb_order[i] in f ]

        all_file_names = [f for f in all_file_names if f not in rgb_triplet]
        yield rgb_triplet

# for t in yield_rgb_triplet():
#     colorized( [cropped + t[0], cropped + t[1], cropped + t[2]] )

def yield_coord(image, shift = [0, 0], stride = [400, 200], axis = 'x'):
    height, width = (4400, 7000)
    if axis =='x':
        for x in range(4000, width, stride[0]):
            yield x + shift[0] , x+ shift[0] +stride[0]
    else:
        for y in range(0, height, stride[1]):
            yield y + shift[1], y+ shift[1] +stride[1]

def grid_crop(input_directory = original, output_directory = cropped):
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            image = Image.open(input_directory + f)
            #print(f)
            if '.tif' in f:
                for y1, y2 in yield_coord( f, axis = 'y'):
                    for x1, x2 in yield_coord( f):
                        small_img = image.crop((x1, y1, x2, y2))
                        save_name = '%s_%s_%s_%s_%s_%s.tif' %(
                        f.split('_')[-1][:3], f.split('_')[0], x1, x2, y1, y2)
                        small_img.save(output_directory + save_name)
#grid_crop()
#
# file_paths = []
# for subdir, dirs, files in os.walk(original):
#     for f in files:
#         file_paths.append( os.path.join(original, f) )
#
# colorized( file_paths, output_path = colored )
# print('colorized!')
