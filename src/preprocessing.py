''' In the data folder, intermediate files generated from this work flow are:
    ../original == colorized()  ==> ../colored == grid_crop() ==> ../cropped
    takes original GE greyscales and generates GUI ready images
'''
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
#from tifffile import TiffFile, imsave, imread
import numpy as np
import os
#from scipy import misc
import pandas as pd

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
original = os.path.join(dir, 'original/')
colored = os.path.join(dir, 'colored/')
cropped = os.path.join(dir, 'cropped/')
#print('Final output saved to %s' %(colored) )

spot5 = pd.read_csv('spot5.csv')
#modified_spot5 = spot5.loc[
#(spot5['Marker 8 Intensity'] < 11.2) & (spot5['Marker 8 Intensity'] > 9.1)]

#(spot5['Marker 8 Intensity'] < 11.5) & (spot5['Marker 8 Intensity'] > 9)]

#print(modified_spot5.shape)

def center(box):
    c_x = (box[2] + box[0])/2
    c_y = (box[3] + box[1])/2
    return int(c_x), int(c_y)

def box(center, rect_h = 500):
    rect_w = rect_h * 1.3
    x1 = max( 0, int( center[0]- (rect_w // 2) ) )
    y1 = max( 0, int( center[1]- (rect_h // 2) ) )
    x2 = int( center[0] + (rect_w // 2) )
    y2 = int( center[1] + (rect_h // 2) )
    return x1, y1, x2, y2

def draw_box(output_directory = cropped, zoomed_output_directory = os.path.join(dir, 'zoomed/')):
    f = 'SOX10_AFRemoved_pyr16_spot_005.png'
    count = 0

    for i, row in modified_spot5.iterrows():
        im = Image.open(colored + f)
        draw = ImageDraw.Draw(im)
        xmin = modified_spot5.ix[i,'XMin']
        ymin = modified_spot5.ix[i,'YMin']
        xmax = modified_spot5.ix[i,'XMax']
        ymax = modified_spot5.ix[i,'YMax']
        cell_id = modified_spot5.ix[i, 'Object Id']
        cd4_intensity = modified_spot5.ix[i, 'Marker 2 Intensity']
        cd4_pos = modified_spot5.ix[i, 'Marker 2 Positive']
        sox10_pos = modified_spot5.ix[i, 'Marker 8 Positive']
        sox10_intensity = modified_spot5.ix[i, 'Marker 8 Intensity']
        if 1:
            count += 1
            draw.rectangle([xmin - 10, ymin - 10, xmax + 10, ymax + 10], outline = 'red')
            c = center([xmin, ymin, xmax, ymax])
            #x1, y1, x2, y2 = box(c )
            #print(xmin, ymin, xmax, ymax, '|',c, box(c))
            big_img = im.crop( ( box(c) ) )
            save_name = '%s_cd4_%.1f_sox10_%s|%.1f.png' %(
            cell_id, cd4_intensity, sox10_pos, sox10_intensity)
            big_img.save(output_directory + save_name, 'PNG')

            small_img = im.crop( ( box(c, rect_h = 250)  ) )
            small_img = small_img.resize((435,335), Image.ANTIALIAS)
            small_img.save(zoomed_output_directory + save_name, 'PNG')

            del draw

            if count % 1000 == 0:
                print( count/modified_spot5.shape[0] )
    #im.save('bounding_box.png')
    print(count)

draw_box()

def zoom(input_directory = os.path.join(dir, 'display/'),
    output_directory = os.path.join(dir, 'zoom/')):
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            img = Image.open(input_directory + f)
            img = img.transform((500, 200), Image.EXTENT, (125, 50, 250, 100 ))
            img.save(output_directory + f)

def colorized(file_paths, output_path = colored):
    ''' given a list of tif file paths, normalize underlying matrices wrt
        pixel intensity and experimental distribution of markers, then fuse
        multiple greyscale into a (8int, 8int, 8int) RGB additive format
    '''
    matrix_stack = []

    for path in file_paths:
        matrix_stack.append( np.array(Image.open(path)) )

    weights = [256-200, 256-175, 256+50]
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

''' \\ -x, -y
    (x1, y1) --------
        |           |
        |           |
        |           |
        |           |
        ---------(x2, y2)
                +x, +y \\
'''
def bounding_box(input_directory, output_directory, bounding_box_file):
    padding1 = 20
    padding2 = padding1 + 20

    #read bounding box
    for box in [[500, 0, 0, 500 ]]:#bounding_ iter:
        # (x1, y1, x2, y2) == box
        box1 = padding(box)
        print(box1)
        box2 = padding(box, padding_dist = 40)

    #write to zoom and normal directory
    #write {filename:label} to file


#bounding_box(colored, cropped, '.csv')

#combine three protein markers of the entire view
# file_paths = []
# for subdir, dirs, files in os.walk(original):
#     for f in files:
#         file_paths.append( os.path.join(original, f) )
#
# rgb_order = ['SOX10', 'CD4', 'S001']
# file_paths = [ f for i in range(3) for f in file_paths if rgb_order[i] in f ]
# colorized( file_paths, output_path = colored )
# print('colorized!')


# def yield_coord(image, shift = [0, 0], stride = [400, 200], axis = 'x'):
#     height, width = (4400, 7000)
#     if axis =='x':
#         for x in range(4000, width, stride[0]):
#             yield x + shift[0] , x+ shift[0] +stride[0]
#     else:
#         for y in range(0, height, stride[1]):
#             yield y + shift[1], y+ shift[1] +stride[1]
#
# def grid_crop(input_directory = original, output_directory = cropped):
#     for subdir, dirs, files in os.walk(input_directory):
#         for f in files:
#             image = Image.open(input_directory + f)
#             #print(f)
#             if '.tif' in f:
#                 for y1, y2 in yield_coord( f, axis = 'y'):
#                     for x1, x2 in yield_coord( f):
#                         small_img = image.crop((x1, y1, x2, y2))
#                         save_name = '%s_%s_%s_%s_%s_%s.tif' %(
#                         f.split('_')[-1][:3], f.split('_')[0], x1, x2, y1, y2)
#                         small_img.save(output_directory + save_name)
#grid_crop()
