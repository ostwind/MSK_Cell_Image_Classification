''' In the data folder, intermediate files generated from this work flow are:
    ../real_original == take_first_image() ==> ../original
    == colorized( user input )  ==> ../colored == draw_box(meta_data) ==>
    ../cropped, ../zoomed, ../zoomed_nobox
    takes original GE greyscales and generates GUI ready images
'''
from PIL import Image, ImageDraw
import numpy as np
import os
import pandas as pd
from tifffile import TiffFile
import glob
import pickle

def take_first_image(input_directory, output_directory):
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            if '.tif' in f : #and '005' in f
                with TiffFile(str(input_directory) + f) as tif:
                    matrix = tif[0].asarray()
                    im = Image.fromarray(matrix)
                    im.save(output_directory + f)
                    #print(output_directory, f)
                    im.close()

def colorized(file_paths, output_path, prepend = ''):
    ''' given a list of tif file paths, normalize underlying matrices wrt
        pixel intensity and experimental distribution of markers, then fuse
        multiple greyscale into a (8int, 8int, 8int) RGB additive format
    '''
    matrix_stack = []
    img_name = prepend
    #print(file_paths)
    for path in file_paths:
        matrix_stack.append( np.array(Image.open(path)) )
        img_name += path.split('/')[-1].split('_')[0] + '_'

    #256
    weights = [150, 200, 150]
    RGB = np.zeros((matrix_stack[0].shape[0], matrix_stack[0].shape[1], 3), "uint8")

    for i in range(len(matrix_stack)):
        RGB[:,:,i] = matrix_stack[i]/weights[i]

    img = Image.fromarray(RGB)
    img.resize((7082,4424), Image.ANTIALIAS)

    #marker_name = file_paths[0].split('/')[-1].split('_')[0]
    img.save( os.path.join(output_path, img_name +'.png'), 'PNG' )
    return img_name + '.png'

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

def draw_box(
input_directory, #= colored + 'SOX10_AFRemoved_pyr16_spot_005.png',
output_directory, zoomed_output_directory, total_metadata
):
        count = 0
        for i, row in metadata.iterrows():
            im = Image.open(input_directory + spot_image)
            draw = ImageDraw.Draw(im)
            xmin = metadata.ix[i,'XMin']
            ymin = metadata.ix[i,'YMin']
            xmax = metadata.ix[i,'XMax']
            ymax = metadata.ix[i,'YMax']
            cell_id = metadata.ix[i, 'Object Id']
            cd4_intensity = metadata.ix[i, 'Marker 2 Intensity']
            cd4_pos = metadata.ix[i, 'Marker 2 Positive']
            sox10_pos = metadata.ix[i, 'Marker 8 Positive']
            sox10_intensity = metadata.ix[i, 'Marker 8 Intensity']
            if 1:
                count += 1
                c = center([xmin, ymin, xmax, ymax])
                save_name = '%s_cd4_%.1f_sox10_%.1f.png' %(
                cell_id, cd4_pos, sox10_pos)

                draw.rectangle([xmin - 10, ymin - 10, xmax + 10, ymax + 10], outline = 'red')

                big_img = im.crop( ( box(c) ) )
                big_img.save(output_directory + '/' + save_name, 'PNG')

                small_img = im.crop( ( box(c, rect_h = 250)  ) )
                small_img = small_img.resize((435,335), Image.ANTIALIAS)
                small_img.save(zoomed_output_directory +'/'+ save_name, 'PNG')

                del draw

                if count % 100 == 0:
                    print( 'percentage of samples generated: ', count/metadata.shape[0] )

def _channel_to_filepath(input_directory, channel_list, spot):
    file_paths = glob.glob(input_directory + '*')
    file_paths = [x for x in file_paths if spot in x]
    channel_names = [ x.split('_')[0].split('/')[-1] for x in file_paths ]
    channel_filepath_map = dict(zip(channel_names, file_paths ))

    rgb_paths = [channel_filepath_map[x] for x in channel_list if
    x in channel_names]

    assert len(rgb_paths) == 3, '%s, %s, %s is not len 3' %(channel_list, spot, repr(rgb_paths))
    return rgb_paths

def gen_samples(dir_lookup, total_metadata, rgb_orders = [['SOX10', 'CD4', 'S001']]):
    #take_first_image( input_directory = dir_lookup['real_original'], output_directory = dir_lookup['original'] )
    #return
    #extract relevant spots from metadata df
    total_metadata['spot'] = (total_metadata['Image Location'].str.split('_') )
    s = total_metadata.apply(lambda x: pd.Series(x['spot']).values[1][4:], axis=1 )#.stack().reset_index(level = 1, drop =  True)
    s.name = 'spot_int'
    total_metadata = total_metadata.join(s)
    subdataframes_by_spot = total_metadata.groupby(['spot_int'])

    for subdataframe in subdataframes_by_spot:
        #groupby element: subdataframe is (int_type: spot number, df_type: dataframe  )
        cur_spot = '00' + str(subdataframe[0])


        #print(type(subdataframe['spot_int']), type(subdataframe['spot_int'][0]) )
        for rgb_order in rgb_orders:
            marker_subset =  _channel_to_filepath(dir_lookup['original'], rgb_order, cur_spot  )
            #marker_subset = [ f for i in range(3) for f in file_paths if order[i] in f and cur_spot in f ]
            colored_file_name = colorized( marker_subset, output_path = dir_lookup['colored'], prepend = cur_spot)
            print('%s colorized' %(colored_file_name))

        #     draw_box(input_directory = dir_lookup['colored'] + colored_file_name,
        #     output_directory = dir_lookup['cropped'] + '_'.join(order),
        #     zoomed_output_directory = dir_lookup['zoomed'] + '_'.join(order), total_metadata = metadata)

        #i += 1

def make_dir_dictionary(rgb_orders, real_original_loc):
    dir_lookup =  { 'real_original':str(real_original_loc), 'original':'', 'colored':'',
    'cropped' : '', 'zoomed': ''  }
#    dir = os.path.normpath(os.getcwd() + os.sep + os.pardir +'/data')
    dir = os.path.dirname(__file__)
    dir = os.path.join(dir, '../data/')
    #cropped = os.path.join(self.dir, '../../data/cropped/')

    if not os.path.exists(dir):
        os.makedirs(dir)

    for dir_name in dir_lookup.keys():
        if dir_name == 'real_original':
            continue

        new_path = os.path.join(dir, dir_name + '/')
        if not os.path.exists(new_path):
            os.makedirs(new_path, exist_ok = True)
            print('%s not found, creating it' %(new_path))

        if dir_name == 'cropped' or dir_name == 'zoomed':
            for order in rgb_orders:
                os.makedirs(new_path+ '/' + '_'.join(order), exist_ok = True )

        dir_lookup[dir_name] = new_path

    return dir_lookup

if __name__ == '__main__':

    rgb_orders = [['PD1', 'CD3', 'CD14'], [ 'CD8', 'CD3', 'CD15'], ['CD8', 'CD3','CD4']]
    dir_lookup = make_dir_dictionary(rgb_orders, '/home/lihan/Documents/image/data/real_original/')

    df = pd.read_csv('../data/cell_metadata.csv')
    #bad_cases = pickle.load( open('mislabeled.p', 'rb'))
    #bad_cases = [int(x) for x in bad_cases]
    #bad_df = df[ df['Object Id'].isin(bad_cases) ]

    #spot5 = pd.read_csv('spot5.csv')
    #modified_spot5 = spot5.loc[(spot5['Marker 8 Intensity'] < 11.8) & (spot5['Marker 8 Intensity'] > 9.1)]
    #intensity range leads to 892 label 1 samples, 880 label 0 samples

    gen_samples(dir_lookup, rgb_orders = rgb_orders, total_metadata = df)#'SOX10_AFRemoved_pyr16_spot_005.png')

''' \\ -x, -y
    (x1, y1) --------
        |           |
        |           |
        |           |
        |           |
        ---------(x2, y2)
                +x, +y \\
'''
