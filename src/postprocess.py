''' cell_id | class probabilities | mislabeled     || HALO dataframe        || rgb order  || original imgs
        |               |               |                   |                       |            |
        integrated dataframe w/ coordinates and probabilities                       colored images
                                                |                                           |
                                                captioned cell profiles w/ class probabilities
'''

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pandas as pd
from tifffile import TiffFile
import random
import glob
import pickle

def _retriev_recent_tf_log(subdir = 'tensor_logs'):
    path = './tf_logs/'
    newest_tf_run = (os.listdir( path ))#, key = os.path.getctime )
    newest_tf_run.sort()
    return path + newest_tf_run[-1] + '/' + subdir + '/'

def collect_tensor_files(recent_run_dir):
    all_tensors_paths = glob.glob(recent_run_dir + '*')
    collected_tensor = []
    for path in all_tensors_paths:
        if '.npy' in path:
            collected_tensor.append( np.load(path) )

    collected_tensor = np.concatenate(collected_tensor, axis = 1)
    collected_tensor = collected_tensor.astype(str)

    df = pd.DataFrame( collected_tensor.T,
    columns = ['path', 'class', 'mislabeled', '0000', '0001', '1010', '1100', '1110', 'others']  )
    df.to_csv(recent_run_dir + 'collected_tensor_data.csv', index = False)

def merge_main_output_dfs(recent_run_dir, main_df_path = '../data/cell_metadata.csv'):
    output_df = pd.read_csv(recent_run_dir + 'collected_tensor_data.csv',
    )
    output_df['Object Id'] = output_df['path'].str.split('/').str[-1].str.split('_').str[0]

    output_df['Object Id'].loc[output_df['Object Id'].str[0] != '1'] = output_df['Object Id'].str[0] + '-' + output_df['Object Id'].str[1:]
    output_df['Object Id'].loc[output_df['Object Id'].str[0] == '1'] = output_df['Object Id'].str[:2] + '-' + output_df['Object Id'].str[2:]

    def deci_to_binary(deci):
        binary_labels = {0: '0000', 1: '0001', 2: '1010', 3: '1100', 4: '1110', 5: 'other' }
        return binary_labels[deci]
    output_df['class'] = output_df.apply(lambda row: deci_to_binary(row['class']), axis = 1)

    main_df = pd.read_csv(main_df_path)
    main_df['spot'] = main_df['Image Location'].str.split('_').str[-1].str.split('.').str[0].str[4:]
    main_df['Object Id'] = main_df['spot'] + '-' + main_df['Object Id'].astype(str)

    main_df = main_df[ main_df['Object Id'].isin( output_df['Object Id'] ) ]
    main_df = pd.merge( output_df, main_df, on ='Object Id' )

    main_df['pred'] = main_df[['0000', '0001', '1010', '1100', '1110', 'others']].idxmax(axis =1)
    main_df['spot'] = main_df['spot'].map('{:3}'.format)#astype('str', errors = 'coerce')

    def _add_zeros(string_num):
        while  len(string_num) < 3:
            string_num = '0' + string_num
        return string_num
    main_df['spot'] = main_df.apply(lambda row: _add_zeros(row['spot']), axis = 1)

    main_df.to_csv( recent_run_dir + 'collected_tensor_data.csv', index = False )
    #print('main_df before merging: ', main_df.shape, 'output_df shape: ', output_df.shape)

def take_first_image(input_directory, output_directory):
    for subdir, dirs, files in os.walk(input_directory):
        for f in files:
            if '.tif' in f and '012' not in f:
                with TiffFile(str(input_directory) + f) as tif:
                    #print(str(input_directory) + f)
                    matrix = tif[0].asarray()
                    im = Image.fromarray(matrix)
                    im.save(output_directory + f)
                    #print(output_directory, f)
                    im.close()

def colorized(file_paths, output_path, prepend_spot = ''):
    ''' given a list of tif file paths, normalize underlying matrices wrt
        pixel intensity and experimental distribution of markers, then fuse
        multiple greyscale into a (8int, 8int, 8int) RGB additive format
    '''
    matrix_stack = []
    img_name = prepend_spot

    assert len(file_paths) == 3, print(file_paths)

    for path in file_paths:
        matrix_stack.append( np.array(Image.open(path)) )
        img_name += path.split('/')[-1].split('_')[0] + '_'
    #256
    weights = [300, 300, 256]
    RGB = np.zeros((matrix_stack[0].shape[0], matrix_stack[0].shape[1], 3), "uint8")

    for i in range(len(matrix_stack)):
        RGB[:,:,i] = matrix_stack[i]/weights[i]

    img = Image.fromarray(RGB)
    #img.resize((7082,4424), Image.ANTIALIAS)

    #marker_name = file_paths[0].split('/')[-1].split('_')[0]
    img.save( os.path.join(output_path, img_name +'.png'), 'PNG' )
    return img_name + '.png'

def color_originals(input_directory, spots, rgb_orders):
    greyscale_paths = glob.glob(input_directory + '*')
    greyscale_file_names = [p.split('/')[-1] for p in greyscale_paths]
    spot_and_channel = [ (f.split('_')[-1].split('.')[0] , f.split('_')[0]) for f in greyscale_file_names  ]

    # (spot value, channel name) => path of file w/ spot and channel name
    select_path = dict(zip(spot_and_channel, greyscale_paths))
    for s in spots:
        for rgb in rgb_orders:
            r, g, b = rgb
            colorized(  [select_path[s, r], select_path[s, g], select_path[s, b]] , colored, prepend_spot = str(s))


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

font = ImageFont.truetype("/home/lihan/fonts/Inconsolata/Inconsolata Bold for Powerline.ttf", 16)

def make_cell_profile( row):
    save_name = '%s_%s_%s_' %(row['Object Id'], row['class'], row['pred']  )
    colored_combo_paths = glob.glob(colored + '*')
    relevant_color_paths = [p for p in colored_combo_paths if '0'+str(row['spot']) in p ]
    assert len(relevant_color_paths) == 3, print(relevant_color_paths)

    f_names = [p.split('/')[-1] for p in relevant_color_paths]
    channels_in_combo = [f.split('.')[0][3:-1] for f in f_names]

    xmin, ymin, xmax, ymax = row['XMin'], row['YMin'], row['XMax'], row['YMax']

    for colored_path, channel_combo in zip(relevant_color_paths, channels_in_combo):
        save_name_color_variation = save_name + channel_combo + '.png'

        im = Image.open(colored_path)
        draw = ImageDraw.Draw(im)
        draw.rectangle([xmin - 10, ymin - 10, xmax + 10, ymax + 10], outline = 'red')

        c = center( (xmin, ymin, xmax, ymax) )
        captioned_img = im.crop(box(c, rect_h = 250) )
        captioned_img = captioned_img.resize((435,335), Image.ANTIALIAS)
        draw2 = ImageDraw.Draw(captioned_img)

        draw2.text((0,0), " 0000 / %.2f | 0001 / %.2f | 1010 / %.2f "
        %(row['0000'], row['0001'], row['1010']), font = font)
        draw2.text((0,20), " 1100 / %.2f | 1110 / %.2f | others / %.2f"
        %(row['1100'], row['1110'], row['others']), font = font)

        draw2.text((0, 260), "spot-cell id / %s"
        %(row['Object Id'] ), font = font  )
        draw2.text((0, 280), "predicted class / %s | true class / %s"
        %(row['pred'], row['class'] ), font = font  )
        draw2.text((0, 300), "CD 3 Intensity / %.1f | CD 4 / %.1f \n CD 8 / %.1f | CD 20 / %.1f"
        %(row['Dye 2 Nucleus Intensity'], row['Dye 3 Nucleus Intensity'], row['Dye 4 Nucleus Intensity'], row['Dye 6 Nucleus Intensity'] ) , font = font )

        captioned_img.save(cropped + '/' + save_name_color_variation, 'PNG')
    return 'done'


def generate_profiles(dataframe, rgb_orders):
    dataframe['written'] = dataframe.apply(lambda row: make_cell_profile(row), axis = 1)

if __name__ == '__main__':

    dir = os.path.dirname(__file__)
    dir = os.path.join(dir, '../data/')
    real_original = os.path.join(dir, 'real_original' + '/')
    original = os.path.join(dir, 'original' + '/')
    colored = os.path.join(dir, 'colored' + '/')
    cropped = os.path.join(dir, 'cropped' + '/')
    zoomed = os.path.join(dir, 'cropped' + '/')

    recent_run_dir = _retriev_recent_tf_log()
    #collect_tensor_files( recent_run_dir )
    #merge_main_output_dfs( recent_run_dir )

    df_path = recent_run_dir + 'collected_tensor_data.csv'
    df = pd.read_csv(df_path)

    spots = list(df['spot'].unique())
    str_spots = []
    def _add_zeros(string_num):
        while  len(string_num) < 3:
            string_num = '0' + string_num
        return string_num

    for s in spots:
        str_spots.append(_add_zeros(str(s)) )

    #print(df.head(), type(list(df['spot'].unique())[0]), list(df['spot'].unique()) )

    #take_first_image(real_original, original)
    rgb_orders = [['CD3', 'CD4', 'S029'], [ 'CD3', 'CD20', 'S029'], ['CD3', 'CD8','S029']]
    #color_originals(original, str_spots, rgb_orders )

    generate_profiles(df, rgb_orders)
