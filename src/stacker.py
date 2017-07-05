import os
from libxmp import XMPFiles, consts
from libxmp.utils import file_to_dict, object_to_dict
from metadata_handler import Table, EditTable
import xml.etree.ElementTree as ET
import re

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
xml_dir = os.path.join(dir, 'src/no_mod_orig_GE.xml')
lab_info = os.path.join(dir, 'src/lab_info.txt')
subpage_lab_info = os.path.join(dir, 'src/subpage_lab_info.txt')
GE = os.path.join(dir, 'data/copyGE/')
big_GE = os.path.join(dir, 'data/big_imgs_GE/')

'''
gen header and xml templates
edit templates
put new xml files into tiff
collect tiff into the same file calling tiffcp
*** currently does not edit the PerkinElmer template after embedding
'''

def extract_xmp(tiff_path):
    ''' extracts xmp of tiff file at tiff_path to the read_meta dir
    '''
    output_meta = os.path.join(dir, 'read_meta/')
    f_name =  tiff_path.split('/')[-1]

    xmpfile = XMPFiles( file_path=  tiff_path, open_forupdate=True )
    xmp = xmpfile.get_xmp()
    with open(output_meta + f_name + '.xml', "w") as f1:
        f1.write(str(xmp))

#extract_xmp(GE + 'FOXP3_AFRemoved_pyr16_spot_004.tif')

def embed_xmp(tiff_path, xmp_path):
    ''' places a xml file into a tiff file
    '''
    #print(tiff_path, xmp_path)
    xmpfile = XMPFiles( file_path=  tiff_path, open_forupdate=True )
    xmp = xmpfile.get_xmp()
    with open(xmp_path, 'r') as f:
        read = f.read()
        xmp.parse_from_str(str(read))
    if not xmpfile.can_put_xmp(xmp):
        raise IOError(("I/O error: cannot place {0} in {1}".format(xmp_path, file_path) ))
    xmpfile.put_xmp(xmp)
    xmpfile.close_file()

def add_content(lab_settings):
    ''' reads lab settings from a .txt
    '''
    with open(lab_settings, 'r') as f:
        filestring = f.read()
        filestring = '' + filestring
    return filestring

def write_xmp(file_dir = GE):
    ''' writes on a copy of lab settings, from PE template
        into an intermediate dir
    '''
    output_meta = os.path.join(dir, 'correct_meta/')
    for subdir, dirs, files in os.walk(file_dir):
        for tif_name in files:
            if '.tif' in tif_name:
                filename = tif_name.split('.')[0] + '.xml'
                with open(xml_dir, 'r') as f:
                    with open(output_meta + filename, "w") as f1:
                        for line in f:
                            if 'MSK_CYTELL' in line:#"PerkinElmer-QPI-ImageDescription" in line:
                                f1.write( add_content(lab_info) )
                                continue
                            f1.write(line )
                embed_xmp(
                tiff_path = GE + tif_name, xmp_path = output_meta + filename )

#write_xmp()

# generating the PerkinElmer template for template.xml
# PE = os.path.join(dir, 'data/GE/ICOS_AFRemoved_pyr16_spot_004.tif')
# xmpfile = XMPFiles( file_path = PE)
# xmp = xmpfile.get_xmp()
#
# with open('no_mod_orig_GE.xml', 'w') as f:
#     f.write(str(xmp))
