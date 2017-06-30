import os
from libxmp import XMPFiles, consts
from libxmp.utils import file_to_dict, object_to_dict
from metadata_handler import Table, EditTable
import xml.etree.ElementTree as ET
import re

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
data_dir = os.path.join(dir, 'temp.tif')
xml_dir = os.path.join(dir, 'src/template.xml')
GE = os.path.join(dir, 'data/big_imgs_GE/')
lab_info = os.path.join(dir, 'src/lab_info.txt')
PE = os.path.join(dir, 'data/PE/PerkinElmerEx.tiff')
'''
gen header and xml templates
edit templates
put new xml files into tiff
collect tiff into the same file calling tiffcp
*** currently does not edit the PerkinElmer template after embedding
'''

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

def add_content(xml):

    with open(lab_info, 'r') as f:
        filestring = f.read()
        filestring = '\t\t\t' + filestring

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
                            #print(line)
                            if "PerkinElmer-QPI-ImageDescription" in line:
                                f1.write( add_content(lab_info) )
                                continue
                            f1.write(line )
                embed_xmp(
                tiff_path = GE + tif_name, xmp_path = output_meta + filename )

write_xmp()

# def add_content2(xml):
#
#     with open(xml, 'r') as f:
#         data = f.read()
#
#     tree = ET.fromstring(data)
#     file_string = str(ET.tostring(tree))
#
#     file_string = file_string.replace('b\'', '')
#     file_string = file_string.replace('\\n', '')
#     file_string = file_string.replace(' ', '')
#     file_string = file_string.replace('\t', '')
#     #file_string = '\t\t\t' + file_string
#
#     return file_string

# xmpfile = XMPFiles( file_path = PE)
# xmp = xmpfile.get_xmp()
#
# with open('no_mod_orig_PE.xml', 'w') as f:
#     f.write(str(xmp))
