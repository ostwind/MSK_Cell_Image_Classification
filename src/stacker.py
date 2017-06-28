import os
from libxmp import XMPFiles, consts
from libxmp.utils import file_to_dict, object_to_dict
from metadata_handler import Table, EditTable

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
data_dir = os.path.join(dir, 'temp.tif')
xml_dir = os.path.join(dir, 'src/template.xml')
GE = os.path.join(dir, 'data/mock_GE/')

'''
gen header and xml templates
edit templates
put new xml files into tiff
collect tiff into the same file calling tiffcp
'''

def gen_template():
    ''' extracts the PerkinElmer xmp as template
        and stores it at src
    '''
    return

def embed_xmp(tiff_path = data_dir, xmp_path = xml_dir):
    ''' places a xml file into a tiff file
    '''
    xmpfile = XMPFiles( file_path=data_dir, open_forupdate=True )
    xmp = xmpfile.get_xmp()
    with open(xml_dir, 'r') as f:
        read = f.read()
        xmp.parse_from_str(str(read))

    if not xmpfile.can_put_xmp(xmp):
        raise IOError(("I/O error: cannot place {0} in {1}".format(xmp_path, file_path) ))
    xmpfile.put_xmp(xmp)

def write_xmp(file_dir = GE):
    ''' writes on a copy of lab settings, from PE template
        into an intermediate dir
    '''
    in_template = os.path.join(dir, 'src/inner_template.xml')
    output_meta = os.path.join(dir, 'correct_meta/')
    for subdir, dirs, files in os.walk(file_dir):
        for f in files:
            if '.tif' in f:
                filename = f.split('.')[0] + '.xml'
                with open(xml_dir, 'r') as f:
                    with open(output_meta + filename, "w") as f1:
                        #print(xml_dir, output_meta+filename)
                        Flag = 1
                        for line in f:
                            if Flag:
                                if "&lt;PerkinElmer-QPI-ImageDescription&gt;" in line:
                                    Flag = 0
                                    continue
                                f1.write(line )
                            else:
                                Flag = 1
                                f1.write(line )

write_xmp()

# def f(file_dir = GE):
#     for subdir, dirs, files in os.walk(file_dir):
#         for f in files:
#             if '.tif' in f:
#                 inner_str = write_xmp()
#                 #embed_xmp
#     return
#
# table_loc = os.getcwd()+'/metadata.json'
# t = Table(table_loc)
# t.view()
