import os
from libxmp import XMPFiles, consts
from libxmp.utils import file_to_dict, object_to_dict

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
data_dir = os.path.join(dir, 'temp.tif')
xml_dir = os.path.join(dir, 'src/template.xml')
GE = os.path.join(dir, 'data/mock_GE/')

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

def markers_list(file_dir = GE):
    ''' returns a list of markers in given directory
    '''
    markers = []
    for subdir, dirs, files in os.walk(file_dir):
        for f in files:
            if '.tif' in f:
                markers.append(f.split('_')[0])
    return markers

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
                        print(xml_dir, output_meta+filename)

                        Flag = True
                        for line in f:
                            if Flag:
                                if "PerkinElmer-QPI-ImageDescription" in line:
                                    Flag = False
                                    continue
                                f1.write(line)
                            else:
                                if "PerkinElmer-QPI-ImageDescription" in line:
                                    Flag = True

                        f.close()
                        f1.close()
    return

write_xmp()

def f(file_dir = GE):
    for subdir, dirs, files in os.walk(file_dir):
        for f in files:
            if '.tif' in f:
                inner_str = write_xmp()
                #embed_xmp

    return

f()

# def write_header(template = str(dir) + '/src/header_template.xml'):
#     #print(data_dir)
#     xmpfile = XMPFiles( file_path=data_dir, open_forupdate=True )
#     xmp = xmpfile.get_xmp()
#
#     #xmp = xmpfile.get_xmp()
#     xmp.set_property(consts.XMP_NS_RDF, u'Alt', 'test1' )
#
#     with open(xml_dir, 'w') as output:
#         output.write(xmp.serialize_and_format())
#
#     xmpfile.put_xmp(  xmp   )
#     #print(diction)
#     xmpfile.close_file()
#     return 0
