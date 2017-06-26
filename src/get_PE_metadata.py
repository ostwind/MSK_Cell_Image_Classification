import os

# open the PerkinElmer file using python module TIFFfile
import tifffile as tiff

# create relative pathing to a PerkinElmer TIFF file
dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
data_dir = os.path.join(dir, 'data/PE/PerkinElmerEx.tiff')

# open TIFF file using module and iterate through pages
with tiff.TiffFile(data_dir) as tif:
    for page in tif:
        # print meta data info, see attachment for output
        print(page.info())
