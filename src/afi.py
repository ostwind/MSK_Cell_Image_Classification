import os
import xml.etree.cElementTree as ET

from tkinter import *
#import lxml.etree as etree



#dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
#tif_dir = os.path.join(dir, 'data/copyGE', '')
# /home/lihan/Documents/image/data/mockGE2
#print(tif_dir)

def init_afi(dirpath, spot):
    afi_name = 'PR_Spot%s_1.afi' %(spot)
    afi_loc = os.path.join(dirpath, afi_name)
    #if afi_loc.is_dir():
    return

def collect_spots(path):
    # collect filenames at this path by spot
    spot_dict = dict()
    all_files = next(os.walk(path))[2]
    for f in all_files:

        #GE format: MarkerName_blah_spot_SpotNum.tif
        #counting 4 '_' ensures this is a GE tif
        if '.tif' in f and f.count('_') == 4:
            #extract spot number from file name
            spot = f.split('.')[0].split('_')[-1]
            if spot in spot_dict.keys():
                spot_dict[spot].append(f)
            else:
                spot_dict[spot] = [f]
    return spot_dict


def link(start_dir):
    ''' traverses all subdir of start_dir, creates a tif_ledger if .tif found
        writes .afi for each spot found at dirpath to dirpath
        (e.g. PR_Spot4_1.afi, PR_Spot2_1.afi at D:\PR_1\AFRemoved\ )
    '''
    found_tif = False

    for dirpath, dirs, files in os.walk(start_dir):
        dirpath = os.path.join(dirpath, '')
        tif_ledger = collect_spots(dirpath)
        #print(tif_ledger)
        for spot in tif_ledger.keys():
            filename = 'PR_Spot%s_1.afi' %(int(spot))
            #with open(, 'w') as write_to:

            root = ET.Element("ImageList")

            for tif_name in tif_ledger[spot]:

                found_tif = True

                image_child = ET.SubElement(root, "Image")
                #PC_dir = "D:\PR_1\AFRemoved\filename.tif"
                path_child = ET.SubElement(
                image_child, "Path").text = str(dirpath + tif_name)

                bitdepth_child = ET.SubElement(
                image_child, "BitDepth").text = "16"

                channelname_child = ET.SubElement(
                image_child, "ChannelName").text = tif_name.split('_')[0]#"CD25"

                # what is the HALO .afi name scheme?

            tree = ET.ElementTree(root)
            print(dirpath + filename)
            tree.write(dirpath + filename)
            tree = 0 #I have no idea how to clear tree and root vals from memory
            root = 0
    return found_tif

def path_exists(path):
    #print(path)
    return os.path.exists(os.path.dirname(path))

def show_entry_fields():
   path = os.path.join(e1.get(), '')
   print(path)
   if not path_exists(path):
       status.set("Not a valid directory path")
       return

   # file written by link after calling it, link also a bool indicating if tiff found at path
   if not link(path):
       status.set('No .tif or .tiff found at this directory')
       return
   status.set('%s written to %s' %('filename', path) )

master = Tk()
master.geometry('900x100+300+300')
master.title("HALO's TIFF Linker")
Label(master, text="Root Directory \n to Traverse").grid(row=0)

status = StringVar()
Label(master, textvariable=status).grid(column=1, row=1)
status.set('TIFFs in directory are stacked by spots (spatial locations)')

e1 = Entry(master, textvariable=1, width=80)
e1.pack(ipady=3)
e1.insert(10,"")

e1.grid(row=0, column=1)

Button(master, text='Quit', command=master.quit).grid(row=3, column=2, sticky=W, pady=0)
Button(master, text='Stack', command=show_entry_fields).grid(row=3, column=1, sticky=W, pady=0)

mainloop( )
