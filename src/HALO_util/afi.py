import os
import xml.etree.cElementTree as ET
try:
    from tkinter import *
except ImportError:
    from Tkinter import *

# /home/lihan/Documents/image/HALO_data/S11_27456_1_1

def collect_spots(path):
    # collect filenames at this path by spot
    spot_dict = dict()
    all_files = next(os.walk(path))[2]
    for f in all_files:
        #GE format: MarkerName_blah_spot_SpotNum.tif
        #counting 4 '_' ensures this is a GE tif
        if '.tif' in f and f.count('_') >= 4:
            #extract spot number from file name
            spot = f.split('.')[0].split('_')[-1]
            if spot in spot_dict.keys():
                spot_dict[spot].append(f)
            else:
                spot_dict[spot] = [f]
    return spot_dict

def case_name(path):
    parent_dir = path.split('/')[-2]
    if '_' not in parent_dir:
        return parent_dir
    return parent_dir #parent_dir.split('_')[0] + '_' + parent_dir.split('_')[1]

def channel_name(file_name):
    file_name_in_list = file_name.split('_')
    cname = file_name_in_list[0]

    if len(file_name_in_list[1]) < 3:
        cname += '_' + file_name_in_list[1]
    if  len(file_name.split('_')[2]) < 3:
        cname += '_' + file_name_in_list[2]
    if 'dapi' in file_name:
        return 'DAPI%s'%( int(cname[1:]) )
    return cname

def write_afi(dirpath, mask_dir, spot, tif_ledger):
    ''' Given where to write, which spot it is writing for
        and ledger (so it can look up filenames), conjugates the filename
        then populates a .afi file, writing to dir
        @param str dirpath: user input passed through traverse from GUI
        @param str mask_dir: user input
        @param str spot: within a dir, each afi corresponds to a spot, which
        links a group of TIF
        @param dic tif_ledeger: keys of this dic are spots, provides a list of filenames
        which belong to this spot
    '''
    # read S[year num]_[5 digit serial]_Spot[spot num]
    filename = '%s_Spot%s.afi' %( case_name(dirpath),  int(spot))
    root = ET.Element("ImageList")

    if mask_dir:
        path_to_write = mask_dir

    else:
        path_to_write = ''

    # next code block to sort channel names alphabetically, since S001-> DAPI1
    # create channel names linked w/ tif_name, sort array,
    channel_names_to_sort = []
    unique_channels = []
    for tif_name in tif_ledger[spot]:
        channel_names_to_sort.append( [channel_name(tif_name), tif_name] )
        unique_channels.append(channel_name(tif_name))
    # duplicate channel name check
    if len(channel_names_to_sort) != len(set(unique_channels)):
        update_usr(listbox, 'files with duplicate channel names at %s, skipped! ' %(dirpath) )
        return False

    channel_names_to_sort.sort(key=lambda x: x[0])

    for c_name_and_tif_name in channel_names_to_sort:

        c_name, tif_name = c_name_and_tif_name
        image_child = ET.SubElement(root, "Image")

        path_child = ET.SubElement(
        image_child, "Path").text = str(tif_name)

        bitdepth_child = ET.SubElement(
        image_child, "BitDepth").text = "16"

        channelname_child = ET.SubElement(
        image_child, "ChannelName").text = c_name

    tree = ET.ElementTree(root)
    #print(dirpath + filename)
    tree.write(dirpath + filename)

def traverse(start_dir, mask_dir, num_stains):
    ''' traverses all subdir of start_dir, creates a tif_ledger if .tif found
        writes .afi for each spot found at dirpath to dirpath
        (e.g. PR_Spot4_1.afi, PR_Spot2_1.afi at D:\PR_1\AFRemoved\ )
        @param str start_dir: GUI field 1, actual path leading to TIF files
        @param None or str mask_dir: GUI field 2, path of another machine to embed into afi
        @param None or str num_stains: GUI field 3, prints warning if spot does not contain this many files
        @return dic tif_ledger: an empty tif_ledger indicates no TIF were found
    '''

    tif_ledger = dict()
    for dirpath, dirs, files in os.walk(start_dir):
        dirpath = os.path.join(dirpath, '')
        tif_ledger = collect_spots(dirpath)
        for spot in tif_ledger.keys():
                if num_stains and len(tif_ledger[spot]) != int(num_stains):
                    update_usr(listbox, 'Spot %s at %s has %s .tif files, no .afi written' %(spot, dirpath, len(tif_ledger[spot])))
                    update_usr(listbox, '!! no .afi written !!' )
                    continue
                if  not num_stains:
                    update_usr(listbox, 'Spot %s at %s has %s .tif files' %(spot, dirpath, len(tif_ledger[spot])))

                write_afi(dirpath, mask_dir, spot, tif_ledger)
    return tif_ledger

def path_exists(path):
    return os.path.exists(os.path.dirname(path))

master = Tk()
master.geometry('1200x500+300+300')
master.title("HALO's TIFF Linker")

top_label = Label(master, text="Root Directory \n to Traverse" )
top_label.grid(row=0, padx = 20)
top_label.config(font=('Helvetica',15))

mid_label = Label(master, text="Directory Mask ")
mid_label.grid(row=1, padx = 20)
mid_label.config(font=('Helvetica',15))

bot_label = Label(master, text="Number of stains \n used by study ")
bot_label.grid(row=2, padx = 20)
bot_label.config(font=('Courier',12))

e1 = Entry(master, textvariable=1, width=80)
e1.grid(row=0, column=1, pady = 20, padx = 10)

e2 = Entry(master, width = 80)
e2.grid(row = 1, column = 1, pady = 20, padx = 10)

e3 = Entry(master, width = 10)
e3.grid(row = 2, column = 1)


w = Canvas(master, width = 120, height=15)
w.grid(row=4, column=1, pady = 0)

status = StringVar()
f = Label(w, textvariable=status)

scrollbar = Scrollbar(w)
scrollbar.pack(side=RIGHT, fill=Y)

listbox = Listbox(w, width = 120, height = 15)
listbox.pack()

listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)


def update_usr(listbox, text = '', end = False ):
    if end:
        listbox.insert(END, '    ' + text)
        listbox.insert(END, ' ')
    else:
        listbox.insert(0, '    ' + text)
        listbox.insert(0, ' ')
def show_entry_fields(listbox):
   ''' calls afi writing functions according to user input into GUI
        e1 corresponds to first field, e2 to second field, so on.
        traverse( ) takes all user inputs and executes the task, also serves as boolean
        in case directory does not contain any .tif
   '''
   path = os.path.join(e1.get(), '')
   if not path_exists(path):
       update_usr(listbox, 'Not a valid directory path')
       return

   mask_dir = e2.get() #os.path.join(e2.get(), '')

   num_stains = (e3.get())
   if num_stains and not num_stains.isdigit():
       update_usr(listbox, 'Number of Stains not an integer')
       return

   # file written by link after calling it, link also a bool indicating if tiff found at path
   if not traverse(path, mask_dir, num_stains):
       update_usr(listbox, 'No .tif or .tiff found anywhere in %s' %(path))
       return

Button(master, text='Link', command= lambda: show_entry_fields(listbox)).grid(row=3, column=1, pady=20)
Button(master, text='Quit', command=master.quit).grid(row=3, column=0, sticky=W, padx = 75, pady=20)

update_usr(listbox, '________General Notes________', True)
update_usr(listbox, 'all subdirectories from the path provided are traversed and searched', True)
update_usr(listbox, 'THIS PROGRAM WILL RE-WRITE EXISTING AFI IF THEY ARE FOUND W/ SAME NAMING SCHEME IN A SUBDIRECTORY', True)
update_usr(listbox, '''afi linking files are written to where linked TIFFs are''', True)
update_usr(listbox, 'In a directory, TIFFs belonging to the same spot are linked', True)

update_usr(listbox, '________Num of Stains________', True)
update_usr(listbox, """Warnings are printed if .tif files for one spot do not contain""", True)
update_usr(listbox, 'designated number of stains (leave blank for no warning)', True)

mainloop( )
