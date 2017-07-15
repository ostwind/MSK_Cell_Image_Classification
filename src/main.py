import os
import numpy as np
try:
    from tkinter import *
except ImportError:
    from Tkinter import *
from PIL import ImageTk, Image
from tifffile import TiffFile, imsave, imread
import itertools

dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
data = os.path.join(dir, 'data/')

window = Tk()
window.geometry('2000x1000+0+0')
window.title("")

#window.config(background = 'white')

batch_to_display = []

for dirpath, dirs, files in os.walk(data):
    for f in files:
        if '.png' in f:

            batch_to_display.append(dirpath+'%s'%(f))

            #break
#image = imread(data + 'B7H3_AFRemoved_pyr16_spot_004.tif')
#matrix = np.array(image, dtype = np.uint16)
#im = Image.fromarray(matrix)

def place_sample(img_index, panel, side = LEFT):
    im = Image.open(  batch_to_display[img_index] )
    im = im.resize((420+80, 240+26), Image.ANTIALIAS )
    tkimage = ImageTk.PhotoImage(im)
    imvar = Label(panel, image = tkimage)
    imvar.image = tkimage
    imvar.pack(side = side)

    input_panel = Frame(panel)
    #input_panel.config(background= 'white')

    var = IntVar()
    tumor= Checkbutton(input_panel, text= 'tumor', variable = var)
    tumor.pack(side = TOP )

    var = IntVar()
    unknown= Checkbutton(input_panel, text= 'unkwn', variable = var)
    unknown.pack(side = TOP )
    input_panel.pack(side = side)

num_panels = len(batch_to_display) // 3
left_overs = len(batch_to_display) % 3

for row in range( num_panels):
    sample_panel = Frame(window)
    #sample_panel.config(background = 'white')

    place_sample( 3*row, sample_panel)
    place_sample( 3*row + 1, sample_panel)
    place_sample( 3*row+2, sample_panel )

    sample_panel.pack(side = TOP)

if left_overs != 0:
    sample_panel = Frame(window)
    place_sample( len(batch_to_display) -1, sample_panel)
    if left_overs == 2:
        place_sample( len(batch_to_display) -2 , sample_panel)
    sample_panel.pack(side= TOP)

conclude_panel = Frame(window)


zoom = Button(conclude_panel, text='zoom / unzoom', pady = 75, padx = 50)
zoom.pack(side = LEFT)

submit = Button(conclude_panel, text='submit', pady = 75, padx = 75)
submit.pack(side = LEFT)

quit = Button(conclude_panel, text='quit', pady = 75, padx = 75, command = window.quit)
quit.pack(side= RIGHT)

conclude_panel.pack(side=TOP)

window.mainloop()





def gen_coord():
    grid = list(range(3))
    all_coordinates = grid*3
    i = -1
    for pos in all_coordinates:
        if pos == 0:
            i += 1
        yield pos, i

# def open_jpg(path):
#     jpg_to_tkinter = Image.open(path)
#     jpg_to_tkinter = jpg_to_tkinter.resize((210, 120), Image.ANTIALIAS)
#     return ImageTk.PhotoImage( jpg_to_tkinter)#, size=(3,3)  )
