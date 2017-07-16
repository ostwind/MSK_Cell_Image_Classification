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
data = os.path.join(dir, 'data/display/')
zoom = os.path.join(dir, 'data/zoom/')

class MainApplication(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.queue_pos = 0
        self.queue = self.fill_queue(data)
        self.zoom_queue = self.fill_queue(zoom)

        ''' here we store user input
            knows: 0 if sample is unknwn, 1 if sample is knwn (default)
            labels: 0 if sample is not tumor (defaul), 1 if sample is tumor
            zooms: 0 if not zoomed (default), 1 if zoomed
        '''
        self.knows = [1 for x in self.queue]
        self.labels = [0 for x in self.queue ]
        self.zooms = [0 for x in self.queue ]

        self.populate()

    def fill_queue(self, path):
        filled_queue = []
        for dirpath, dirs, files in os.walk(path):
            for f in files:
                if '.png' in f:
                    filled_queue.append(dirpath+'%s'%(f))
        return filled_queue

    def place_sample(self, panel, side = LEFT):

        print(self.queue_pos, len(self.queue))
        im = Image.open(  self.queue[self.queue_pos] )
        im = im.resize((420+80, 240+26), Image.ANTIALIAS )
        tkimage = ImageTk.PhotoImage(im)
        imvar = Label(panel, image = tkimage)
        imvar.image = tkimage
        imvar.pack(side = side)

        input_panel = Frame(panel)

        t_var = IntVar()
        tumor= Checkbutton(
        input_panel, text= 'tumor', variable = t_var, command = lambda: self.flick(
        self.pos(t_var)))
        tumor.var = t_var
        tumor.pack(side = TOP )

        u_var = IntVar()
        unknown= Checkbutton(
        input_panel, text= 'unkwn', variable = u_var, pady = 20, padx = 20,
        command = lambda: self.flick(
        self.pos(u_var, offset = 1), array='knows'))
        unknown.var = u_var
        unknown.pack(side = TOP )

        z_var = IntVar()
        zoom = Button(
        input_panel, text='zoom/unzoom', command = lambda: self.flick(
        self.pos(z_var, offset = 2) , image_var = imvar, array = 'zooms'))#, pady = 50, padx = 30)
        zoom.var = z_var
        zoom.pack(side = BOTTOM)

        input_panel.pack(side = side)

    def pos(self, var, offset = 0):
        var = str(var)
        var = ''.join(i for i in var if i.isdigit())
        print(var)
        return int( (int(var) - offset) /3)

    def flick(self, pos, image_var = '', array = 'labels'):
        edit_queue = eval('self.' + array)
        print(pos, len(edit_queue))
        if edit_queue[pos] == 0:
            edit_queue[pos] = 1
        else:
            edit_queue[pos] = 0
        if image_var:
            self.zoom_func(image_var, pos, edit_queue[pos])
        print(array, pos,  edit_queue)

    def place_row(self):
        sample_panel = Frame(window)

        if self.queue_pos < len(self.queue):
            self.place_sample(sample_panel)
            self.queue_pos += 1
            while self.queue_pos % 3 != 0 and self.queue_pos < len(self.queue):
                self.place_sample( sample_panel)
                self.queue_pos += 1

        sample_panel.pack(side = TOP)

    def zoom_func(self, imvar, pos, zoom = 0):
        if zoom:
            self.update_img( imvar, self.zoom_queue[0] )
        else:
            self.update_img( imvar, self.queue[0] )

    def update_img(self, imvar, path):
        im = Image.open(path)
        im = im.resize((420+80, 240+26), Image.ANTIALIAS )
        tkimage = ImageTk.PhotoImage(im)
        imvar.configure(image = tkimage)
        imvar.image = tkimage
        imvar.pack(side = LEFT)

    def populate( self, imgs_per_page = 9 ):
        l = window.pack_slaves()
        if l:
            for item in l:
                item.destroy()

        num_rows = imgs_per_page // 3
        left_overs = imgs_per_page % 3

        while self.queue_pos < len(self.queue):
            self.place_row()
        ####################

        conclude_panel = Frame(window)

        submit = Button(
        conclude_panel, text='submit', pady = 50, padx = 50,
        command = lambda: self.update() )
        submit.pack(side = LEFT)

        quit = Button(conclude_panel, text='quit', pady = 50, padx = 50, command = window.quit)
        quit.pack(side= RIGHT)

        conclude_panel.pack(side=TOP)

if __name__== "__main__":

    window = Tk()
    window.geometry('2000x1000+0+0')
    window.title("Tumor-or-Not v1.0")

    MainApplication(window).pack(side='top', fill = 'both', expand = True)

    #window.bind("<Return>", MainApplication.update)
    window.mainloop()

####################







# def gen_coord():
#     grid = list(range(3))
#     all_coordinates = grid*3
#     i = -1
#     for pos in all_coordinates:
#         if pos == 0:
#             i += 1
#         yield pos, i
