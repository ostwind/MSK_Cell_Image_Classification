import os
import numpy as np
try:
    from tkinter import *
except ImportError:
    from Tkinter import *
from PIL import ImageTk, Image

def get_subdir(a_dir):
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]

class MainApplication(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.color_order = 0
        self.queue_pos = 0
        self.last_visited = 0

        self.fill_queues()

        #self.queue = self.queue[:100]
        #self.zoom_queue = self.zoom_queue[:100]
        ''' here we store user input
            knows: 0 if sample is unknwn, 1 if sample is knwn (default)
            labels: 0 if sample is not tumor (defaul), 1 if sample is tumor
            zooms: 0 if not zoomed (default), 1 if zoomed
        '''
        self.knows = [1 for x in self.queues[0]]
        self.labels = [0 for x in self.queues[0] ]
        self.zooms = [1 for x in self.queues[0] ]

        self.image_dim = [435, 335] # optimal display dim: [w = 1.3h, h]
        self.pg = 0
        self.last_pg = ''
        # points to UI elements to set images/ checks
        self.label_check_pointers = []
        self.know_check_pointers = []
        self.image_pointers = []
        self.caption_pointers = []

        self.label_storage_path = os.getcwd() + '/' + 'labels.txt'
        self.populate()
        self.import_labels()

    def fill_queues(self):#, path = zoom, zoom_path = zoom):
        self.queues = [[],[],[]]
        self.zoom_queues = [[],[],[]]
        color_orders = get_subdir(zoomed)
        for c in range(len(color_orders )):
            order_subdir = cropped + color_orders[c]
            #print(order_subdir)
            zoomed_order_subdir = zoomed + color_orders[c]
            for dirpath, dirs, files in os.walk( order_subdir ):
                i = 0
                for f in files:
                    i += 1
                    if '.png' in f and i < 100:
                        #print(order_subdir + '/' +  f)
                        self.queues[c].append(order_subdir + '/' +  f)
                        self.zoom_queues[c].append(zoomed_order_subdir + '/' + f )

    def place_sample(self, panel, side = LEFT):
        #print(self.zoom_queues[0][self.queue_pos])
        im = Image.open( self.zoom_queues[0][self.queue_pos] )
        im = im.resize((self.image_dim[0], self.image_dim[1]), Image.ANTIALIAS )
        tkimage = ImageTk.PhotoImage(im)
        imvar = Label(panel, image = tkimage)
        self.image_pointers.append(imvar)
        imvar.image = tkimage
        imvar.pack(side = LEFT)

        input_panel = Frame(panel)

        t_var = IntVar()
        tumor= Checkbutton(
        input_panel, text= 'tumor', variable = t_var, pady = 20, padx = 0,
        command = lambda: self.check_write(
        tumor))
        tumor.var = t_var
        tumor.pack(side = TOP )
        self.label_check_pointers.append( tumor )

        u_var = IntVar()
        unknown= Checkbutton(
        input_panel, text= 'unkwn', variable = u_var, pady = 20, padx = 0,
        onvalue = 0, offvalue = 1,
        command = lambda: self.check_write(unknown , array='knows'))
        unknown.var = u_var
        unknown.deselect()
        unknown.pack(side = TOP )
        self.know_check_pointers.append(unknown)

        caption = Label(input_panel, pady=20, text= self.queues[0][self.queue_pos].split('/')[-1].split('_')[0] )
        caption.pack(side = BOTTOM)
        self.caption_pointers.append(caption)

        z_var = IntVar()
        zoom = Button(
        input_panel, text='zoom',
        command = lambda: self.check_write(
        zoom, image_var = imvar, array = 'zooms') )#, pady = 50, padx = 30)
        zoom.var = z_var
        zoom.pack(side = BOTTOM)

        input_panel.pack(side = side)

    def place_row(self, gallery_panel):
        sample_panel = Frame(gallery_panel)

        if self.queue_pos < len(self.queues[0]):
            self.place_sample(sample_panel)
            self.queue_pos += 1
            while self.queue_pos % 3 != 0 and self.queue_pos < len(self.queues[0]):
                self.place_sample( sample_panel)
                self.queue_pos += 1

        sample_panel.pack(side = TOP)

    def pos(self, var, offset = 0):
        var = str(var)
        var = ''.join(i for i in var if i.isdigit())
        position = int( (int(var) - offset) /3) + (self.pg * 9)
        # user may quit halfway, assume everything upto the last bit manipulated
        # is actually valid to export, see self.export()
        if position > self.last_visited:
            self.last_visited = position
        return position

    def uncheck_all(self):
        for c in self.label_check_pointers:
            c.deselect()
        for d in self.know_check_pointers:
            d.deselect()

    def read_check(self, pos = '', checkbutton = ''):
        ''' select checkbuttons on a new page
            :param pos: if given, checks UI box according to image position in queue
            :param button: if given, checks UI box according to box's location
        '''
        if checkbutton:
            pos = self.pos(checkbutton.var)
        else:
            checkbutton = self.label_check_pointers[pos % 9]
            knows_checkbutton = self.know_check_pointers[pos % 9]
        if pos < len(self.queues[0]):
            # if unknwn button checked from prev page, check it now
            if self.knows[pos] == 0:
                knows_checkbutton.select()
            # if tumor is checked from prev page, check it now
            if self.labels[pos] == 1:
                #print('checkbutton checked at', pos, self.labels[pos])
                checkbutton.select()

    def check_write(self, checkbutton, image_var = '', array = 'labels'):
        ''' reads checkbutton states and writes them to self.knows, self.labels
            :param checkbutton: box to get value from
            :param image_var: if given, zoom image_var's image
            :param array: determine which array to write to (self.knows, self.labels)
        '''
        edit_queue = eval('self.' + array)
        pos = self.pos(checkbutton.var)

        if image_var:
            # zoom is a button. Instead of reading state, manually 0/1 flip
            edit_queue[ pos ] = 1 - edit_queue[ pos ]
            self.zoom_func(image_var, pos, edit_queue[pos])
            return
        edit_queue[ pos ] = checkbutton.var.get()
        #print(array, edit_queue, pos)

    def zoom_func(self, imvar, pos, zoom = 0):
        print(zoom)
        if zoom:
            self.update_img( imvar, self.zoom_queues[self.color_order][pos] )
        else:
            self.update_img( imvar, self.queues[self.color_order][pos] )

    def update_img(self, imvar, path):
        im = Image.open(path)
        im = im.resize((self.image_dim[0], self.image_dim[1]), Image.ANTIALIAS )
        tkimage = ImageTk.PhotoImage(im)
        imvar.configure(image = tkimage)
        imvar.image = tkimage
        imvar.pack(side = LEFT)

    def update_cap(self, cap, filepath):
        cell_id = filepath.split('/')[-1].split('_')[0]
        cap.config(text = cell_id)

    def update(self, next = False, prev = False ):
        """ Loads new page. Verifies page turning is possible, unchecks boxes,
            loads next image in the slot or a blank
            :param prev: bit to determine if next page or prev page, sets appropriate
            interval of images to traverse (self.pg * 9) in queue
        """
        if prev:
            if self.pg == 0:
                return
            self.pg -= 1

        elif next:
            if self.last_pg == self.pg:
                return
            self.pg += 1

        self.uncheck_all()
        blank = os.path.join(dir, 'data/util/pixel.png')

        self.queue_pos = self.pg * 9
        while self.queue_pos != (self.pg + 1) * 9:
            for pointer in self.image_pointers:
                if self.queue_pos >= len(self.queues[0]):
                    self.last_pg = self.pg
                    img_up_next = blank
                else:
                    img_up_next = self.zoom_queues[self.color_order][self.queue_pos]
                self.update_img(pointer, img_up_next )
                self.read_check(pos = self.queue_pos, checkbutton = ''  )
                self.update_cap(self.caption_pointers[self.queue_pos % 9], img_up_next)
                #print(self.queue_pos % 9, len(self.labels), self.queue_pos, self.labels[self.queue_pos])
                self.queue_pos += 1

    def toggle_color(self, button):
        self.color_order = (self.color_order + 1) % 3
        cur_color = get_subdir(zoomed)[self.color_order]
        cur_color =  '\n'.join(cur_color.split('_'))
        button.config(text = cur_color)
        self.update()

    def populate( self, imgs_per_page = 9 ):
        gallery_panel = Frame(window)
        for i in range(3):
            self.place_row(gallery_panel)
        gallery_panel.pack(side = LEFT)

        conclude_panel = Frame(window)

        toggle_ = Button(
        conclude_panel, text='change \n color \n scheme', pady = 50, padx = 45,
        command = lambda: self.toggle_color( toggle_ ))
        toggle_.pack(side = TOP)

        prev_ = Button(
        conclude_panel, text='prev', pady = 50, padx = 50,
        command = lambda: self.update( prev = True))
        prev_.pack(side = TOP)

        next_ = Button(
        conclude_panel, text='next', pady = 50, padx = 50,
        command = lambda: self.update( next = True) )
        next_.pack(side = TOP)

        submit = Button(
        conclude_panel, text='submit', pady = 50, padx = 43,
        command = lambda: self.export_labels() )
        submit.pack(side = TOP)

        instr = Label(conclude_panel, pady = 50,
        text = 'check unknwn only due to image quality\n press submit to save progress \n labels stored at \n %s' %(self.label_storage_path))
        instr.pack(side = BOTTOM)

        quit = Button(conclude_panel, text='quit', pady = 50, padx = 50,
        command = window.quit)
        quit.pack(side= BOTTOM)

        conclude_panel.pack(side=RIGHT)#, pady = 20)

    def export_labels(self):
        label_dict = dict()
        self.labels = [x if x != 0 else x - 1 for x in self.labels ]
        for i in range(self.last_visited+1):
            label_dict[ self.queues[0][i] ] = self.labels[i] * self.knows[i]
        with open(self.label_storage_path, 'w') as f:
            f.write( str(label_dict) )

    def import_labels(self):
        if not os.path.isfile(self.label_storage_path):
            with open(self.label_storage_path, 'w') as e:
                e.write( 'dict()')

        with open(self.label_storage_path, 'r') as f:
            label_dict = eval(f.read())

        for i in range(len(self.queues[0])):
            if self.queues[0][i] in label_dict.keys():
                #print('pos at _ have value _ ', i, label_dict[self.queue[i]])
                if label_dict[self.queues[0][i]] == 0:
                    self.knows[i] = 0
                if label_dict[self.queues[0][i]] == 1:
                    self.labels[i] = 1
                if label_dict[self.queues[0][i]] == -1:
                    self.labels[i] = 0
                if i < 9:
                    self.read_check(pos = i)
if __name__== "__main__":

    dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    cropped = os.path.join(dir, 'data/cropped/')
    zoomed = os.path.join(dir, 'data/zoomed/')

    window = Tk()
    window.geometry('2000x1000+0+0')
    window.title("Tumor Classification UI v0.0")

    MainApplication(window).pack(side='top', fill = 'both', expand = True)

    window.mainloop()

# TODO
# color key
# user chosen proteins
# make zoom images






# def gen_coord():
#     grid = list(range(3))
#     all_coordinates = grid*3
#     i = -1
#     for pos in all_coordinates:
#         if pos == 0:
#             i += 1
#         yield pos, i
