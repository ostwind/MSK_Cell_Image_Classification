import os
import numpy as np
try:
    from tkinter import *
except ImportError:
    from Tkinter import *
from PIL import ImageTk, Image
from functools import partial


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

        ''' storing user input
            ZOOM BUTTON < == > self.zooms ARRAY
            zooms: 0 if not zoomed (default), 1 if zoomed

            RADIO BUTTONS < == > self.labels MATRIX
            |tumor: 0 if sample is not tumor (default), 1 if sample is tumor
            |t-cell/helper :
            |t-cell/killer :
            |knows: 0 if sample is unknwn, 1 if sample is knwn (default)
            |sequence of image paths shown
            |(see self.import_labels for matrix initialization)
        '''

        self.zooms = [1 for x in self.queues[0] ]

        self.image_dim = [435, 335] # optimal display dim: [w = 1.3h, h]
        self.pg = 0
        self.last_pg = ''
        # points to UI elements to set images/ radio buttons
        self.radio_button_pointers = []
        self.image_pointers = []
        self.caption_pointers = []
        self.radio_variable_pointers =[]
        window.bind("<Key>", lambda event: self.key(event) )

        self.label_storage_path = dir+ '/' + 'labels.txt'
        self.populate()
        self.import_labels()
        print(self.zoom_queues[0] == self.labels[4])

    def key(self, event):
        window.focus_set()
        #print(event)
        #print(repr(event.char))
        if event.keycode == 114:
            #print('--> right arrow')
            self.update(next=True)

        if event.keycode == 113:
            #print('<-- left arrow')
            self.update(prev=True)

        if event.keycode == 111:
            #print('^ up arrow')
            self.toggle_color()

        #if event.keycode == 116:
        #    print('v down arrow')

        checkbuttons_numpad = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        if event.char in checkbuttons_numpad:
            position = checkbuttons_numpad.index(event.char)
            sample_position = self.queue_pos - 9 + position

            def cycle(sample_position):
                sample_cur_label = self._retrieve_label_index(sample_position)
                sample_next_label = (sample_cur_label + 1) % 4
                self._write_label_index(sample_position, sample_next_label)

            cycle(sample_position)
            self.update()

    def fill_queues(self):
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
                    if '.png' in f and i < 101:
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
        MODES = [
        ("tumor", 0),
        (" CD8+", 1),
        (" CD4+", 2),
        ("unknw", 3)]
        v = IntVar()
        v.set(3)
        for text, mode in MODES:
            b = Radiobutton(input_panel, text=text, variable=v, value=mode,
            command = lambda: self.radio_button_write(b ,v) )
            b.var = v
            b.pack(side=TOP)
            self.radio_button_pointers.append( b )

        self.radio_variable_pointers.append(v)

        caption = Label(input_panel, pady=20, text= self.queues[0][self.queue_pos].split('/')[-1].split('_')[0] )
        caption.pack(side = BOTTOM)
        self.caption_pointers.append(caption)

        z_var = IntVar()
        zoom = Button(
        input_panel, text='zoom',
        command = lambda: self.zoom_record(
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

    def _retrieve_label_index(self, sample_index):
        if sample_index >= len(self.queues[0]):
            return

        for array in self.labels[:-1]:
            if array[sample_index]:
                return self.labels.index(array)

    def read_to_radio(self, pos = '', checkbutton = ''):
        ''' select radio buttons according to matrix
            :param pos: if given, checks UI box according to image position in queue
            :param button: if given, checks UI box according to box's location
        '''
        for sample in range(  self.queue_pos - 9, self.queue_pos ):
            radio_var_to_set = self.radio_variable_pointers[ sample % 9 ]
            radio_var_to_set.set(self._retrieve_label_index(sample) )

    def _write_label_index(self, sample_position, selection):
        #print(selection)
        for ind in range(len(self.labels[:-1])):
            if ind == selection:
                self.labels[ind][sample_position] = 1
            else:
                self.labels[ind][sample_position] = 0

    def radio_button_write(self, button, v):
        button_position = self.radio_button_pointers.index(button)
        sample_position = (button_position - (button_position % 4) ) //4
        sample_position += self.queue_pos - 9
        selection = v.get()
        self._write_label_index( int(sample_position), int(selection) )

    def zoom_record(self, button):
        pos = self.pos(button.var)
        if image_var:
            # zoom is a button. Instead of reading state, manually 0/1 flip
            self.zooms[pos] = 1 - self.zooms[ pos ]
            self.zoom_func(image_var, pos, self.zooms[pos])
            return

    def zoom_func(self, imvar, pos, zoom = 0):
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

        blank = os.path.join(dir, '../../data/util/pixel.png')

        self.queue_pos = self.pg * 9
        while self.queue_pos != (self.pg + 1) * 9:
            for pointer in self.image_pointers:
                if self.queue_pos >= len(self.queues[0]):
                    self.last_pg = self.pg
                    img_up_next = blank
                    #print(self.queue_pos, len(self.queues[0]) )
                else:
                    img_up_next = self.zoom_queues[self.color_order][self.queue_pos]
                self.update_img(pointer, img_up_next )
                # update checkbutton selections
                self.read_to_radio(  )
                # update captions
                self.update_cap(self.caption_pointers[self.queue_pos % 9], img_up_next)
                #print(self.queue_pos % 9, len(self.labels), self.queue_pos, self.labels[self.queue_pos])
                self.queue_pos += 1

    def toggle_color(self):
        self.color_order = (self.color_order + 1) % 3
        cur_color = get_subdir(zoomed)[self.color_order]
        cur_color =  '\n'.join(cur_color.split('_'))
        self.toggle_.config(text = cur_color)
        self.update()

    def populate( self, imgs_per_page = 9 ):
        gallery_panel = Frame(window)

        for i in range(3):
            self.place_row(gallery_panel)
        gallery_panel.pack(side = LEFT)

        conclude_panel = Frame(window)

        self.toggle_ = Button(
        conclude_panel, text='change \n color \n scheme', pady = 50, padx = 45,
        command = lambda: self.toggle_color( ))
        self.toggle_.pack(side = TOP)

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
        with open(self.label_storage_path, 'w') as f:
            f.write( repr(self.labels ) )

    def import_labels(self):
        if not os.path.isfile(self.label_storage_path):
            tumor_indicator_vect = [0 for x in self.queues[0]]
            tcell_helper_indicator_vect = [0 for x in self.queues[0]]
            tcell_killer_indicator_vect = [0 for x in self.queues[0]]
            know_indicator_vect = [1 for x in self.queues[0] ]

            self.labels = [ tumor_indicator_vect, tcell_helper_indicator_vect,
            tcell_killer_indicator_vect, know_indicator_vect, self.zoom_queues[0] ]

            self.export_labels()

        with open(self.label_storage_path, 'r') as f:
            self.labels = eval(f.read())

        # if current sample belongs on first page, tick box accordingly
        for i in range(9):
            self.read_to_radio(pos = i)

if __name__== "__main__":

    #dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    #cropped = os.path.join(dir, '../data/cropped/')
    #zoomed = os.path.join(dir, '../data/zoomed/')
    dir = os.path.dirname(__file__)
    zoomed = os.path.join(dir, '../../data/zoomed/')
    cropped = os.path.join(dir, '../../data/cropped/')

    window = Tk()
    window.geometry('2000x1000+0+0')
    window.title("Tumor Classification UI v0.0")

    MainApplication(window).pack(side='top', fill = 'both', expand = True)

    window.mainloop()

# TODO
