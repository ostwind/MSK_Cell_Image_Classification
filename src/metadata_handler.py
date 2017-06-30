import pandas as pd
import os
import json
import re
import configargparse

class Table():
	def __init__(self, table_loc):
		self.not_init = False
		self.table_loc = table_loc
		self._read()

	def _read(self):
		''' checks metatable is there. Removes /n and /t
			in metadata file then loads metadata.json as pandas dataframe
			todo: some heuristics to correct metadata.json in case of corruption
		'''
		if not os.path.isfile(self.table_loc):
			print('No such file at %s' %(self.table_loc) )
			self.not_init = True
			return
		try:
			with open(self.table_loc, 'r') as t:
				s = ''.join(t.read().split())
			self.data = pd.read_json(s)
		except IOError:
			print("Cannot read %s, file corrupted" %(self.table_loc))

	def _in_table(self, val, attr = 'File'):
		return val in self.data[attr].values

	def view(self, lookup_vals = [], attr = 'File'):
		''' provides an internal view of the metadata table, also fetches
			relevant entries as a pandas dataframe
		'''
		if not lookup_vals:
			print(self.data)
			return self.data
		# handling a string
		if type(lookup_vals) != list:
			lookup_vals = [lookup_vals]

		for f in lookup_vals:
			if not self._in_table(f, attr):
				print(str(f) + ' not in database.')
				continue
			print(self.data[ self.data[attr] == f ])
		return self.data[ self.data[attr].isin(lookup_vals) ]

	def erase(self, key):
		if key == 'delete this file':
			os.remove(self.table_loc)
			print('%s erased' %(self.table_loc))
		else:
			print('To erase this .json, type erase(\'delete this file\')')

class EditTable(Table):
	def __init__(self, mft, input_dir, table_loc, bitdepth, resolution):
		# call parent class's init: check file exists
		super().__init__(table_loc)
		self.mft = mft
		self.bitdepth = bitdepth
		self.resolution = resolution
		self.input_dir = input_dir
		self.attributes = ['File', 'Marker', 'Spot', 'BitDepth', 'Resolution', 'Hardware']

		if self.not_init:
		 	self.write(new_file = True)
		self.add_dir()
		self.write()

	def _add_entry(self, filepath ):
		''' generates file name and marker name, append new entry
			to self.data as a row if file name was not encountered before.
			Default schema is GE: [filename, marker name, spot, bitdepth, resolution, mft]
		'''
		fname = filepath.split('/')[-1].split('.')[0]
		if self._in_table(fname):
			self.repeats.append(fname)
			return

		if self.mft == 'GE':
			attr1 = fname.split('_')[0]
			attr2 = fname.split('_')[4]

			new_row = pd.DataFrame(
			[[fname, attr1, attr2, self.bitdepth, self.resolution, self.mft ] ], columns = self.attributes)
			self.data = self.data.append([new_row], ignore_index = True)

	def add_dir(self):
		''' traverse input_dir, ignoring files w/out .tif or .tiff extensions
			calls add_entry per file visited.
			If file name already found in table, report and ignore
		'''
		self.repeats = []
		for subdir, dirs, files in os.walk(self.input_dir):
			if not files:
				print( '%s empty, halting process.'   %(self.input_dir))
				return
			for f in files:
				if '.tif' in f:
					self._add_entry( self.input_dir + f )

		if self.repeats:
			print('%s entries in %s already exist in %s'
			%(len(self.repeats), self.input_dir, self.table_loc.split('/')[-1]) )

	def write(self, new_file = False):
		''' write to disk, if first time: write an empty table to file
		'''
		if new_file:
			print('Creating new table .json at %s' %(self.table_loc))
			self.data = pd.DataFrame(columns = self.attributes)
		self.data.to_json(self.table_loc)

if __name__ == "__main__":
	argp = configargparse.ArgParser()

	argp.add_argument('--meta_dir',
	help='dir of metadata .json file (if none exists, default is ./metadata)',
	default=os.getcwd()+'/metadata/')

	argp.add_argument('--mft', help = 'microscope manufacture (default = GE)', default = 'GE')
	argp.add_argument('--dir', help='directory path to import .tif files (enter nothing to enter view mode)', default = None)
	argp.add_argument('--y_resolution', help='Y resolution of imported files (default = 4000)', default= 4000)
	argp.add_argument('--x_resolution', help='X resolution of imported files (default = 7000)', default= 7000)
	argp.add_argument('--bitdepth', help='bit depth of imported files (default = 0.293)', default=0.293)
	args = argp.parse_args()

	if not args.dir:
		v = Table(table_loc = args.meta_dir + args.mft + '.json')
		v.view(lookup_vals = '004', attr = 'Spot')

	if args.dir:
		if not os.path.exists(args.meta_dir):
			print(args.meta_dir)
			os.makedirs(args.meta_dir)

		t = EditTable(mft = args.mft,
		input_dir = args.dir,
		table_loc = args.meta_dir + args.mft+'.json',
		bitdepth = args.bitdepth,
		resolution = [args.x_resolution, args.y_resolution])
		#t.erase('delete this file')

# todo:
# ask about hardware/software so the naming scheme can be consulted
# arbitrary data attributes (create new column to accomodate different attr)
