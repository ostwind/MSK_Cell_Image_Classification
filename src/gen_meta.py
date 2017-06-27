import pandas as pd
import os
import json
import re
import configargparse

class table():
	def __init__(self, input_dir, table_loc, bitdepth, resolution):
		self.bitdepth = bitdepth
		self.resolution = resolution
		self.table_loc = table_loc
		self.input_dir = input_dir
		self.attributes = ['File', 'Marker', 'BitDepth', 'Resolution']

		if not os.path.isfile(table_loc):
			self.write(instantiate = True)
			return

		self.read()
		self.add_dir()
		self.write()

	def _in_table(self, val):
		return val in self.data['File'].values

	def add_entry(self, filepath ):
		''' generates file name and marker name, append new entry
			to self.data as a row if file name was not encountered before.
		'''
		fname = filepath.split('/')[-1]
		marker_name = fname.split('_')[0]
		if self._in_table(fname):
			self.repeats.append(fname)
			return

		new_row = pd.DataFrame(
		[[fname, marker_name, self.bitdepth, self.resolution ] ], columns = self.attributes)
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
					#print(self.input_dir + '/' + f)
					self.add_entry( self.input_dir + f )

		if self.repeats:
			for f in self.repeats:
				print(f)
			print('%s entries in %s already exist in %s'
			%(len(self.repeats), self.input_dir, self.table_loc.split('/')[-1]) )

	def write(self, instantiate = False):
		''' write to disk, if first time: write an empty table to file
		'''
		if instantiate:
			print('%s does not exist, creating new table at source code.' %(self.table_loc))
			self.data = pd.DataFrame(columns = self.attributes)
		self.data.to_json(self.table_loc)

	def read(self):
		''' removes /n and /t in metadata file then loads metadata.json
			as pandas dataframe
			todo: some heuristics to correct metadata.json in case of corruption
		'''
		try:
			with open(self.table_loc, 'r') as t:
				s = ''.join(t.read().split())
			print(s)
			self.data = pd.read_json(s)

		except IOError:
			print("Cannot read %s, file corrupted" %(self.table_loc))

if __name__ == "__main__":
	prev_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
	files_to_be_added = os.path.join(prev_dir, 'data/GE/')

	argp = configargparse.ArgParser()
	argp.add_argument('--dir', help='directory path to import .tif files', default = files_to_be_added)
	argp.add_argument('--view_mode', help='view current metadata (default = n)', choices=['y', 'n'], default='n')

	argp.add_argument(
	'--metadata_loc',
	help='location of metadata .json file (if none exists, new one will be created, default is current directory)',
	default=os.getcwd()+'/metadata.json')

	argp.add_argument('--y_resolution', help='Y resolution of imported files (default = 4000)', default= 4000)
	argp.add_argument('--x_resolution', help='X resolution of imported files (default = 7000)', default= 7000)
	argp.add_argument('--bitdepth', help='bit depth of imported files (default = 0.293)', default=0.293)
	args = argp.parse_args()

	table(input_dir = args.dir,
	table_loc = args.metadata_loc,
	bitdepth = args.bitdepth,
	resolution = [args.x_resolution, args.y_resolution])

# todo: view mode - file lookup
# 		arbitrary data attributes (create new column to accomodate different attr)
#		
