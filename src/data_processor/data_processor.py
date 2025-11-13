import os

from parser import parse

DATA_FOLDER = '../../data'
RAW_FOLDER = 'raw'
PARSED_FOLDER = 'parsed'

parse(os.path.join(DATA_FOLDER, RAW_FOLDER),  os.path.join(DATA_FOLDER, PARSED_FOLDER))


