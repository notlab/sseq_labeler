import os
import unicodedata
import string
import glob
from io import open

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_REGX = '../data/lstm/names/*.txt'
DATA_PATH = os.path.join(BASE_PATH, DATA_REGX)

CHARS = string.ascii_letters + " .,;'"
N_CHARS = len(CHARS)

def _unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c in CHARS)

def _read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [_unicode_to_ascii(line) for line in lines]

def _getidx(char):
    return CHARS.find(char)

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_CHARS)
    for li, char in enumerate(line):
        tensor[li][0][getidx(char)] = 1
    return tensor

def inputs():
    category_lines = {}
    all_categories = []

    for filename in glob.glob(DATA_PATH):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = _read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories
