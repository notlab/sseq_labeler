import os
import random
import unicodedata
import string
import glob
from io import open

import torch
from torch import autograd

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

def decode_prediction(categories, pred):
    _, top_i = pred.data.topk(1) 
    category_i = top_i[0][0]
    return categories[category_i], category_i

def randelem(lst):
    return lst[random.randint(0, len(lst) - 1)]

def random_sample(categories, examples, one_hot_categories=False):
    category = randelem(categories)
    category_tensor = None
    sample = randelem(examples[category])
    sample_tensor = autograd.Variable(sample_to_tensor(sample))
    
    if one_hot_categories:
        idx = categories.index(category)
        category_tensor = torch.zeros(1, len(categories))
        category_tensor[0][idx] = 1
        category_tensor = autograd.Variable(category_tensor)
    else:
        category_tensor = autograd.Variable(torch.LongTensor([categories.index(category)]))
        
    return category, sample, category_tensor, sample_tensor

def sample_to_tensor(sample):
    tensor = torch.zeros(len(sample), 1, N_CHARS)
    for si, char in enumerate(sample):
        tensor[si][0][_getidx(char)] = 1
    return tensor

def target_encode(sample):
    indicies = [CHARS.find(line[idx]) for idx in range(1, len(idx))]
    indicies.append(N_CHARS - 1)
    return torch.LongTensor(indicies)

def inputs():
    categories = []
    examples = {}

    for filename in glob.glob(DATA_PATH):
        category = filename.split('/')[-1].split('.')[0]
        categories.append(category)
        lines = _read_lines(filename)
        examples[category] = lines

    return categories, examples
