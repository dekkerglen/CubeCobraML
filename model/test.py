from model import CubeCobraMLSystem
import tensorflow as tf
from generator import DataGenerator
import numpy as np
import json
import os
import os.path
import random
import sys
import pdb

def reset_random_seeds(seed):
    # currently not used
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

data_dir = '../data/'

print('Loading Data...\n')

# open json files
with open(os.path.join(data_dir, 'cubes.json')) as f:
    cubes = json.load(f)
print('cubes loaded')
with open(os.path.join(data_dir, 'decks.json')) as f:
    decks = json.load(f)
print('decks loaded')
with open(os.path.join(data_dir, 'picks.json')) as f:
    picks = json.load(f)
print('picks loaded')
with open(os.path.join(data_dir, 'oracleFrequency.json')) as f:
    card_freqs = json.load(f)
print('card_freqs loaded')

print('Creating Data Generator...\n')


generator = DataGenerator(
    cubes,
    decks,
    picks,
    card_freqs,
)

x,y = generator.generate_picks(np.array([1]))
x1, x2 = x

def decode_vector(vector):
    return [i for i, x in enumerate(vector) if x == 1]

print(decode_vector(x1[0]), decode_vector(x2[0]),  decode_vector(y[0]))
