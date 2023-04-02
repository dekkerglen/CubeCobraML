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

num_gpus = len(tf.config.list_physical_devices('GPU'))

print("Num GPUs Available:", num_gpus)

if num_gpus <= 0:
    print('No GPUs found. Exiting...')
    sys.exit(1)

print('Loading Data...\n')

# open json files
with open(os.path.join(data_dir, 'cubes.json')) as f:
    cubes = json.load(f)
with open(os.path.join(data_dir, 'decks.json')) as f:
    decks = json.load(f)
with open(os.path.join(data_dir, 'picks.json')) as f:
    picks = json.load(f)
with open(os.path.join(data_dir, 'oracleFrequency.json')) as f:
    card_freqs = json.load(f)

print('Creating Data Generator...\n')


generator = DataGenerator(
    cubes,
    decks,
    picks,
    card_freqs,
)

print('Creating Model...\n')

model = CubeCobraMLSystem(len(card_freqs))

model.compile(
    optimizer='adam',
    loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy'],
    loss_weights=[1.0,1.0,1.0],
    metrics='accuracy'
)

print('Training Model...\n')

model.fit(
    generator,
    epochs=1,
    verbose=1,
)

dest = os.path.join(data_dir, 'model.h5')
print('Saving Model to {}...\n'.format(dest))
model.save(dest)
