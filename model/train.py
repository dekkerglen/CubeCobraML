from model import CubeCobraMLSystem
import tensorflow as tf
from generator import DataGenerator
import numpy as np
import json
import os
import os.path
import random
import sys

# get params
params = sys.argv[1:]

epochs = int(params[0])
batch_size = int(params[1])
continue_training = params[2]

def reset_random_seeds(seed):
    # currently not used
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

data_dir = '../data/train/'
model_dir = './model/'

print('Loading Data...\n')

# open json files
with open(os.path.join(data_dir, 'cubes.json')) as f:
    cubes = json.load(f)
with open(os.path.join(data_dir, 'decks.json')) as f:
    decks = json.load(f)
# with open(os.path.join(data_dir, 'picks.json')) as f:
#     picks = json.load(f)
with open(os.path.join(data_dir, 'oracleFrequency.json')) as f:
    card_freqs = json.load(f)

print('Creating Data Generator...\n')


generator = DataGenerator(
    cubes,
    decks,
    # picks,
    card_freqs,
    batch_size=batch_size,
)

print('Creating Model...\n')

model = CubeCobraMLSystem(len(card_freqs))

model.compile(
    optimizer='adam',
    loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy'],
    loss_weights=[1.0,1.0,1.0],
    metrics='accuracy'
)

if continue_training == 'true':
    print('Loading Model...\n')
    model.load_weights(model_dir)

print('Training Model...\n')

model.fit(
    generator,
    epochs=epochs,
    verbose=1,
)

print('Saving Model to {}...\n'.format(model_dir))
model.save_weights(model_dir)
