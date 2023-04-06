from model import CubeCobraMLSystem
import tensorflow as tf
from generator import DataGenerator
import numpy as np
import json
import os
import os.path
from metrics import top_rated_percent, relative_pick

data_dir = '../data/test/'
model_dir = './model/'

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
    
with open(os.path.join(data_dir, 'oracleFrequency.json')) as f:
    card_freqs = json.load(f)

print('Creating Data Generator...\n')

generator = DataGenerator(
    cubes,
    decks,
    picks,
    card_freqs,
    batch_size=64,
)

print('Loading Model...\n')

model = CubeCobraMLSystem(len(card_freqs))

model.load_weights(model_dir)

print('Predicting...\n')

x, y = generator.__getitem__(0)

pred1, pred2, pred3 = model.predict(x)

print('Trained cubes accuracy: ', top_rated_percent(y[0], pred1))
print('Trained decks accuracy: ', top_rated_percent(y[1], pred2))
print('Trained picks accuracy: ', relative_pick(x[2], y[2], pred3))

exit()
