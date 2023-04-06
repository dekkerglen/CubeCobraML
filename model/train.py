from model import CubeCobraMLSystem
import tensorflow as tf
from generator import DataGenerator
import numpy as np
import json
import os
import os.path
import sys
from metrics import top_rated_percent, relative_pick

# get params
params = sys.argv[1:]

epochs = int(params[0])
batch_size = int(params[1])
continue_training = params[2]
loss_weights = float(params[3])

data_dir = '../data/train/'
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
    
with open(os.path.join(data_dir, 'elos.json')) as f:
    elos = json.load(f)

print('Creating Data Generator...\n')


generator = DataGenerator(
    cubes,
    decks,
    picks,
    card_freqs,
    batch_size=batch_size,
)

print('Creating Model...\n')

model = CubeCobraMLSystem(len(card_freqs), elos)

model.compile(
    optimizer='adam',
    loss=['binary_crossentropy','binary_crossentropy','categorical_crossentropy'],
    loss_weights=loss_weights,
    metrics='accuracy'
)

if continue_training == 'true':
    print('Loading Model...\n')
    model.load_weights(model_dir)

print('Training Model...\n')

x, y = generator.__getitem__(0)

pred1, pred2, pred3 = model.predict(x)

print('Untrained cubes accuracy: ', top_rated_percent(y[0], pred1))
print('Untrained decks accuracy: ', top_rated_percent(y[1], pred2))
print('Untrained picks accuracy: ', relative_pick(x[2], y[2], pred3))

model.fit(
    generator,
    epochs=epochs
)

print('Saving Model to {}...\n'.format(model_dir))
model.save_weights(model_dir)

print('Done.\n')

pred1, pred2, pred3 = model.predict(x)

print('Trained cubes accuracy: ', top_rated_percent(y[0], pred1))
print('Trained decks accuracy: ', top_rated_percent(y[1], pred2))
print('Trained picks accuracy: ', relative_pick(x[2], y[2], pred3))
