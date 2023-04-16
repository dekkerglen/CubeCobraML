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
    
with open(os.path.join(data_dir, 'elos.json')) as f:
    elos = json.load(f)
    

print('Creating Data Generator...\n')

generator = DataGenerator(
    cubes,
    decks,
    picks,
    card_freqs,
    batch_size=256,
)

print('Loading Model...\n')

model = CubeCobraMLSystem(len(card_freqs), elos)

model.load_weights(model_dir)

print('Predicting...\n')

accuracies = [0, 0, 0, 0]

batch_count = generator.__len__()

for i in range(batch_count):
    print('Batch: ', i + 1, '/', batch_count)
    x, y = generator.__getitem__(i)
    pred1, pred2, pred3 = model.predict(x)
    accuracies[0] += top_rated_percent(y[0], pred1)
    accuracies[1] += top_rated_percent(y[1], pred2)
    accuracies[2] += relative_pick(y[2], pred3, 1)
    accuracies[3] += relative_pick(y[2], pred3, 3)


print('Trained cubes accuracy: ', accuracies[0] / batch_count)
print('Trained decks accuracy: ', accuracies[1] / batch_count)
print('Trained picks top1: ', accuracies[2] / batch_count)
print('Trained picks top3: ', accuracies[3] / batch_count)

exit()
