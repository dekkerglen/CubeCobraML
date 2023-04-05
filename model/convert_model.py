from model import CubeCobraMLSystem
import tensorflow as tf
from generator import DataGenerator
import numpy as np
import json
import os
import os.path
from metrics import top_rated_percent

data_dir = '../data/test/'
model_dir = './model/'

with open(os.path.join(data_dir, 'oracleFrequency.json')) as f:
    card_freqs = json.load(f)
    
with open(os.path.join(data_dir, 'oracleFrequency.json')) as f:
    card_freqs = json.load(f)

print('Loading Model...\n')

model = CubeCobraMLSystem(len(card_freqs))

model.load_weights(model_dir)

print('Converting...\n')

exit()
