from model import CubeCobraMLSystem
import tensorflow as tf
from generator_disk import DataGenerator
import numpy as np
import json
import os
import os.path
import sys
# TopKCategoricalAccuracy
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# get params
params = sys.argv[1:]

epochs = int(params[0])
batch_size = int(params[1])
continue_training = params[2]
loss_weights = float(params[3])

data_dir = '../data/train/'
model_dir = './model/'

    
print('Creating Data Generator...\n')


generator = DataGenerator(
    '{}cubes/'.format(data_dir),
    '{}decks/'.format(data_dir),
    '{}picks/'.format(data_dir),
    '{}oracleFrequency.json'.format(data_dir),
    '{}correlations.json'.format(data_dir),
    num_batches=batch_size,
)

print('Creating Model...\n')

model = CubeCobraMLSystem(generator.num_cards)

model.compile(
    optimizer='adam',
    loss=['binary_crossentropy', 'binary_crossentropy', 'categorical_crossentropy', 'kullback_leibler_divergence'],
    loss_weights=[loss_weights, loss_weights, loss_weights, loss_weights],
    metrics={
        'output_1': 'accuracy',
        'output_2': 'accuracy',
        'output_3': [
            TopKCategoricalAccuracy(k=1, name="top1"),
            TopKCategoricalAccuracy(k=3, name="top3")
        ],
        'output_4': 'accuracy'
    }
)
# top_k_categorical_accuracy

if continue_training == 'true':
    print('Loading Model...\n')
    model.load_weights(model_dir)

print('Training Model...\n')

model.fit(
    generator,
    epochs=epochs
)

print('Saving Model to {}...\n'.format(model_dir))
model.save_weights(model_dir)

print('Done.\n')
