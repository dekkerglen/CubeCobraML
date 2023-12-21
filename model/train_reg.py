from model import RegularizationSystem
import tensorflow as tf
from generator_reg import DataGenerator
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
    '{}oracleFrequency.json'.format(data_dir),
    '{}correlations.json'.format(data_dir),
    batch_size=batch_size,
)

print('Creating Model...\n')

model = RegularizationSystem(generator.num_cards)

model.compile(
    optimizer='adam',
    loss=['kullback_leibler_divergence'],
    loss_weights=[loss_weights],
    metrics={'output_1': 'accuracy' }
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
