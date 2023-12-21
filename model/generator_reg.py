from tensorflow.keras.utils import Sequence
import numpy as np
import json
import os

class DataGenerator(Sequence):
    def __init__(
        self,
        freq_path,
        correlations_path,
        batch_size=128,
        noise=0.2,
        noise_std=0.1,
    ):
        super().__init__()

        print('Loading Data...\n')
        with open(freq_path) as f:
            card_freqs = json.load(f)
        with open(correlations_path) as f:
            card_correlations = json.load(f)

        self.batch_size = batch_size
        self.noise = noise
        self.noise_std = noise_std
        self.num_cards = len(card_freqs)

        # card_correlations is stored in 1d format, but we need it in 2d format based on num_cards x num_cards
        self.card_correlations_y = np.array(card_correlations).astype(np.uint)
        self.card_correlations_y = self.card_correlations_y.reshape((self.num_cards, self.num_cards))


        for i in range(100):
            print(i, self.card_correlations_y[i].sum())

        # Normalize each distribution
        epsilon = 1
        self.card_correlations_y = self.card_correlations_y / (self.card_correlations_y.sum(axis=1, keepdims=True) + epsilon)

        for i in range(100):
            print(i, self.card_correlations_y[i].sum())

        self.card_correlations_x = np.identity(self.num_cards)

        self.corr_indices = np.arange(self.num_cards)

        self.samples_per_epoch = len(self.corr_indices)
        
        self.prep_next_epoch()
    
    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def __getitem__(self, batch_number):
        corr_indeces = self.corr_indices[batch_number * self.batch_size:(batch_number + 1) * self.batch_size]
        x_corr = self.card_correlations_x[corr_indeces]
        y_corr = self.card_correlations_y[corr_indeces]

        return [x_corr, y_corr]
        
    def prep_next_epoch(self):
        np.random.shuffle(self.corr_indices)

    def on_epoch_end(self):
        self.prep_next_epoch()
