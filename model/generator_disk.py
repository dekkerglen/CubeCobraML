from tensorflow.keras.utils import Sequence
import numpy as np
import json
import os

class DataGenerator(Sequence):
    def __init__(
        self,
        cubes_path,
        decks_path,
        picks_path,
        freq_path,
        correlations_path,
        num_batches=128,
        noise=0.2,
        noise_std=0.1,
        corr_multiplier=32, # loop through correlations 32 times per epoch
        cube_multiplier=1,
    ):
        super().__init__()

        print('Loading Data...\n')
        with open(freq_path) as f:
            card_freqs = json.load(f)
        with open(correlations_path) as f:
            card_correlations = json.load(f)

        self.num_batches = num_batches
        self.noise = noise
        self.noise_std = noise_std
        self.num_cards = len(card_freqs)

        # card_correlations is stored in 1d format, but we need it in 2d format based on num_cards x num_cards
        self.card_correlations_y = np.array(card_correlations).astype(np.uint)
        self.card_correlations_y = self.card_correlations_y.reshape((self.num_cards, self.num_cards))

        # Normalize each distribution
        epsilon = 1
        self.card_correlations_y = self.card_correlations_y / (self.card_correlations_y.sum(axis=1, keepdims=True) + epsilon)

        self.card_correlations_x = np.identity(self.num_cards)

        # inverse of card frequency
        self.neg_sampler = np.array([1/(freq+1) for freq in card_freqs])

        self.cubes_path = cubes_path
        self.decks_path = decks_path
        self.picks_path = picks_path

        # list of files in cubes_path
        self.x_cubes_files = np.array(os.listdir(cubes_path))
        self.x_decks_files = np.array(os.listdir(decks_path))
        self.x_picks_files = np.array(os.listdir(picks_path))

        self.x_cubes = len(self.x_cubes_files)
        self.x_decks = len(self.x_decks_files)
        self.x_picks = len(self.x_picks_files)

        self.corr_indices = np.arange(self.num_cards)
        self.cube_indices = np.arange(self.x_cubes)
        self.deck_indices = np.arange(self.x_decks)
        self.pick_indices = np.arange(self.x_picks)

        self.corr_batch_size = len(self.corr_indices) // self.num_batches * corr_multiplier
        self.cube_batch_size = len(self.x_cubes_files) // self.num_batches * cube_multiplier
        self.deck_batch_size = len(self.x_decks_files) // self.num_batches
        self.pick_batch_size = len(self.x_picks_files) // self.num_batches

        print("Cube Batch Size: {}, for {} cubes".format(self.cube_batch_size, len(self.x_cubes_files)))
        print("Deck Batch Size: {}, for {} decks".format(self.deck_batch_size, len(self.x_picks_files)))
        print("Pick Batch Size: {}, for {} picks".format(self.pick_batch_size, len(self.x_picks_files)))
        print("Correlation Batch Size: {}, for {} correlations".format(self.corr_batch_size, len(self.corr_indices)))
        
        self.prep_next_epoch()

    def load_all(self, files, path):
        for file, i in zip(files, range(len(files))):
            print("Loading {} of {} from {}: {}".format(i, len(files), path, file))
            with open(os.path.join(path, file)) as f:
                yield json.load(f)
    
    def __len__(self):
        return self.num_batches

    def __getitem__(self, batch_number):
        # load those files
        cubes = []
        cube_index = (batch_number * self.cube_batch_size) % self.x_cubes
        for i in range(cube_index, cube_index + self.cube_batch_size):
            cubes.extend(next(self.load_all([self.x_cubes_files[i % self.x_cubes]], self.cubes_path)))
        
        decks = []
        deck_index = (batch_number * self.deck_batch_size) % self.x_decks
        for i in range(deck_index, deck_index + self.deck_batch_size):
            decks.extend(next(self.load_all([self.x_decks_files[i % self.x_decks]], self.decks_path)))

        picks = []
        pick_index = (batch_number * self.pick_batch_size) % self.x_picks
        for i in range(pick_index, pick_index + self.pick_batch_size):
            picks.extend(next(self.load_all([self.x_picks_files[i % self.x_picks]], self.picks_path)))

        X_cubes, y_cubes = self.generate_cubes(np.array(cubes), self.cube_batch_size)
        X_decks, y_decks = self.generate_decks(np.array(decks), self.deck_batch_size)
        X_picks, y_picks = self.generate_picks(np.array(picks), self.pick_batch_size)

        corr_start = (batch_number * self.corr_batch_size) % self.num_cards
        if corr_start + self.corr_batch_size < self.num_cards:
            corr_indeces = self.corr_indices[corr_start:corr_start+self.corr_batch_size]
        else:
            corr_indeces = np.concatenate((self.corr_indices[corr_start:], self.corr_indices[:self.corr_batch_size - (self.num_cards - corr_start)]))
        x_corr = self.card_correlations_x[corr_indeces]
        y_corr = self.card_correlations_y[corr_indeces]

        return [[X_cubes, X_decks, X_picks, x_corr], [y_cubes, y_decks, y_picks, y_corr]]
        
    def prep_next_epoch(self):
        # shuffle all indices
        np.random.shuffle(self.cube_indices)
        np.random.shuffle(self.deck_indices)
        np.random.shuffle(self.pick_indices)
        np.random.shuffle(self.corr_indices)

    def on_epoch_end(self):
        self.prep_next_epoch()

    def to_vector_encoding(self, indices, batch_size):
        vec = np.zeros((batch_size,self.num_cards))
        for i,index in enumerate(indices):
            vec[i,index] = 1
        return vec

    def generate_cubes(self, raw_cubes, batch_size):
        cubes = self.to_vector_encoding(raw_cubes, batch_size)

        cut_mask = np.zeros((batch_size,self.num_cards))
        add_mask = np.zeros((batch_size,self.num_cards))
        y_cut_mask = np.zeros((batch_size,self.num_cards))

        for i,cube in enumerate(cubes):
            includes = np.where(cube == 1)[0]
            excludes = np.where(cube == 0)[0]
            size = len(includes)
            noise = np.clip(
                np.random.normal(self.noise,self.noise_std),
                a_min = 0.05,
                a_max = 0.8,
            )
            flip_amount = int(size * noise)
            flip_include = np.random.choice(includes, flip_amount)

            sampler = self.neg_sampler[excludes]
            normalized = sampler / np.sum(sampler)

            flip_exclude = np.random.choice(excludes, flip_amount, p=normalized)
            y_flip_include = np.random.choice(flip_include, flip_amount//4)
            cut_mask[i,flip_include] = -1
            y_cut_mask[i,y_flip_include] = -1
            add_mask[i,flip_exclude] = 1

        x_cubes = cubes + cut_mask + add_mask
        y_cubes = cubes + y_cut_mask

        return [x_cubes, y_cubes]
    
    
    def encode_deck(self, mainboards, batch_size):
        vec = np.zeros((batch_size,self.num_cards))
        for i,mainboard in enumerate(mainboards):
            for index in mainboard:
                vec[i,index] = 1
        return vec
    
    def encode_pool(self, mainboards, sideboards, batch_size):       
        vec = np.zeros((batch_size, self.num_cards))
        for i,mainboard in enumerate(mainboards):
            for index in mainboard:
                vec[i,index] = 1
        for i,sideboard in enumerate(sideboards):
            for index in sideboard:
                vec[i,index] = 1
        return vec

    # decks have mainboard, sideboard, basics
    def generate_decks(self, decks, batch_size):        
        mainboards = np.array(list(map(lambda x: x['mainboard'], decks)))
        sideboards = np.array(list(map(lambda x: x['sideboard'], decks)))

        x = self.encode_pool(mainboards, sideboards, batch_size)
        y = self.encode_deck(mainboards, batch_size)

        return [x, y]
    
    def generate_picks(self, picks, batch_size):        
        pools = np.array(list(map(lambda x: x['pool'], picks)))
        packs = np.array(list(map(lambda x: x['pack'], picks)))
        pick = np.array(list(map(lambda x: [x['pick']], picks)))

        x_pool = self.to_vector_encoding(pools, batch_size)
        x_pack = self.to_vector_encoding(packs, batch_size)
        y_pick = self.to_vector_encoding(pick, batch_size)

        return [[x_pool, x_pack], y_pick]
