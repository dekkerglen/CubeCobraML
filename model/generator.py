from tensorflow.keras.utils import Sequence
import numpy as np
import json
import os

class DataGenerator(Sequence):
    def __init__(
        self,
        cubes_path,
        decks_path,
        picks_pack,
        metadata_path,
        card_freqs,
        card_correlations,
        batch_size=128,
        noise=0.2,
        noise_std=0.1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.noise = noise
        self.noise_std = noise_std
        self.num_cards = len(card_freqs)
        
        self.card_correlations_y = np.array(card_correlations)
        self.card_correlations_x = np.identity(self.num_cards)

        # inverse of card frequency
        self.neg_sampler = np.array([1/(freq+1) for freq in card_freqs])

        self.cubes_path = cubes_path
        self.decks_path = decks_path
        self.picks_pack = picks_pack

        # load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        # number of files in cubes_path
        self.x_cubes = metadata['numCubes']
        self.x_decks = metadata['numDecks']
        self.x_picks = metadata['numPicks']

        # list of files in cubes_path
        self.x_cubes_files = np.array(os.listdir(cubes_path))
        self.x_decks_files = np.array(os.listdir(decks_path))
        self.x_picks_files = np.array(os.listdir(picks_pack))

        self.cube_indices = np.arange(len(self.x_cubes_files))
        self.deck_indices = np.arange(len(self.x_decks_files))
        self.pick_indices = np.arange(len(self.x_picks_files))
        self.corr_indices = np.arange(self.num_cards)

        self.cube_file = []
        self.deck_file = []
        self.pick_file = []

        self.samples_per_epoch = max(self.x_cubes, self.x_decks, self.x_picks)
        
        self.prep_next_epoch()
        
    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def __getitem__(self, batch_number):
        # load those files
        cubes = []
        while len(cubes) < self.batch_size:
            curr_cube_file = self.x_cubes_files[self.cube_indices[self.cube_files_index]]

            if self.cube_index == 0:
                with open(os.path.join(self.cubes_path, curr_cube_file)) as f:
                    self.cube_file = json.load(f)

            file_length = len(self.cube_file)
            current_length = len(cubes)
            needed = self.batch_size - current_length

            # if we need more than what's left in the file, add what's left and move to next file
            if needed > file_length + self.cube_index:
                # append rest of file
                cubes += self.cube_file[self.cube_index:]
                # move to next file
                self.cube_index = 0
                self.cube_files_index += 1
                self.cube_files_index %= len(self.x_cubes_files)
            else:
                cubes += self.cube_file[self.cube_index:self.cube_index+needed]
                self.cube_index += needed

        decks = []
        while len(decks) < self.batch_size:
            curr_deck_file = self.x_decks_files[self.deck_indices[self.deck_index]]

            if self.deck_index == 0:              
                with open(os.path.join(self.decks_path, curr_deck_file)) as f:
                    self.deck_file = json.load(f)

            file_length = len(self.deck_file)
            current_length = len(decks)
            needed = self.batch_size - current_length

            # if we need more than what's left in the file, add what's left and move to next file
            if needed > file_length + self.deck_index:
                # append rest of file
                decks += self.deck_file[self.deck_index:]
                # move to next file
                self.deck_index = 0
                self.deck_files_index += 1
                self.deck_files_index %= len(self.x_decks_files)
            else:
                decks += self.deck_file[self.deck_index:self.deck_index+needed]
                self.deck_index += needed

        picks = []
        while len(picks) < self.batch_size:
            curr_pick_file = self.x_picks_files[self.pick_indices[self.pick_index]]

            if self.pick_index == 0:
                with open(os.path.join(self.picks_pack, curr_pick_file)) as f:
                    self.pick_file = json.load(f)

            file_length = len(self.pick_file)
            current_length = len(picks)
            needed = self.batch_size - current_length

            # if we need more than what's left in the file, add what's left and move to next file
            if needed > file_length + self.pick_index:
                # append rest of file
                picks += self.pick_file[self.pick_index:]
                # move to next file
                self.pick_index = 0
                self.pick_files_index += 1
                self.pick_files_index %= len(self.x_picks_files)
            else:
                picks += self.pick_file[self.pick_index:self.pick_index+needed]
                self.pick_index += needed

        X_cubes, y_cubes = self.generate_cubes(np.array(cubes), self.batch_size)
        X_decks, y_decks = self.generate_decks(np.array(decks), self.batch_size)
        X_picks, y_picks = self.generate_picks(np.array(picks), self.batch_size)

        corr_indeces = self.corr_indices[batch_number * self.batch_size:(batch_number + 1) * self.batch_size]
        x_corr = self.card_correlations_x[corr_indeces]
        y_corr = self.card_correlations_y[corr_indeces]
        
        return [[X_cubes, X_decks, X_picks, x_corr], [y_cubes, y_decks, y_picks, y_corr]]
        
    def prep_next_epoch(self):
        # shuffle all indices
        np.random.shuffle(self.cube_indices)
        np.random.shuffle(self.deck_indices)
        np.random.shuffle(self.pick_indices)
        np.random.shuffle(self.corr_indices)

        self.cube_index = 0
        self.deck_index = 0
        self.pick_index = 0

        self.cube_files_index = 0
        self.deck_files_index = 0
        self.pick_files_index = 0

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

