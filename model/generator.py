from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):
    def __init__(
        self,
        cubes,
        decks,
        picks,
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

        
        self.card_correlations_y = card_correlations
        self.card_correlations_x = np.identity(self.num_cards)

        # inverse of card frequency
        self.neg_sampler = np.array([1/(freq+1) for freq in card_freqs])
        print (self.neg_sampler)

        self.x_cubes = np.array(cubes)
        self.x_decks = np.array(decks, dtype=object)
        self.x_picks = np.array(picks, dtype=object)
        
        self.cube_indices = np.arange(len(self.x_cubes))
        self.deck_indices = np.arange(len(self.x_decks))
        self.pick_indices = np.arange(len(self.x_picks))
        self.corr_indices = np.arange(self.num_cards)
        
        self.shuffle_indeces()
        
    def __len__(self):
        return min(len(self.x_cubes), len(self.x_decks), len(self.x_picks)) // self.batch_size

    def __getitem__(self, batch_number):
        X_cubes, y_cubes = self.generate_cubes(self.cube_indices[batch_number * self.batch_size:(batch_number + 1) * self.batch_size], self.batch_size)
        X_decks, y_decks = self.generate_decks(self.deck_indices[batch_number * self.batch_size:(batch_number + 1) * self.batch_size], self.batch_size)
        X_picks, y_picks = self.generate_picks(self.pick_indices[batch_number * self.batch_size:(batch_number + 1) * self.batch_size], self.batch_size)

        corr_indeces = self.corr_indices[batch_number * self.batch_size:(batch_number + 1) * self.batch_size]
        x_corr = self.card_correlations_x[corr_indeces]
        y_corr = self.card_correlations_y[corr_indeces]
        
        return [[X_cubes, X_decks, X_picks, x_corr], [y_cubes, y_decks, y_picks, y_corr]]
        
    def shuffle_indeces(self):
        np.random.shuffle(self.cube_indices)
        np.random.shuffle(self.deck_indices)
        np.random.shuffle(self.pick_indices)
        np.random.shuffle(self.corr_indices)

    def on_epoch_end(self):
        self.shuffle_indeces()

    def to_vector_encoding(self, indices, batch_size):
        vec = np.zeros((batch_size,self.num_cards))
        for i,index in enumerate(indices):
            vec[i,index] = 1
        return vec

    def generate_cubes(self, main_indices, batch_size):
        cubes = self.to_vector_encoding(self.x_cubes[main_indices], batch_size)

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
    def generate_decks(self, main_indices, batch_size):        
        decks = self.x_decks[main_indices]

        mainboards = np.array(list(map(lambda x: x['mainboard'], decks)))
        sideboards = np.array(list(map(lambda x: x['sideboard'], decks)))

        x = self.encode_pool(mainboards, sideboards, batch_size)
        y = self.encode_deck(mainboards, batch_size)

        return [x, y]
    
    def generate_picks(self, main_indices, batch_size):
        picks = self.x_picks[main_indices]
        
        pools = np.array(list(map(lambda x: x['pool'], picks)))
        packs = np.array(list(map(lambda x: x['pack'], picks)))
        pick = np.array(list(map(lambda x: [x['pick']], picks)))

        x_pool = self.to_vector_encoding(pools, batch_size)
        x_pack = self.to_vector_encoding(packs, batch_size)
        y_pick = self.to_vector_encoding(pick, batch_size)

        return [[x_pool, x_pack], y_pick]

