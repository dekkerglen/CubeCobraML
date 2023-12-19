import unittest
import numpy as np
import os
import json
from generator import DataGenerator  # Assuming the class name is Generator

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.cubes_path = "../data/train/cubes"
        self.decks_path = "../data/train/decks"
        self.picks_pack = "../data/train/picks"
        self.metadata_path = "../data/train/metadata.json"
        
        
        with open("../data/train/oracleFrequency.json") as f:
            self.card_freqs = json.load(f)
        with open("../data/train/correlations.json") as f:
            self.card_correlations = json.load(f)

        self.batch_size = 128
        self.noise = 0.2
        self.noise_std = 0.1

        self.generator = DataGenerator(
            self.cubes_path,
            self.decks_path,
            self.picks_pack,
            self.metadata_path,
            self.card_freqs,
            self.card_correlations,
            self.batch_size,
            self.noise,
            self.noise_std
        )

    def test_init(self):
        self.assertEqual(self.generator.batch_size, self.batch_size)
        self.assertEqual(self.generator.noise, self.noise)
        self.assertEqual(self.generator.noise_std, self.noise_std)
        self.assertEqual(self.generator.num_cards, len(self.card_freqs))
        self.assertEqual(self.generator.cubes_path, self.cubes_path)
        self.assertEqual(self.generator.decks_path, self.decks_path)
        self.assertEqual(self.generator.picks_pack, self.picks_pack)

        with open(self.metadata_path) as f:
            metadata = json.load(f)

        self.assertEqual(self.generator.x_cubes, metadata['numCubes'])
        self.assertEqual(self.generator.x_decks, metadata['numDecks'])
        self.assertEqual(self.generator.x_picks, metadata['numPicks'])

        self.assertEqual(len(self.generator.x_cubes_files), len(os.listdir(self.cubes_path)))
        self.assertEqual(len(self.generator.x_decks_files), len(os.listdir(self.decks_path)))
        self.assertEqual(len(self.generator.x_picks_files), len(os.listdir(self.picks_pack)))

        self.assertEqual(len(self.generator.cube_indices), len(os.listdir(self.cubes_path)))
        self.assertEqual(len(self.generator.deck_indices), len(os.listdir(self.decks_path)))
        self.assertEqual(len(self.generator.pick_indices), len(os.listdir(self.picks_pack)))
        self.assertEqual(len(self.generator.corr_indices), len(self.card_freqs))

        self.assertEqual(self.generator.samples_per_epoch, max(metadata['numCubes'], metadata['numDecks'], metadata['numPicks']))
    
    def test_getitem(self):
        self.generator.prep_next_epoch()
        result = self.generator.__getitem__(0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        for item in result:
            self.assertEqual(len(item), self.batch_size)

if __name__ == "__main__":
    unittest.main()
