"""
Build a streaming `tf.data.Dataset` that yields exactly the same structure
the old `DataGenerator` produced – but without ever materialising more than
one micro-batch in RAM.

Every element is:
    ([x_cubes,
      x_decks,
      (pool_vec, pack_vec),   # picks input
      x_corr],
     [y_cubes,
      y_decks,
      y_pick_vec,
      y_corr])
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Tuple

import numpy as np
import tensorflow as tf


# --------------------------------------------------------------------- util
def _one_hot(indices: List[int], num_cards: int) -> np.ndarray:
    vec = np.zeros(num_cards, dtype=np.float32)
    vec[indices] = 1.0
    return vec


# ----------------------------- cube augmentation (old generate_cubes logic)
def _augment_cube(
    indices: List[int],
    num_cards: int,
    neg_sampler: np.ndarray,
    noise: float,
    noise_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cube = _one_hot(indices, num_cards)

    includes = np.array(indices, dtype=np.int32)
    size = len(includes)

    noise_level = np.clip(np.random.normal(noise, noise_std), a_min=0.05, a_max=0.8)
    flip_amount = int(size * noise_level)
    if flip_amount == 0:
        return cube, cube

    flip_include = np.random.choice(includes, flip_amount, replace=False)

    excludes = np.setdiff1d(
        np.arange(num_cards, dtype=np.int32), includes, assume_unique=True
    )
    probs = neg_sampler[excludes] / neg_sampler[excludes].sum()
    flip_exclude = np.random.choice(excludes, flip_amount, p=probs, replace=False)

    y_flip_include = np.random.choice(flip_include, flip_amount // 4, replace=False)

    x_cube, y_cube = cube.copy(), cube.copy()
    x_cube[flip_include] = 0.0  # cut
    x_cube[flip_exclude] = 1.0  # add
    y_cube[y_flip_include] = 0.0  # supervision

    return x_cube, y_cube


# ----------------------------------------------------------------- streams
def _cube_stream(
    files: List[str],
    num_cards: int,
    neg_sampler: np.ndarray,
    noise: float,
    noise_std: float,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    while True:
        np.random.shuffle(files)
        for fname in files:
            for indices in json.load(open(fname)):
                yield _augment_cube(indices, num_cards, neg_sampler, noise, noise_std)


def _deck_stream(
    files: List[str], num_cards: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    while True:
        np.random.shuffle(files)
        for fname in files:
            for rec in json.load(open(fname)):
                main = _one_hot(rec["mainboard"], num_cards)
                side = _one_hot(rec["sideboard"], num_cards)
                yield np.clip(main + side, 0.0, 1.0), main


def _pick_stream(
    files: List[str], num_cards: int
) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    while True:
        np.random.shuffle(files)
        for fname in files:
            for rec in json.load(open(fname)):
                pool = _one_hot(rec["pool"], num_cards)
                pack = _one_hot(rec["pack"], num_cards)
                pick = _one_hot([rec["pick"]], num_cards)
                yield (pool, pack), pick


def _corr_stream(
    x_eye: np.ndarray, y_softmax: np.ndarray
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    rows = x_eye.shape[0]
    while True:
        for i in np.random.permutation(rows):
            yield x_eye[i], y_softmax[i]


# --------------------------------------------------------------- factory
def build_dataset(
    cubes_path: str,
    decks_path: str,
    picks_path: str,
    freq_path: str,
    correlations_path: str,
    batch_size: int = 32,
    noise: float = 0.20,
    noise_std: float = 0.1,
    cube_multiplier: int = 16,
    corr_multiplier: int = 32,
):
    """Returns `(dataset, num_cards)` ready for `model.fit()`."""
    card_freqs = json.load(open(freq_path))
    num_cards = len(card_freqs)
    neg_sampler = np.array([1.0 / (f + 1) for f in card_freqs], dtype=np.float32)

    corr_raw = np.array(json.load(open(correlations_path)), dtype=np.float32)
    y_corr = corr_raw.reshape((num_cards, num_cards))
    y_corr /= y_corr.sum(axis=1, keepdims=True) + 1.0
    x_corr = np.eye(num_cards, dtype=np.float32)

    cube_files = [os.path.join(cubes_path, f) for f in os.listdir(cubes_path)]
    deck_files = [os.path.join(decks_path, f) for f in os.listdir(decks_path)]
    pick_files = [os.path.join(picks_path, f) for f in os.listdir(picks_path)]

    cube_ds = (
        tf.data.Dataset.from_generator(
            lambda: _cube_stream(cube_files, num_cards, neg_sampler, noise, noise_std),
            output_signature=(
                tf.TensorSpec((num_cards,), tf.float32),
                tf.TensorSpec((num_cards,), tf.float32),
            ),
        )
        .batch(batch_size, drop_remainder=True)
        .repeat(cube_multiplier)
    )

    deck_ds = tf.data.Dataset.from_generator(
        lambda: _deck_stream(deck_files, num_cards),
        output_signature=(
            tf.TensorSpec((num_cards,), tf.float32),
            tf.TensorSpec((num_cards,), tf.float32),
        ),
    ).batch(batch_size, drop_remainder=True)

    pick_ds = tf.data.Dataset.from_generator(
        lambda: _pick_stream(pick_files, num_cards),
        output_signature=(
            (
                tf.TensorSpec((num_cards,), tf.float32),
                tf.TensorSpec((num_cards,), tf.float32),
            ),
            tf.TensorSpec((num_cards,), tf.float32),
        ),
    ).batch(batch_size, drop_remainder=True)

    corr_ds = (
        tf.data.Dataset.from_generator(
            lambda: _corr_stream(x_corr, y_corr),
            output_signature=(
                tf.TensorSpec((num_cards,), tf.float32),
                tf.TensorSpec((num_cards,), tf.float32),
            ),
        )
        .batch(batch_size, drop_remainder=True)
        .repeat(corr_multiplier)
    )

    def _merge(cube, deck, pick, corr):
        x_cube, y_cube = cube
        x_deck, y_deck = deck
        (pool, pack), y_pick = pick
        x_corr, y_corr = corr
        return (
            (x_cube, x_deck, (pool, pack), x_corr),
            (y_cube, y_deck, y_pick, y_corr),
        )

    dataset = tf.data.Dataset.zip((cube_ds, deck_ds, pick_ds, corr_ds))
    dataset = dataset.map(_merge, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, num_cards
