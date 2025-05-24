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

import json
import math
import os
from typing import Iterator, Literal

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from ccml.utils import humanize_number


def _count_records(files: list[str]) -> int:
    total = 0
    print("Counting records in", len(files), "files")
    for f in tqdm(files):
        # every JSON file is a list [...]  – count its length quickly
        with open(f) as fh:
            total += len(json.load(fh))
    return total


def _repeat_take(ds: tf.data.Dataset, n_batches: int, n_epoch_batches: int):
    """Repeat a dataset just enough times to guarantee that no data gets dropped"""
    if n_batches == 0:
        raise ValueError("batch_size is larger than this dataset")
    rep = math.ceil(n_epoch_batches / n_batches)
    return ds.repeat(rep).take(n_epoch_batches)


def _create_array_representation(indices: list[int], num_cards: int) -> np.ndarray:
    vec = np.zeros(num_cards, dtype=np.float32)
    vec[indices] = 1.0
    return vec


def _augment_cube(
    indices: list[int],
    num_cards: int,
    neg_sampler: np.ndarray,
    noise: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """create a vector representation of the cube alongside a noisy version for the recsys

    Args:
        indices: the indices of the cards in this particular cube
        num_cards: the number of total cards that exist on the date of training
        neg_sampler: probabilities such that when we fake add cards to the cube, we more often
            add popular cards. This is what prevents the model from learning to always add
            popular cards because we give it a bunch of examples of removing them.
        noise: the average percent of cards in the cube to have be fake
        noise_std: the standard deviation of the noise, so the recsys can't determine a fixed
            amount of cards to remove.

    Returns:
        x_cube: the cube with some cards removed and some added
        y_cube: a subset of the original cube list, where 25% of the cards removed from x_cube are also removed.
    """
    cube = _create_array_representation(indices, num_cards)

    includes = np.array(indices, dtype=np.int32)
    size = len(includes)

    noise_level = np.clip(np.random.normal(noise, noise_std), a_min=0.05, a_max=0.8)
    flip_amount = int(size * noise_level)
    if flip_amount == 0:
        return cube, cube

    flip_include = np.random.choice(includes, flip_amount, replace=False)

    excludes = np.setdiff1d(np.arange(num_cards, dtype=np.int32), includes, assume_unique=True)
    probs = neg_sampler[excludes] / neg_sampler[excludes].sum()
    flip_exclude = np.random.choice(excludes, flip_amount, p=probs, replace=False)

    x_cube, y_cube = cube.copy(), cube.copy()
    x_cube[flip_include] = 0.0  # cut
    x_cube[flip_exclude] = 1.0  # add
    # what this does is select a random 25 percent of the cards that were cut and also
    # remove them from the y_cube representation. I honestly don't remembery why I (@RyanSaxe)
    # did this originally. I think maybe there were some benefits of not always having the
    # the same exact y_cube representation to prevent overfitting. You could experiment
    # with removing this line and seeing if it helps or hurts the model.
    y_flip_include = np.random.choice(flip_include, flip_amount // 4, replace=False)
    y_cube[y_flip_include] = 0.0

    return x_cube, y_cube


# -----------------------------------------------Stream Functions-----------------------------------------------
# These functions are used to stream data from disk to avoid materializing large matrices in RAM.


def _cube_stream(
    files: list[str],
    num_cards: int,
    neg_sampler: np.ndarray,
    noise: float,
    noise_std: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    while True:
        np.random.shuffle(files)
        for fname in files:
            for indices in json.load(open(fname)):
                yield _augment_cube(indices, num_cards, neg_sampler, noise, noise_std)


def _deck_stream(files: list[str], num_cards: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    while True:
        np.random.shuffle(files)
        for fname in files:
            for rec in json.load(open(fname)):
                main = _create_array_representation(rec["mainboard"], num_cards)
                side = _create_array_representation(rec["sideboard"], num_cards)
                yield np.clip(main + side, 0.0, 1.0), main


def _pick_stream(files: list[str], num_cards: int) -> Iterator[tuple[tuple[np.ndarray, np.ndarray], np.ndarray]]:
    while True:
        np.random.shuffle(files)
        for fname in files:
            for rec in json.load(open(fname)):
                pool = _create_array_representation(rec["pool"], num_cards)
                pack = _create_array_representation(rec["pack"], num_cards)
                pick = _create_array_representation([rec["pick"]], num_cards)
                yield (pool, pack), pick


def _corr_stream(x_eye: np.ndarray, y_softmax: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rows = x_eye.shape[0]
    while True:
        for i in np.random.permutation(rows):
            yield x_eye[i], y_softmax[i]


def _prepare_stream(
    ds: tf.data.Dataset,
    record_count: int,
    batch_size: int,
    steps_per_epoch: int,
    total_epochs: int,
) -> tf.data.Dataset:
    """
    Repeat the *example* stream so that, after batching with
    drop_remainder=True, it contains exactly steps_per_epoch × total_epochs
    """
    needed_examples = steps_per_epoch * total_epochs * batch_size
    rep = math.ceil(needed_examples / record_count)
    return ds.repeat(rep).take(needed_examples).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def build_dataset(
    cubes_path: str,
    decks_path: str,
    picks_path: str,
    freq_path: str,
    correlations_path: str,
    batch_size: int,
    target_epochs: int,
    primary: Literal["cube", "deck", "pick", "card"],
    noise: float = 0.20,
    noise_std: float = 0.1,
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

    counts = dict(
        cube=_count_records(cube_files),
        deck=_count_records(deck_files),
        pick=_count_records(pick_files),
        card=len(json.load(open(freq_path))),
    )

    steps_per_epoch = math.ceil(counts[primary] / batch_size)

    # epochs needed so that EVERY dataset is covered at least once
    min_epochs = max(math.ceil(c / (steps_per_epoch * batch_size)) for c in counts.values())

    epochs = max(target_epochs, min_epochs)
    if epochs > target_epochs:
        print(f"[info] Asked for {target_epochs} epochs but {epochs} needed to see every dataset once.")

    print(f"Calculated epoch size: {steps_per_epoch} steps with batch_size = {batch_size}")
    print("Total examples in each dataset:")
    for k, v in counts.items():
        print(f"{k}: {humanize_number(v)}")

    cube_ds_raw = tf.data.Dataset.from_generator(
        lambda: _cube_stream(cube_files, num_cards, neg_sampler, noise, noise_std),
        output_signature=(
            tf.TensorSpec((num_cards,), tf.float32),
            tf.TensorSpec((num_cards,), tf.float32),
        ),
    )

    deck_ds_raw = tf.data.Dataset.from_generator(
        lambda: _deck_stream(deck_files, num_cards),
        output_signature=(
            tf.TensorSpec((num_cards,), tf.float32),
            tf.TensorSpec((num_cards,), tf.float32),
        ),
    )

    pick_ds_raw = tf.data.Dataset.from_generator(
        lambda: _pick_stream(pick_files, num_cards),
        output_signature=(
            (
                tf.TensorSpec((num_cards,), tf.float32),
                tf.TensorSpec((num_cards,), tf.float32),
            ),
            tf.TensorSpec((num_cards,), tf.float32),
        ),
    )

    corr_ds_raw = tf.data.Dataset.from_generator(
        lambda: _corr_stream(x_corr, y_corr),
        output_signature=(
            tf.TensorSpec((num_cards,), tf.float32),
            tf.TensorSpec((num_cards,), tf.float32),
        ),
    )

    cube_ds = _prepare_stream(cube_ds_raw, counts["cube"], batch_size, steps_per_epoch, epochs)
    deck_ds = _prepare_stream(deck_ds_raw, counts["deck"], batch_size, steps_per_epoch, epochs)
    pick_ds = _prepare_stream(pick_ds_raw, counts["pick"], batch_size, steps_per_epoch, epochs)
    corr_ds = _prepare_stream(corr_ds_raw, counts["card"], batch_size, steps_per_epoch, epochs)

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
    return dataset, counts["card"], steps_per_epoch, epochs
