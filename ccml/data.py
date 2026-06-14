"""Streaming `tf.data.Dataset` pipelines for training and validation.

SPARSE PIPELINE: generators yield padded int32 card-index arrays (~1KB per
record) instead of dense 37k-float one-hots (~150KB). Records stay sparse
through repeat/shuffle/padded_batch — so the shuffle buffer holds indices,
not megabytes — and a single post-batch `_merge` map densifies the targets
(and the draft pack mask) with scatter ops. The encoder consumes the index
arrays directly (see `MultiHotEmbedding`); identical math to the dense path.

Every training element after `_merge` is:
    ([cube_idx,                 # padded int32 (B, L) — encoder gathers
      deck_pool_idx,
      (pick_pool_idx, pack_mask_dense),  # pack mask stays dense for softmax masking
      corr_idx],
     [y_cube, y_deck, y_pick, y_corr],   # dense float (B, V), built post-batch
     (ones, ones, w_pick, ones))         # per-output sample weights; only picks weighted

Validation elements omit the sample weights (val is an unweighted
measurement) and are fully deterministic — no augmentation, no shuffling.
The stable val set produced by `scripts/process_data.js` is small enough
(target ≤100k picks) to evaluate exhaustively every epoch, so val metrics
are stable across epochs and trustworthy as a deployment gate.
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
        with open(f) as fh:
            total += len(json.load(fh))
    return total


def _create_array_representation(indices: list[int], num_cards: int) -> np.ndarray:
    vec = np.zeros(num_cards, dtype=np.float32)
    vec[indices] = 1.0
    return vec


def _idx(indices) -> np.ndarray:
    """Unique sorted int32 index array — the sparse equivalent of
    `_create_array_representation` (whose `vec[indices] = 1.0` dedupes
    duplicate card copies implicitly).

    Drops negative indices: process_data.js emits -1 for cube cards missing
    from the oracle vocabulary. The legacy dense path silently aliased those
    onto the LAST card via numpy negative indexing — a bug; scatter_nd
    rejects them loudly, so we filter instead."""
    arr = np.unique(np.asarray(indices, dtype=np.int32))
    return arr[arr >= 0]


def _augment_cube(
    indices: list[int],
    num_cards: int,
    pop_sampler: np.ndarray,
    rare_sampler: np.ndarray,
    noise: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create index-array representations of the cube alongside a noisy
    version for the recsys. Same augmentation semantics as the dense-vector
    version, now returning sparse index arrays.

    Args:
        indices: the indices of the cards in this particular cube
        num_cards: the number of total cards that exist on the date of training
        pop_sampler: per-card weights ∝ frequency. Used to pick which cards to *add* to x_cube
            as fake additions — biased toward popular cards so the model learns it's good to
            remove popular cards from a cube.
        rare_sampler: per-card weights ∝ 1 / frequency. Used to pick which cards to *cut* from
            x_cube — biased toward less-popular cards so the model learns it's good to add
            rare/synergistic cards to a cube.
        noise: the average percent of cards in the cube to have be fake
        noise_std: the standard deviation of the noise

    Returns:
        x_cube: index array — the cube with some cards removed and some added
        y_cube: index array — the original cube minus 25% of the cards removed from x_cube
    """
    includes = _idx(indices)
    size = len(includes)

    noise_level = np.clip(np.random.normal(noise, noise_std), a_min=0.05, a_max=0.8)
    flip_amount = int(size * noise_level)
    if flip_amount == 0:
        return includes, includes

    include_probs = rare_sampler[includes] / rare_sampler[includes].sum()
    flip_include = np.random.choice(includes, flip_amount, p=include_probs, replace=False)

    excludes = np.setdiff1d(np.arange(num_cards, dtype=np.int32), includes, assume_unique=True)
    exclude_probs = pop_sampler[excludes] / pop_sampler[excludes].sum()
    flip_exclude = np.random.choice(excludes, flip_amount, p=exclude_probs, replace=False)

    x_cube = np.union1d(np.setdiff1d(includes, flip_include), flip_exclude).astype(np.int32)
    y_flip_include = np.random.choice(flip_include, flip_amount // 4, replace=False)
    y_cube = np.setdiff1d(includes, y_flip_include).astype(np.int32)

    return x_cube, y_cube


# -----------------------------------------------Stream Functions-----------------------------------------------
# These functions are used to stream data from disk to avoid materializing large matrices in RAM.


def _cube_stream(
    files: list[str],
    num_cards: int,
    pop_sampler: np.ndarray,
    rare_sampler: np.ndarray,
    noise: float,
    noise_std: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    while True:
        np.random.shuffle(files)
        for fname in files:
            for indices in json.load(open(fname)):
                yield _augment_cube(indices, num_cards, pop_sampler, rare_sampler, noise, noise_std)


def _deck_stream(
    files: list[str],
    num_cards: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream decks for the deck-build head: (pool_idx, mainboard_idx).

    The deck record's `cube_cards` field stays on disk for the dashboard but
    is no longer read — the v3 deck head takes the pool embedding only.
    """
    del num_cards
    while True:
        np.random.shuffle(files)
        for fname in files:
            for rec in json.load(open(fname)):
                main = _idx(rec["mainboard"])
                pool = np.union1d(main, _idx(rec["sideboard"])).astype(np.int32)
                yield pool, main


def _make_pick_target(picked_idx: int, pack_idxs: list[int], num_cards: int, label_smoothing: float) -> np.ndarray:
    """Pack-targeted label smoothing (dense reference implementation).

    The training pipeline now builds this target with scatter ops inside the
    post-batch `_merge` map (see `_densify_pick_targets`); this function is
    kept as the executable spec the TF version is verified against.

    Hard one-hot target gives non-picked pack cards target=0, which lets the
    model collapse their logits arbitrarily low — even for cards humans pick
    often. Distributing `label_smoothing` mass uniformly across the pack keeps
    every pack card with target ≥ ε/K, anchoring their logits.

      target[picked]      = (1 - ε) + ε/K
      target[other pack]  = ε/K
      target[not in pack] = 0

    With label_smoothing=0 this collapses to the original one-hot.
    """
    target = np.zeros(num_cards, dtype=np.float32)
    if label_smoothing > 0 and len(pack_idxs) > 1:
        per = label_smoothing / len(pack_idxs)
        for c in pack_idxs:
            target[c] = per
        target[picked_idx] += 1.0 - label_smoothing
    else:
        target[picked_idx] = 1.0
    return target


def _pack_position_weight(pack_size: int, pack_start: int, power: float) -> float:
    """Sample weight by position within the pack.

    Raw fraction f = (K−1)/(pack_start−1) is 1.0 at the pack's first pick and
    0.0 at the forced last pick; w = f^power shapes the decay. power=0.5
    (sqrt) keeps mid-pack picks valuable — linear (power=1.0) would claim
    "halfway through the pack matters half as much as P1P1," which is wrong.
    Normalizing by the pack's own starting size makes weights comparable
    across drafts with different pack sizes.
    """
    if pack_start <= 1:
        return 0.0
    f = (pack_size - 1) / (pack_start - 1)
    if f <= 0.0:
        return 0.0
    return float(f**power)


def _pick_file_records(
    fname: str,
    weighting: bool,
    weight_power: float,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.int32, np.int32, np.float32]]:
    """Yield (pool_idx, pack_idx, picked, pack_len_raw, weight) for one pick
    file, in file order. `pack_len_raw` is the pack size INCLUDING duplicate
    copies — the label-smoothing denominator and weight detector use it,
    while `pack_idx` is unique'd for the scatter ops.

    Pack-boundary detection relies on the file's per-session sequential
    order (process_data.js preserves it): a new session starts when the pool
    is empty; a new pack within a session starts when the pack grows.
    """
    prev_pack_size = 0
    pack_start = 1
    for rec in json.load(open(fname)):
        pack_size = len(rec["pack"])
        if len(rec["pool"]) == 0 or pack_size > prev_pack_size:
            pack_start = pack_size
        prev_pack_size = pack_size
        w = _pack_position_weight(pack_size, pack_start, weight_power) if weighting else 1.0

        yield (_idx(rec["pool"]), _idx(rec["pack"]), np.int32(rec["pick"]), np.int32(pack_size), np.float32(w))


def _pick_stream(
    pick_files: list[str],
    weighting: bool = True,
    weight_power: float = 0.5,
    interleave_files: int = 16,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.int32, np.int32, np.float32]]:
    """Stream picks round-robin across `interleave_files` open files.

    Consecutive records in one pick file come from the same draft, so a
    sequential read feeds the shuffle buffer long correlated runs.
    Interleaving spreads the buffer's contents across ~interleave_files
    files, decorrelating batches. The pack-boundary state needed for sample
    weights lives inside each `_pick_file_records` iterator, so interleaving
    cannot corrupt per-draft detection.
    """

    def _open(fname: str):
        return _pick_file_records(fname, weighting, weight_power)

    while True:
        np.random.shuffle(pick_files)
        queue = list(pick_files)
        open_iters = [_open(queue.pop()) for _ in range(min(interleave_files, len(queue)))]
        while open_iters:
            i = 0
            while i < len(open_iters):
                record = next(open_iters[i], None)
                if record is None:
                    if queue:
                        open_iters[i] = _open(queue.pop())
                    else:
                        open_iters.pop(i)
                    continue
                yield record
                i += 1


def _corr_stream(y_softmax: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ([card_idx], dense_target_row). The input is a single-card
    index array (the sparse identity row); the target row is inherently
    dense real-valued, so it stays dense — the stream is already permuted
    at the source, so it skips the tf.data shuffle buffer entirely."""
    rows = y_softmax.shape[0]
    while True:
        for i in np.random.permutation(rows):
            yield np.array([i], dtype=np.int32), y_softmax[i]


def _prepare_stream(
    ds: tf.data.Dataset,
    record_count: int,
    batch_size: int,
    steps_per_epoch: int,
    total_epochs: int,
    padding_values,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Repeat + shuffle + padded-batch a training stream so the model sees
    exactly `steps_per_epoch × total_epochs` batches.

    Records are sparse index arrays (~1KB), so the shuffle buffer
    (SHUFFLE_BUFFER, default 16384 — cheap to raise to 100k+) holds indices
    instead of dense vectors. The shuffle breaks draft-level correlation:
    consecutive records in a pick file come from the same draft (~45 picks).
    `shuffle=False` is for streams already permuted at the source (corr).
    """
    needed_examples = steps_per_epoch * total_epochs * batch_size
    rep = math.ceil(needed_examples / record_count)
    ds = ds.repeat(rep).take(needed_examples)
    if shuffle:
        shuffle_buffer = int(os.environ.get("SHUFFLE_BUFFER", "16384"))
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    return ds.padded_batch(batch_size, padding_values=padding_values, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def _scatter_multi_hot(idx: tf.Tensor, num_cards: int) -> tf.Tensor:
    """Padded int32 (B, L) index array → dense float32 (B, num_cards)
    multi-hot. Padding entries (== num_cards) are dropped. Index arrays are
    unique per row, so scatter never double-counts."""
    batch = tf.shape(idx, out_type=tf.int64)[0]
    coords = tf.where(tf.not_equal(idx, num_cards))
    rows = coords[:, 0]
    cards = tf.cast(tf.gather_nd(idx, coords), tf.int64)
    return tf.scatter_nd(
        tf.stack([rows, cards], axis=1),
        tf.ones_like(rows, dtype=tf.float32),
        (batch, num_cards),
    )


def _densify_pick_targets(
    pack_idx: tf.Tensor,
    picked: tf.Tensor,
    pack_len: tf.Tensor,
    num_cards: int,
    label_smoothing: float,
) -> tf.Tensor:
    """TF reimplementation of `_make_pick_target`, vectorized over the batch:
      pack cards get ε/K each (K = raw pack size incl. duplicate copies),
      the picked card gets +(1−ε); single-card packs collapse to one-hot.
    Verified equal to the python reference in the pre-launch smokes."""
    batch = tf.shape(pack_idx, out_type=tf.int64)[0]
    pack_len_f = tf.cast(pack_len, tf.float32)
    if label_smoothing > 0.0:
        smooth_per_row = tf.where(pack_len > 1, label_smoothing / pack_len_f, tf.zeros_like(pack_len_f))
        picked_add = tf.where(
            pack_len > 1,
            tf.fill(tf.shape(pack_len_f), 1.0 - label_smoothing),
            tf.ones_like(pack_len_f),
        )
    else:
        smooth_per_row = tf.zeros_like(pack_len_f)
        picked_add = tf.ones_like(pack_len_f)

    coords = tf.where(tf.not_equal(pack_idx, num_cards))
    rows = coords[:, 0]
    cards = tf.cast(tf.gather_nd(pack_idx, coords), tf.int64)
    target = tf.scatter_nd(
        tf.stack([rows, cards], axis=1),
        tf.gather(smooth_per_row, rows),
        (batch, num_cards),
    )
    pick_rows = tf.range(batch)
    target += tf.scatter_nd(
        tf.stack([pick_rows, tf.cast(picked, tf.int64)], axis=1),
        picked_add,
        (batch, num_cards),
    )
    return target


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
    respect_target_epochs: bool = False,
    pick_label_smoothing: float = 0.0,
    pack_weighting: bool = True,
    pack_weight_power: float = 0.5,
    interleave_files: int = 16,
):
    """Build the training tf.data.Dataset."""
    card_freqs = json.load(open(freq_path))
    num_cards = len(card_freqs)
    # rare_sampler ∝ 1/freq — for cube augmentation (cut popular, add rare)
    rare_sampler = np.array([1.0 / (f + 1) for f in card_freqs], dtype=np.float32)
    pop_sampler = np.array([f + 1.0 for f in card_freqs], dtype=np.float32)

    corr_raw = np.array(json.load(open(correlations_path)), dtype=np.float32)
    y_corr = corr_raw.reshape((num_cards, num_cards))
    y_corr /= y_corr.sum(axis=1, keepdims=True) + 1.0

    cube_files = [os.path.join(cubes_path, f) for f in os.listdir(cubes_path)]
    deck_files = [os.path.join(decks_path, f) for f in os.listdir(decks_path)]
    pick_files = [os.path.join(picks_path, f) for f in os.listdir(picks_path)]

    counts = dict(
        cube=_count_records(cube_files),
        deck=_count_records(deck_files),
        pick=_count_records(pick_files),
        card=num_cards,
    )

    steps_per_epoch = math.ceil(counts[primary] / batch_size)

    min_epochs = max(math.ceil(c / (steps_per_epoch * batch_size)) for c in counts.values())

    if respect_target_epochs:
        epochs = target_epochs
        if target_epochs < min_epochs:
            print(
                f"[quick] Using {target_epochs} epochs as requested; the largest stream "
                f"would normally need {min_epochs} epochs to be seen once."
            )
    else:
        epochs = max(target_epochs, min_epochs)
        if epochs > target_epochs:
            print(f"[info] Asked for {target_epochs} epochs but {epochs} needed to see every dataset once.")

    print(f"Calculated epoch size: {steps_per_epoch} steps with batch_size = {batch_size}")
    print("Total examples in each dataset:")
    for k, v in counts.items():
        print(f"{k}: {humanize_number(v)}")

    idx_spec = tf.TensorSpec((None,), tf.int32)
    pad_id = tf.constant(num_cards, tf.int32)

    cube_ds_raw = tf.data.Dataset.from_generator(
        lambda: _cube_stream(cube_files, num_cards, pop_sampler, rare_sampler, noise, noise_std),
        output_signature=(idx_spec, idx_spec),  # x_cube_idx, y_cube_idx
    )

    deck_ds_raw = tf.data.Dataset.from_generator(
        lambda: _deck_stream(deck_files, num_cards),
        output_signature=(idx_spec, idx_spec),  # pool_idx, mainboard_idx
    )

    pick_ds_raw = tf.data.Dataset.from_generator(
        lambda: _pick_stream(
            pick_files, weighting=pack_weighting, weight_power=pack_weight_power, interleave_files=interleave_files
        ),
        output_signature=(
            idx_spec,  # pool_idx
            idx_spec,  # pack_idx (unique)
            tf.TensorSpec((), tf.int32),  # picked card idx
            tf.TensorSpec((), tf.int32),  # raw pack size
            tf.TensorSpec((), tf.float32),  # sample weight
        ),
    )

    corr_ds_raw = tf.data.Dataset.from_generator(
        lambda: _corr_stream(y_corr),
        output_signature=(
            tf.TensorSpec((1,), tf.int32),  # single card idx
            tf.TensorSpec((num_cards,), tf.float32),  # dense target row
        ),
    )

    cube_ds = _prepare_stream(
        cube_ds_raw, counts["cube"], batch_size, steps_per_epoch, epochs, padding_values=(pad_id, pad_id)
    )
    deck_ds = _prepare_stream(
        deck_ds_raw, counts["deck"], batch_size, steps_per_epoch, epochs, padding_values=(pad_id, pad_id)
    )
    pick_ds = _prepare_stream(
        pick_ds_raw,
        counts["pick"],
        batch_size,
        steps_per_epoch,
        epochs,
        padding_values=(
            pad_id,
            pad_id,
            tf.constant(0, tf.int32),
            tf.constant(0, tf.int32),
            tf.constant(0.0, tf.float32),
        ),
    )
    # Corr is permuted at the source and its dense target rows are large —
    # skip the shuffle buffer entirely.
    corr_ds = _prepare_stream(
        corr_ds_raw,
        counts["card"],
        batch_size,
        steps_per_epoch,
        epochs,
        padding_values=(pad_id, tf.constant(0.0, tf.float32)),
        shuffle=False,
    )

    def _merge(cube, deck, pick, corr):
        x_cube_idx, y_cube_idx = cube
        deck_pool_idx, y_deck_idx = deck
        pool_idx, pack_idx, picked, pack_len, w_pick = pick
        corr_idx, y_corr_ = corr
        pack_mask = _scatter_multi_hot(pack_idx, num_cards)
        y_cube = _scatter_multi_hot(y_cube_idx, num_cards)
        y_deck = _scatter_multi_hot(y_deck_idx, num_cards)
        y_pick = _densify_pick_targets(pack_idx, picked, pack_len, num_cards, pick_label_smoothing)
        ones = tf.ones_like(w_pick)
        return (
            (x_cube_idx, deck_pool_idx, (pool_idx, pack_mask), corr_idx),
            (y_cube, y_deck, y_pick, y_corr_),
            (ones, ones, w_pick, ones),
        )

    dataset = tf.data.Dataset.zip((cube_ds, deck_ds, pick_ds, corr_ds))
    dataset = dataset.map(_merge, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, counts["card"], steps_per_epoch, epochs


def _val_pick_records(
    pick_files: list[str],
) -> Iterator[tuple[np.ndarray, np.ndarray, np.int32, np.int32]]:
    """Yield all val picks in deterministic order:
    (pool_idx, pack_idx, picked, pack_len_raw). The smoothed target is built
    post-batch by `_densify_pick_targets` with the same `label_smoothing` as
    training, so train/val loss values stay directly comparable. No sample
    weights — val is an unweighted measurement.
    """
    for fname in sorted(pick_files):
        for rec in json.load(open(fname)):
            yield (_idx(rec["pool"]), _idx(rec["pack"]), np.int32(rec["pick"]), np.int32(len(rec["pack"])))


def _val_deck_records(files: list[str]):
    """Yield val decks in deterministic order: (pool_idx, mainboard_idx)."""
    for fname in sorted(files):
        for rec in json.load(open(fname)):
            main = _idx(rec["mainboard"])
            pool = np.union1d(main, _idx(rec["sideboard"])).astype(np.int32)
            yield pool, main


def _val_cube_records(files: list[str]):
    """Yield val cubes in deterministic order. NO augmentation — x = y.

    Validation cube task is essentially "identity" without noise; the val
    cube_bce loss isn't a deployment metric (val_bomb_agreement is), so we
    keep this trivial for stability and speed.
    """
    for fname in sorted(files):
        for indices in json.load(open(fname)):
            idx = _idx(indices)
            yield idx, idx


def build_validation_dataset(
    cubes_path: str,
    decks_path: str,
    picks_path: str,
    train_freq_path: str,
    train_correlations_path: str,
    batch_size: int,
    pick_label_smoothing: float = 0.0,
):
    """Build a deterministic, fully-cached validation dataset.

    The new process_data.js produces a small (≤100k picks), stable val set
    that fits in memory. We evaluate the FULL val set every epoch with no
    shuffling or augmentation — so val metrics reflect actual model changes,
    not data-sampling noise.

    Returns (val_ds, val_steps_per_epoch, val_counts). Returns (None, 0, {})
    if any test split directory is empty.
    """
    if not (
        os.path.isdir(cubes_path)
        and os.listdir(cubes_path)
        and os.path.isdir(decks_path)
        and os.listdir(decks_path)
        and os.path.isdir(picks_path)
        and os.listdir(picks_path)
    ):
        return None, 0, {}

    card_freqs = json.load(open(train_freq_path))
    num_cards = len(card_freqs)

    corr_raw = np.array(json.load(open(train_correlations_path)), dtype=np.float32)
    y_corr = corr_raw.reshape((num_cards, num_cards))
    y_corr /= y_corr.sum(axis=1, keepdims=True) + 1.0

    cube_files = [os.path.join(cubes_path, f) for f in os.listdir(cubes_path)]
    deck_files = [os.path.join(decks_path, f) for f in os.listdir(decks_path)]
    pick_files = [os.path.join(picks_path, f) for f in os.listdir(picks_path)]

    val_counts = dict(
        cube=_count_records(cube_files),
        deck=_count_records(deck_files),
        pick=_count_records(pick_files),
        card=num_cards,
    )

    # Pick stream drives val_steps_per_epoch — it's the main signal we care
    # about (bomb_agreement is on the pick head). Others are repeated/cycled
    # to match the pick volume.
    pick_count = val_counts["pick"]
    val_steps_per_epoch = max(1, math.ceil(pick_count / batch_size))

    print(f"Validation: full pass over {pick_count} picks ({val_steps_per_epoch} steps at batch_size={batch_size})")
    print("Held-out examples per stream:")
    for k, v in val_counts.items():
        print(f"  {k}: {humanize_number(v)}")

    # Build deterministic streams (NO shuffling, NO augmentation, NO noise).
    idx_spec = tf.TensorSpec((None,), tf.int32)
    pad_id = tf.constant(num_cards, tf.int32)

    cube_ds_raw = tf.data.Dataset.from_generator(
        lambda: _val_cube_records(cube_files),
        output_signature=(idx_spec, idx_spec),
    )
    deck_ds_raw = tf.data.Dataset.from_generator(
        lambda: _val_deck_records(deck_files),
        output_signature=(idx_spec, idx_spec),
    )
    pick_ds_raw = tf.data.Dataset.from_generator(
        lambda: _val_pick_records(pick_files),
        output_signature=(
            idx_spec,  # pool_idx
            idx_spec,  # pack_idx (unique)
            tf.TensorSpec((), tf.int32),  # picked card idx
            tf.TensorSpec((), tf.int32),  # raw pack size
        ),
    )
    corr_ds_raw = tf.data.Dataset.from_generator(
        lambda: _corr_stream(y_corr),
        output_signature=(
            tf.TensorSpec((1,), tf.int32),
            tf.TensorSpec((num_cards,), tf.float32),
        ),
    )

    # Each non-pick stream is .repeat()'d so val_steps_per_epoch batches can
    # be drawn from each. The pick stream determines coverage; others cycle.
    def _val_prepare(ds, padding_values):
        return (
            ds.repeat()
            .padded_batch(batch_size, padding_values=padding_values, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    cube_ds = _val_prepare(cube_ds_raw, (pad_id, pad_id))
    deck_ds = _val_prepare(deck_ds_raw, (pad_id, pad_id))
    pick_ds = _val_prepare(pick_ds_raw, (pad_id, pad_id, tf.constant(0, tf.int32), tf.constant(0, tf.int32)))
    corr_ds = _val_prepare(corr_ds_raw, (pad_id, tf.constant(0.0, tf.float32)))

    def _merge(cube, deck, pick, corr):
        x_cube_idx, y_cube_idx = cube
        deck_pool_idx, y_deck_idx = deck
        pool_idx, pack_idx, picked, pack_len = pick
        corr_idx, y_corr_ = corr
        pack_mask = _scatter_multi_hot(pack_idx, num_cards)
        y_cube = _scatter_multi_hot(y_cube_idx, num_cards)
        y_deck = _scatter_multi_hot(y_deck_idx, num_cards)
        y_pick = _densify_pick_targets(pack_idx, picked, pack_len, num_cards, pick_label_smoothing)
        return (
            (x_cube_idx, deck_pool_idx, (pool_idx, pack_mask), corr_idx),
            (y_cube, y_deck, y_pick, y_corr_),
        )

    val_ds = tf.data.Dataset.zip((cube_ds, deck_ds, pick_ds, corr_ds))
    val_ds = val_ds.map(_merge, num_parallel_calls=tf.data.AUTOTUNE)
    # NO .cache(): the densified targets are still ~hundreds of MB per batch
    # at large batch sizes; caching the full val pass OOM-killed runs before.
    # Re-reading + re-densifying each epoch costs a few minutes but is safe.
    val_ds = val_ds.take(val_steps_per_epoch).prefetch(tf.data.AUTOTUNE)
    return val_ds, val_steps_per_epoch, val_counts
