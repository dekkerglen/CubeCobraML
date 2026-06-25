"""The sparse MultiHotEmbedding path must equal the dense matmul it replaces.

This is the load-bearing correctness claim of the sparse rewrite: the encoder
consumes padded int32 index arrays during training but exposes a dense path for
inference, and the two must produce identical embeddings from the same weights.
"""

import numpy as np
import tensorflow as tf

from ccml.model import Encoder

VOCAB = 200


def _dense_and_idx(card_lists):
    """Build a dense multi-hot batch and the padded-index batch for the same
    card sets, so we can feed both representations through the encoder."""
    max_len = max((len(c) for c in card_lists), default=1)
    idx = np.full((len(card_lists), max_len), VOCAB, dtype=np.int32)
    dense = np.zeros((len(card_lists), VOCAB), dtype=np.float32)
    for i, cards in enumerate(card_lists):
        idx[i, : len(cards)] = cards
        if cards:
            dense[i, cards] = 1.0
    return dense, idx


def test_sparse_equals_dense():
    rng = np.random.default_rng(0)
    enc = Encoder("encoder", VOCAB)
    card_lists = [
        [],  # empty pool — the padding-only edge case
        [7],  # single card
        sorted(rng.choice(VOCAB, 5, replace=False).tolist()),
        sorted(rng.choice(VOCAB, 40, replace=False).tolist()),
    ]
    dense, idx = _dense_and_idx(card_lists)

    out_sparse = enc(tf.constant(idx)).numpy()
    out_dense = enc(tf.constant(dense)).numpy()

    assert np.abs(out_sparse - out_dense).max() < 1e-4


def test_ckpt_round_trip(tmp_path):
    """save_weights → load_weights reproduces the encoder exactly (the path the
    dashboard + export rely on to reload a trained checkpoint)."""
    enc = Encoder("encoder", VOCAB)
    dense, _ = _dense_and_idx([sorted(np.random.default_rng(1).choice(VOCAB, 30, replace=False).tolist())])
    before = enc(tf.constant(dense)).numpy()

    enc.save_weights(str(tmp_path / "encoder" / "model"))
    reloaded = Encoder("encoder", VOCAB)
    reloaded.load_weights(str(tmp_path / "encoder" / "model"))
    after = reloaded(tf.constant(dense)).numpy()

    assert np.abs(before - after).max() < 1e-6
