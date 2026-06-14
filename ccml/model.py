import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2 as l2_reg


def _keras_path(stem: str) -> str:
    return stem + ".keras"


def _load_into(target: Sequential, path: str) -> None:
    # Keras 3 forbids reassigning sub-models on a built parent layer, so we load
    # into a temporary model and copy the weights — building `target` first if
    # it hasn't been built yet so it has matching variables to receive them.
    source = keras.models.load_model(path)
    if not target.built:
        target.build(source.input_shape)
    target.set_weights(source.get_weights())


@keras.utils.register_keras_serializable(package="ccml")
class MultiHotEmbedding(keras.layers.Layer):
    """Sparse-equivalent of `relu(multi_hot @ W + b)` computed by gather-sum.

    For a multi-hot input, the dense matmul is literally the sum of the
    kernel rows for the cards present — >99.8% of the FLOPs multiply zeros.
    This layer takes padded int32 card-index arrays instead and gathers
    those rows directly. The table has `vocab + 1` rows: the last row is
    the padding sentinel, masked out of both the sum and the gradient.

    `dense_call` computes the identical result from a dense multi-hot via
    matmul against the same weights — used by the inference APIs so the
    dashboard/TFJS export keep their dense signatures.
    """

    def __init__(self, vocab: int, units: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab = int(vocab)
        self.units = int(units)

    def build(self, input_shape):
        del input_shape
        self.table = self.add_weight(
            name="table",
            shape=(self.vocab + 1, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, indices):
        # Keras 3 autocasts inputs to the layer's compute dtype (float32);
        # cast back — index values are exact well below float32's 2^24 limit.
        indices = tf.cast(indices, tf.int32)
        mask = tf.cast(tf.not_equal(indices, self.vocab), tf.float32)
        emb = tf.gather(self.table, indices)
        summed = tf.reduce_sum(emb * mask[:, :, None], axis=1)
        return tf.nn.relu(summed + self.bias)

    def compute_output_shape(self, input_shape):
        # Input is int32 indices (B, L); output is float32 (B, units). Keras 3
        # can't trace this dtype change symbolically without the hint.
        return (input_shape[0], self.units)

    def dense_call(self, x):
        return tf.nn.relu(tf.matmul(x, self.table[: self.vocab]) + self.bias)

    def get_config(self):
        return {**super().get_config(), "vocab": self.vocab, "units": self.units}


class Encoder(Model):
    def __init__(self, name, num_cards):
        super().__init__()
        self.num_cards = int(num_cards)
        self.model = Sequential(
            [
                MultiHotEmbedding(num_cards, 512, name=name + "_e1"),
                Dense(256, activation="relu", name=name + "_e3"),
                Dense(128, activation="linear", name=name + "_bottleneck"),
            ]
        )

    def call(self, x):
        # Integer input = padded card-index arrays (training path);
        # float input = dense multi-hot (inference APIs, dashboard, export).
        # Same weights, identical math either way.
        if x.dtype.is_integer:
            return self.model(x)
        # The dense path calls the embedding layer's method directly, which
        # bypasses Keras's build-on-call — make sure the weights exist first.
        self._build_layers()
        embed, hidden, bottleneck = self.model.layers
        return bottleneck(hidden(embed.dense_call(x)))

    def _build_layers(self):
        if not self.model.built:
            self.model(tf.zeros((1, 1), dtype=tf.int32))

    def save_weights(self, filename):
        path = _keras_path(filename)
        print("Saving weights to " + path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._build_layers()
        self.model.save(path)

    def load_weights(self, filename):
        """Loads both v3 ckpts (MultiHotEmbedding table) and legacy ckpts
        (Dense first layer): a legacy `(vocab, 512)` kernel is copied into
        `table[:vocab]` with a zeroed sentinel row — bit-identical behavior
        on the dense path, so old runs stay comparable in the dashboard."""
        source = keras.models.load_model(_keras_path(filename))
        weights = source.get_weights()
        self._build_layers()
        expected_rows = self.num_cards + 1
        if weights and weights[0].shape[0] == expected_rows - 1:
            kernel = weights[0]
            sentinel = np.zeros((1, kernel.shape[1]), dtype=kernel.dtype)
            weights[0] = np.vstack([kernel, sentinel])
        self.model.set_weights(weights)


class Decoder(Model):
    def __init__(
        self,
        name,
        output_dim,
        output_act,
        output_bias_init: np.ndarray | None = None,
        output_kernel_l2: float = 0.0,
    ):
        super().__init__()

        # Output layer kwargs: optionally seed bias with a per-card prior
        # (e.g. log marginal pick rate) and/or apply L2 to the per-card output
        # kernel — both target rare-card miscalibration without changing shapes.
        out_kwargs: dict = {}
        if output_bias_init is not None:
            bias_vals = np.asarray(output_bias_init, dtype=np.float32)
            if bias_vals.shape != (output_dim,):
                raise ValueError(f"output_bias_init shape {bias_vals.shape} != ({output_dim},)")
            out_kwargs["bias_initializer"] = Constant(bias_vals)
        if output_kernel_l2 > 0.0:
            out_kwargs["kernel_regularizer"] = l2_reg(output_kernel_l2)

        self.model = Sequential(
            [
                Dense(256, activation="relu", name=name + "_d1"),
                Dense(512, activation="relu", name=name + "_d3"),
                Dense(
                    output_dim,
                    activation=output_act,
                    name=name + "_reconstruction",
                    **out_kwargs,
                ),
            ]
        )

    def call(self, x):
        return self.model(x)

    def save_weights(self, filename):
        path = _keras_path(filename)
        print("Saving weights to " + path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load_weights(self, filename):
        path = _keras_path(filename)
        print("Loading weights from " + path)
        _load_into(self.model, path)


class CubeCobraMLSystem(Model):
    def __init__(
        self,
        num_cards,
        draft_bias_init: np.ndarray | None = None,
        draft_output_l2: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder("encoder", num_cards)
        self.cube_decoder = Decoder("recommend", num_cards, "sigmoid")
        # Draft head: pool through the SHARED encoder, decoded to per-card
        # logits. draft_bias_init seeds per-card logit bias from the log
        # marginal pick rate (helps prevent rare-card overpick).
        self.draft_decoder = Decoder(
            "draft",
            num_cards,
            "linear",
            output_bias_init=draft_bias_init,
            output_kernel_l2=draft_output_l2,
        )
        # Deck head: pool embedding only (128-dim input), same as prod.
        self.deck_build_decoder = Decoder("deck_build", num_cards, "sigmoid")
        self.correlation_decoder = Decoder("correlate", num_cards, "softmax")

    # inputs is:
    #   [cubes,                        # for cube_decoder (recsys task)
    #    deck_pools,                   # for deck_build_decoder
    #    (draft_pools, draft_packs),   # for draft head — packs stay DENSE
    #    cards]                        # for correlation_decoder
    #
    # The training pipeline feeds padded int32 card-index arrays for cubes,
    # deck_pools, draft_pools, and cards (the encoder dispatches on dtype);
    # draft_packs is a dense float mask because the softmax masking needs it.
    # Inference callers keep passing dense multi-hot floats everywhere.
    def call(self, inputs, training=None):
        cube_pred = self.recommend(inputs[0], training=training)
        deck_pred = self.deck_build(inputs[1], training=training)
        draft_pred = self.draft(inputs[2][0], inputs[2][1], training=training)
        corr_pred = self.correlate(inputs[3], training=training)
        return [cube_pred, deck_pred, draft_pred, corr_pred]

    def recommend(self, cubes, training=None):
        embedding = self.encoder(cubes, training=training)
        return self.cube_decoder(embedding, training=training)

    def deck_build(self, pools, training=None):
        pool_embedding = self.encoder(pools, training=training)
        return self.deck_build_decoder(pool_embedding, training=training)

    def draft(self, pools, packs, training=None):
        """Public inference API: returns pack-masked softmax over cards."""
        pool_embedding = self.encoder(pools, training=training)
        logits = self.draft_decoder(pool_embedding, training=training)
        mask = 1e9 * (1 - packs)
        return tf.nn.softmax(logits * packs - mask)

    def correlate(self, inputs, training=None):
        embedding = self.encoder(inputs, training=training)
        return self.correlation_decoder(embedding, training=training)

    def save_weights(self, filename):
        self.encoder.save_weights(os.path.join(filename, "encoder", "model"))
        self.cube_decoder.save_weights(os.path.join(filename, "cube_decoder", "model"))
        self.draft_decoder.save_weights(os.path.join(filename, "draft_decoder", "model"))
        self.deck_build_decoder.save_weights(os.path.join(filename, "deck_build_decoder", "model"))
        self.correlation_decoder.save_weights(os.path.join(filename, "correlation_decoder", "model"))

    def load_weights(self, filename):
        self.encoder.load_weights(os.path.join(filename, "encoder", "model"))
        self.cube_decoder.load_weights(os.path.join(filename, "cube_decoder", "model"))
        self.draft_decoder.load_weights(os.path.join(filename, "draft_decoder", "model"))
        self.deck_build_decoder.load_weights(os.path.join(filename, "deck_build_decoder", "model"))
        self.correlation_decoder.load_weights(os.path.join(filename, "correlation_decoder", "model"))
