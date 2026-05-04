import os

# Keep TF 2.16+ on the Keras-2 code path — we subclass Model and override
# save_weights/load_weights, which Keras 3 broke.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential


class Encoder(Model):
    def __init__(self, name):
        super().__init__()
        self.model = Sequential(
            [
                Dense(512, activation="relu", name=name + "_e1"),
                Dense(256, activation="relu", name=name + "_e3"),
                Dense(128, activation="linear", name=name + "_bottleneck"),
            ]
        )

    def call(self, x):
        return self.model(x)

    def save_weights(self, filename):
        print("Saving weights to " + filename)
        self.model.save(filename)

    def load_weights(self, filename):
        self.model = keras.models.load_model(filename)


class Decoder(Model):
    def __init__(self, name, output_dim, output_act):
        super().__init__()

        self.model = Sequential(
            [
                Dense(256, activation="relu", name=name + "_d1"),
                Dense(512, activation="relu", name=name + "_d3"),
                Dense(output_dim, activation=output_act, name=name + "_reconstruction"),
            ]
        )

    def call(self, x):
        return self.model(x)

    def save_weights(self, filename):
        print("Saving weights to " + filename)
        self.model.save(filename)

    def load_weights(self, filename):
        print("Loading weights from " + filename)
        self.model = keras.models.load_model(filename)


class CubeCobraMLSystem(Model):
    def __init__(
        self,
        num_cards,
        is_land_mask=None,
        use_cube_context: bool = True,
    ):
        super().__init__()
        self.use_cube_context = bool(use_cube_context)

        self.encoder = Encoder("encoder")
        self.cube_decoder = Decoder("recommend", num_cards, tf.nn.sigmoid)
        self.draft_decoder = Decoder("draft", num_cards, "linear")
        self.deck_build_decoder = Decoder("deck_build", num_cards, tf.nn.sigmoid)
        self.correlation_decoder = Decoder("correlate", num_cards, tf.nn.softmax)

        if is_land_mask is None:
            is_land_mask = np.zeros(num_cards, dtype=np.float32)
        # (1, num_cards) so it broadcasts cleanly over a (batch, num_cards) tensor.
        self.is_land_mask = tf.constant(
            np.asarray(is_land_mask, dtype=np.float32).reshape(1, -1)
        )

    # inputs is:
    #   [cubes,
    #    (deck_pools, deck_cube_ctx),
    #    (draft_pools, draft_packs, draft_cube_ctx),
    #    cards]
    # When use_cube_context=False, the deck_cube_ctx / draft_cube_ctx slots
    # still exist (the dataset always emits them), but we ignore their values
    # in the decoder branches and feed the pool embedding alone.
    def call(self, inputs, training=None):
        cube_pred = self.recommend(inputs[0], training=training)
        deck_pred = self.deck_build(inputs[1][0], inputs[1][1], training=training)
        draft_pred = self.draft(
            inputs[2][0], inputs[2][1], inputs[2][2], training=training
        )
        corr_pred = self.correlate(inputs[3], training=training)
        return [cube_pred, deck_pred, draft_pred, corr_pred]

    @tf.function
    def recommend(self, cubes, training=None):
        embedding = self.encoder(cubes, training=training)
        return self.cube_decoder(embedding, training=training)

    @tf.function
    def deck_build(self, pools, cube_context, training=None):
        pool_embedding = self.encoder(pools, training=training)
        if self.use_cube_context:
            cube_embedding = self.encoder(cube_context, training=training)
            combined = tf.concat([pool_embedding, cube_embedding], axis=-1)
        else:
            # Match the 256-dim shape decoders expect by duplicating the pool embedding.
            combined = tf.concat([pool_embedding, tf.zeros_like(pool_embedding)], axis=-1)
        return self.deck_build_decoder(combined, training=training)

    @tf.function
    def draft(self, pools, packs, cube_context, training=None):
        pool_embedding = self.encoder(pools, training=training)
        if self.use_cube_context:
            cube_embedding = self.encoder(cube_context, training=training)
            combined = tf.concat([pool_embedding, cube_embedding], axis=-1)
        else:
            combined = tf.concat([pool_embedding, tf.zeros_like(pool_embedding)], axis=-1)
        best_possible_picks = self.draft_decoder(combined, training=training)
        mask = 1e9 * (1 - packs)
        return tf.nn.softmax(best_possible_picks * packs - mask)

    @tf.function
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

    # ── lightweight checkpoint format ─────────────────────────────────────────
    # save_weights above writes a full SavedModel per sub-model (graph + sigs +
    # weights). That's necessary for the final tfjs conversion, but it makes
    # in-training checkpoints absurdly large. The methods below write only the
    # raw parameter tensors as .h5 — orders of magnitude smaller, fine for
    # rollback/inspection. They work because each sub-model's architecture is
    # deterministic (defined in __init__), so we can rebuild the graph and
    # load only the weights.
    _SUBMODELS = ("encoder", "cube_decoder", "draft_decoder", "deck_build_decoder", "correlation_decoder")

    def save_checkpoint(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        for name in self._SUBMODELS:
            getattr(self, name).model.save_weights(os.path.join(dirpath, f"{name}.weights.h5"))

    def load_checkpoint(self, dirpath):
        # Sub-models are lazily built; do a dummy forward pass to materialize
        # variables before loading. Caller is responsible for that — typically
        # by running one batch of data, or by calling .build() with the right
        # input shape if they know it.
        for name in self._SUBMODELS:
            getattr(self, name).model.load_weights(os.path.join(dirpath, f"{name}.weights.h5"))


def make_draft_loss(is_land_mask: np.ndarray, land_penalty_weight: float):
    """Build a custom loss for the draft head.

    On top of categorical crossentropy, adds a penalty when the bot puts
    probability mass on lands but the human's true pick was a non-land.
    The natural pressure: the model has to learn *when* picking a land is
    appropriate (because picking one when the human didn't always costs
    extra), without collapsing onto "never pick lands".
    """
    mask = tf.constant(np.asarray(is_land_mask, dtype=np.float32).reshape(1, -1))
    weight = float(land_penalty_weight)

    def draft_loss(y_true, y_pred):
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        # 1.0 if the human picked a land, else 0.0. Per example, shape (batch,).
        human_picked_land = tf.reduce_sum(y_true * mask, axis=-1)
        # Total probability the bot put on lands. Shape (batch,).
        bot_land_prob = tf.reduce_sum(y_pred * mask, axis=-1)
        penalty = bot_land_prob * (1.0 - human_picked_land)
        return cce + weight * penalty

    draft_loss.__name__ = "draft_loss"
    return draft_loss
