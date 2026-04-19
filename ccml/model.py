import os

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
    def __init__(self, num_cards):
        super().__init__()
        self.encoder = Encoder("encoder")
        self.cube_decoder = Decoder("recommend", num_cards, tf.nn.sigmoid)
        # Draft and deck-build decoders receive twice-wide input:
        # concat(encoded_pool, encoded_cube_context) → 256-dim
        # Draft decoder gets +2 for landCount/nonlandCount → 258-dim
        self.draft_decoder = Decoder("draft", num_cards, "linear")
        self.deck_build_decoder = Decoder("deck_build", num_cards, tf.nn.sigmoid)
        self.correlation_decoder = Decoder("correlate", num_cards, tf.nn.softmax)

    # inputs is:
    #   [cubes,
    #    (deck_pools, deck_cube_ctx),
    #    (draft_pools, draft_packs, draft_cube_ctx, draft_counts),
    #    cards]
    @tf.function
    def call(self, inputs, training=None):
        return [
            self.recommend(inputs[0], training=training),
            self.deck_build(inputs[1][0], inputs[1][1], training=training),
            self.draft(inputs[2][0], inputs[2][1], inputs[2][2], inputs[2][3], training=training),
            self.correlate(inputs[3], training=training),
        ]

    @tf.function
    def recommend(self, cubes, training=None):
        embedding = self.encoder(cubes, training=training)
        return self.cube_decoder(embedding, training=training)

    @tf.function
    def deck_build(self, pools, cube_context, training=None):
        pool_embedding = self.encoder(pools, training=training)
        cube_embedding = self.encoder(cube_context, training=training)
        combined = tf.concat([pool_embedding, cube_embedding], axis=-1)
        return self.deck_build_decoder(combined, training=training)

    @tf.function
    def draft(self, pools, packs, cube_context, draft_counts, training=None):
        pool_embedding = self.encoder(pools, training=training)
        cube_embedding = self.encoder(cube_context, training=training)
        combined = tf.concat([pool_embedding, cube_embedding, draft_counts], axis=-1)
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
