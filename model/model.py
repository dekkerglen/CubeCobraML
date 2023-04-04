import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential

class Encoder(Model):
    def __init__(self,name):
        super().__init__()
        self.model = Sequential([
            Dense(512, activation='relu', name=name + "_e1"),
            Dense(256, activation='relu', name=name + "_e2"),
            Dense(128, activation='relu', name=name + "_e3"),
            Dense(64, activation='relu', name=name + "_bottleneck")
        ])
    
    def call(self, x):
        return self.model(x)
    
    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)
    
class Decoder(Model):
    def __init__(self, name, output_dim, output_act):
        super().__init__()

        self.model = Sequential([
            Dense(128, activation='relu', name=name + "_d1"),
            Dense(256, activation='relu', name=name + "_d2"),
            Dense(512, activation='relu', name=name + "_d3"),
            Dense(output_dim, activation=output_act, name=name + "_reconstruction")
        ])
    
    def call(self, x):
        return self.model(x)
    
    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)
class CubeCobraMLSystem(Model):
    def __init__(self, num_cards):
        super().__init__()
        self.encoder = Encoder('encoder')
        self.cube_decoder = Decoder('recommend', num_cards, tf.nn.sigmoid)
        self.draft_decoder = Decoder('draft', num_cards, tf.nn.sigmoid)
        self.deck_build_decoder = Decoder('deck_build', num_cards, tf.nn.sigmoid)

    # inputs is [[cubes], [decks], [[packs], [pools]]]
    def call(self, inputs, training=None):
        return [
            self.recommend(inputs[0], training=training),
            self.deck_build(inputs[1], training=training),
            self.draft(inputs[2][0], inputs[2][1], training=training)
        ]

    def recommend(self, cubes, training=None):
        embedding = self.encoder(cubes, training=training)
        card_rankings = self.cube_decoder(embedding, training=training)
        viable_recommendations = tf.math.abs(cubes - 1)
        return card_rankings * viable_recommendations
    
    def deck_build(self, pools, training=None):
        embedding = self.encoder(pools, training=training)
        card_rankings = self.deck_build_decoder(embedding, training=training)
        return card_rankings * pools

    def draft(self, packs, pools, training=None):
        embedding = self.encoder(pools, training=training)
        best_possible_picks = self.draft_decoder(embedding, training=training)
        best_pick_from_pack = best_possible_picks * packs
        return best_pick_from_pack

    def save_weights(self, filename):
        self.encoder.save_weights(filename + "_encoder")
        self.cube_decoder.save_weights(filename + "_cube_decoder")
        self.draft_decoder.save_weights(filename + "_draft_decoder")
        self.deck_build_decoder.save_weights(filename + "_deck_build_decoder")

    def load_weights(self, filename):
        self.encoder.load_weights(filename + "_encoder")
        self.cube_decoder.load_weights(filename + "_cube_decoder")
        self.draft_decoder.load_weights(filename + "_draft_decoder")
        self.deck_build_decoder.load_weights(filename + "_deck_build_decoder")
        