import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class Encoder(Model):
    def __init__(self,name):
        super().__init__()
        self.encoded_1 = Dense(512, activation='relu', name=name + "_e1")
        self.encoded_2 = Dense(256, activation='relu', name=name + "_e2")
        self.encoded_3 = Dense(128, activation='relu', name=name + "_e3")
        self.bottleneck = Dense(64, activation='relu', name=name + "_bottleneck")
    
    def call(self, x):
        encoded = self.encoded_1(x)
        encoded = self.encoded_2(encoded)
        encoded = self.encoded_3(encoded)
        return self.bottleneck(encoded)
    
class Decoder(Model):
    def __init__(self, name, output_dim, output_act):
        super().__init__()
        self.decoded_1 = Dense(128, activation='relu', name=name + "_d1")
        self.decoded_2 = Dense(256, activation='relu', name=name + "_d2")
        self.decoded_3 = Dense(512, activation='relu', name=name + "_d3")
        self.reconstruct = Dense(output_dim, activation=output_act, name=name + "_reconstruction")
    
    def call(self, x):
        decoded = self.decoded_1(x)
        decoded = self.decoded_2(decoded)
        decoded = self.decoded_3(decoded)
        return self.reconstruct(decoded)

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
