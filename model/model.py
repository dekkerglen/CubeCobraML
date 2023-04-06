import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
# import tensorflowjs as tfjs
from tensorflow import keras

import os

class Encoder(Model):
    def __init__(self,name):
        super().__init__()
        self.model = Sequential([
            Dense(512, activation='relu', name=name + "_e1"),
            Dense(256, activation='relu', name=name + "_e3"),
            Dense(128, activation='relu', name=name + "_bottleneck")
        ])
    
    def call(self, x):
        return self.model(x)
    
    def save_weights(self, filename):
        print('Saving weights to ' + filename)
        self.model.save(filename)

    def load_weights(self, filename):
        self.model = keras.models.load_model(filename)

    # def save_json(self, filename):
    #     tfjs.converters.save_keras_model(self.model, filename)
    
class Decoder(Model):
    def __init__(self, name, output_dim, output_act):
        super().__init__()

        self.model = Sequential([
            Dense(256, activation='relu', name=name + "_d1"),
            Dense(512, activation='relu', name=name + "_d3"),
            Dense(output_dim, activation=output_act, name=name + "_reconstruction")
        ])
    
    def call(self, x):
        return self.model(x)
    
    def save_weights(self, filename):
        print('Saving weights to ' + filename)
        self.model.save(filename)

    def load_weights(self, filename):
        print('Loading weights from ' + filename)
        self.model = keras.models.load_model(filename)
        
    # def save_json(self, filename):
    #     tfjs.converters.save_keras_model(self.model, filename)
        
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
        return self.cube_decoder(embedding, training=training)
    
    def deck_build(self, pools, training=None):
        embedding = self.encoder(pools, training=training)
        return self.deck_build_decoder(embedding, training=training)

    def draft(self, packs, pools, training=None):
        embedding = self.encoder(pools, training=training)
        best_possible_picks = self.draft_decoder(embedding, training=training)
        return best_possible_picks * packs

    def save_weights(self, filename):
        self.encoder.save_weights(os.path.join(filename, "encoder", 'model'))
        self.cube_decoder.save_weights(os.path.join(filename, "cube_decoder", 'model'))
        self.draft_decoder.save_weights(os.path.join(filename, "draft_decoder"))
        self.deck_build_decoder.save_weights(os.path.join(filename, "deck_build_decoder", 'model'))

    def load_weights(self, filename):
        self.encoder.load_weights(os.path.join(filename, "encoder", 'model'))
        self.cube_decoder.load_weights(os.path.join(filename, "cube_decoder", 'model'))
        self.draft_decoder.load_weights(os.path.join(filename, "draft_decoder"))
        self.deck_build_decoder.load_weights(os.path.join(filename, "deck_build_decoder", 'model'))
        
