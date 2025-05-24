"""
Usage:  python train.py <epochs> <steps_per_epoch> <continue_train:true|false>
The second CLI argument now means *steps per epoch* (was “num_batches”).
"""

import os
import sys

from tensorflow.keras.metrics import TopKCategoricalAccuracy

from ccml.data import build_dataset
from ccml.metrics import top_rated_percent
from ccml.model import CubeCobraMLSystem
from ccml.utils import DATA_DIR, MODEL_DIR

DATA_DIR = DATA_DIR / "train"
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
continue_flag = sys.argv[3].lower() == "true"
primary_stream = sys.argv[4].lower()

dataset, num_cards, steps_per_epoch, epochs_final = build_dataset(
    cubes_path=os.path.join(DATA_DIR, "cubes"),
    decks_path=os.path.join(DATA_DIR, "decks"),
    picks_path=os.path.join(DATA_DIR, "picks"),
    freq_path=os.path.join(DATA_DIR, "oracleFrequency.json"),
    correlations_path=os.path.join(DATA_DIR, "correlations.json"),
    batch_size=batch_size,
    target_epochs=epochs,
    primary=primary_stream,
)


print("Creating / loading model …")
model = CubeCobraMLSystem(num_cards)

losses = [
    "binary_crossentropy",
    "binary_crossentropy",
    "categorical_crossentropy",
    "kullback_leibler_divergence",
]

model.compile(
    optimizer="adam",
    loss=losses,
    loss_weights=[1.0] * 4,
    metrics={
        "output_1": top_rated_percent,
        "output_2": top_rated_percent,
        "output_3": [
            TopKCategoricalAccuracy(k=1, name="top1"),
            TopKCategoricalAccuracy(k=3, name="top3"),
        ],
        "output_4": top_rated_percent,
    },
)

if continue_flag and os.path.isdir(MODEL_DIR):
    model.load_weights(MODEL_DIR)
    print("Weights restored.")


# fit the model and save it even if the process gets disrupted
try:
    model.fit(dataset, epochs=epochs_final, steps_per_epoch=steps_per_epoch)
finally:
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_weights(MODEL_DIR)
    print(f"Saved weights to {MODEL_DIR}")
