import numpy as np

from ccml.data import build_dataset
from ccml.metrics import relative_pick, top_rated_percent
from ccml.model import CubeCobraMLSystem
from ccml.utils import DATA_DIR, MODEL_DIR

TEST_DIR = DATA_DIR / "test"
BATCH_SIZE = 128
PRIMARY = "pick"

test_ds, card_count, steps = build_dataset(
    cubes_path=TEST_DIR / "cubes.json",
    decks_path=TEST_DIR / "decks.json",
    picks_path=TEST_DIR / "picks.json",
    freq_path=TEST_DIR / "oracleFrequency.json",
    correlations_path=TEST_DIR / "oracleFrequency.json",
    batch_size=BATCH_SIZE,
    target_epochs=1,
    primary=PRIMARY,
)

model = CubeCobraMLSystem(card_count)
model.load_weights(MODEL_DIR)
print(f"✔  Loaded weights from {MODEL_DIR}")

totals = np.zeros(4)
for b, (x, y_true) in enumerate(test_ds.take(steps)):
    y_pred = model(x, training=False)

    totals[0] += top_rated_percent(y_true[0].numpy(), y_pred[0].numpy())
    totals[1] += top_rated_percent(y_true[1].numpy(), y_pred[1].numpy())
    totals[2] += relative_pick(y_true[2].numpy(), y_pred[2].numpy(), k=1)
    totals[3] += relative_pick(y_true[2].numpy(), y_pred[2].numpy(), k=3)

print("\n---  Final metrics  ---")
print(f"Cubes accuracy        : {totals[0] / steps:.4f}")
print(f"Decks accuracy        : {totals[1] / steps:.4f}")
print(f"Pick top-1 (relative) : {totals[2] / steps:.4f}")
print(f"Pick top-3 (relative) : {totals[3] / steps:.4f}")
