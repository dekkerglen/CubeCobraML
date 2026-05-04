"""Promote a lightweight epoch checkpoint to full SavedModels in ./model/,
ready for scripts/convert.sh to convert to tfjs.

The training loop saves cheap .h5 weight files per epoch under
model/runs/<run>/checkpoints/epoch_NNNN/. scripts/convert.sh, however, needs
full SavedModels at ./model/<sub>/model/.

We rebuild every sub-model fresh from ccml.model rather than loading the
existing SavedModels — those can drift from the current architecture (e.g.
the older draft_decoder SavedModel was trained with a 258-dim input, while
the current code feeds 256). num_cards is sniffed once from the existing
cube_decoder SavedModel's output dim.

Usage:
    uv run python -m scripts.promote_checkpoint <checkpoint_dir>

Example:
    uv run python -m scripts.promote_checkpoint model/runs/no-context-bisect/checkpoints/epoch_0200
    ./scripts/convert.sh
"""

import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from ccml.model import Decoder, Encoder  # noqa: E402
from ccml.utils import MODEL_DIR  # noqa: E402

# (on-disk dir name, Decoder ctor name, input dim for inner Sequential, output activation)
DECODERS = [
    ("cube_decoder",        "recommend",  128, tf.nn.sigmoid),
    ("draft_decoder",       "draft",      256, "linear"),
    ("deck_build_decoder",  "deck_build", 256, tf.nn.sigmoid),
    ("correlation_decoder", "correlate",  128, tf.nn.softmax),
]


def _sniff_num_cards() -> int:
    cd_path = os.path.join(str(MODEL_DIR), "cube_decoder", "model")
    if not os.path.isdir(cd_path):
        sys.exit(
            f"need {cd_path} to infer num_cards. Either keep an old SavedModel around "
            "or hardcode num_cards here."
        )
    cd = keras.models.load_model(cd_path)
    return int(cd.output_shape[-1])


def _promote_encoder(ckpt_dir: str, num_cards: int) -> None:
    h5 = os.path.join(ckpt_dir, "encoder.weights.h5")
    out = os.path.join(str(MODEL_DIR), "encoder", "model")
    print(f"[encoder] building fresh, loading {h5}, saving → {out}")
    enc = Encoder("encoder")
    enc.model.build((None, num_cards))
    enc.model.load_weights(h5)
    enc.model.save(out)


def _promote_decoder(ckpt_dir: str, disk_name: str, ctor_name: str, in_dim: int, act, num_cards: int) -> None:
    h5 = os.path.join(ckpt_dir, f"{disk_name}.weights.h5")
    out = os.path.join(str(MODEL_DIR), disk_name, "model")
    print(f"[{disk_name}] building fresh (in={in_dim}, out={num_cards}), loading {h5}, saving → {out}")
    dec = Decoder(ctor_name, num_cards, act)
    dec.model.build((None, in_dim))
    dec.model.load_weights(h5)
    dec.model.save(out)


def main(ckpt_dir: str) -> None:
    if not os.path.isdir(ckpt_dir):
        sys.exit(f"checkpoint dir not found: {ckpt_dir}")
    for disk_name, *_ in DECODERS:
        h5 = os.path.join(ckpt_dir, f"{disk_name}.weights.h5")
        if not os.path.isfile(h5):
            sys.exit(f"missing checkpoint weights: {h5}")
    if not os.path.isfile(os.path.join(ckpt_dir, "encoder.weights.h5")):
        sys.exit(f"missing checkpoint weights: {ckpt_dir}/encoder.weights.h5")

    num_cards = _sniff_num_cards()
    print(f"inferred num_cards = {num_cards}")

    _promote_encoder(ckpt_dir, num_cards)
    for disk_name, ctor_name, in_dim, act in DECODERS:
        _promote_decoder(ckpt_dir, disk_name, ctor_name, in_dim, act, num_cards)

    print("\nDone. Now run: ./scripts/convert.sh")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m scripts.promote_checkpoint <checkpoint_dir>")
    main(sys.argv[1])
