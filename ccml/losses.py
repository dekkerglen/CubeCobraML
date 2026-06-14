"""Custom losses for the draft (pick) head."""

import tensorflow as tf
from tensorflow import keras

_EPS = 1e-7


def pick_histogram_divergence(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Symmetric KL between the batch-mean target and prediction histograms.

    `mean(y_true, 0)` is the human pick distribution over this batch's packs;
    `mean(y_pred, 0)` is the model's. Cards not offered in this batch have
    ~zero mass in BOTH histograms, so the divergence is well-defined on the
    batch's own support — no global "card never offered" problem.

    Two-sided on purpose: KL(human‖model) alone is blind to overpick
    (model mass where humans have ~none), KL(model‖human) alone is blind to
    collapse (model puts zero mass where humans pick). Symmetric KL punishes
    both failure modes.
    """
    ht = tf.reduce_mean(y_true, axis=0) + _EPS
    hp = tf.reduce_mean(y_pred, axis=0) + _EPS
    ht = ht / tf.reduce_sum(ht)
    hp = hp / tf.reduce_sum(hp)
    log_ratio = tf.math.log(ht) - tf.math.log(hp)
    return tf.reduce_sum(ht * log_ratio) - tf.reduce_sum(hp * log_ratio)


class PickHistogramCCE(keras.losses.Loss):
    """Categorical crossentropy + intra-batch pick-histogram regularizer.

    Per-example CCE behaves exactly like Keras's "categorical_crossentropy"
    on probability outputs. The histogram term `hist_weight * sym_kl` is a
    batch-level scalar broadcast onto every example, which is equivalent to
    adding it once to the total loss.

    Note: the histogram term uses UNWEIGHTED batch means — `Loss.call` never
    sees sample_weight, so pack-size weights apply only to the per-example
    CCE (Keras weights it downstream as usual). That is the desired behavior:
    the marginal pick distribution should match humans across all picks, not
    just early-pack ones.
    """

    def __init__(self, hist_weight: float, name: str = "pick_histogram_cce"):
        super().__init__(name=name)
        self.hist_weight = float(hist_weight)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred_clipped = tf.clip_by_value(y_pred, _EPS, 1.0)
        cce = -tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=-1)
        if self.hist_weight <= 0.0:
            return cce
        return cce + self.hist_weight * pick_histogram_divergence(y_true, y_pred)
