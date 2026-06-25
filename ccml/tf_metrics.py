import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric

from ccml.losses import pick_histogram_divergence


class BombAgreement(Metric):
    """Agreement (argmax == target) restricted to examples where the target
    card is in the bomb cohort — the top N most-picked cards in the training
    data. Catches the basin-flip failure mode that cohort-average metrics
    cannot see: when the model swaps its "favorite" top-cohort cards
    (e.g. abandons Black Lotus for Taiga), `val_bomb_agreement` collapses
    while `cohort_target_top1_top` stays high — because both cards are in
    the same top cohort.

    This is the headline metric for `BEST_METRIC` and deployment gating."""

    def __init__(self, bomb_mask, name="bomb_agreement", **kwargs):
        super().__init__(name=name, **kwargs)
        self.bomb_mask = tf.constant(np.asarray(bomb_mask, dtype=np.float32))
        self.total = self.add_weight(name="total", initializer="zeros")
        self.correct = self.add_weight(name="correct", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        target_idx = tf.argmax(y_true, axis=-1)
        pred_idx = tf.argmax(y_pred, axis=-1)
        target_is_bomb = tf.gather(self.bomb_mask, target_idx)
        correct = tf.cast(tf.equal(target_idx, pred_idx), tf.float32)
        self.total.assign_add(tf.reduce_sum(target_is_bomb))
        self.correct.assign_add(tf.reduce_sum(correct * target_is_bomb))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.total.assign(0.0)
        self.correct.assign(0.0)


class TopRatedPercent(Metric):
    """Recall-style metric for multi-label outputs (cube and deck heads).

    For each example, picks the top-k predictions (where k = 1.2 * num_true)
    and reports the fraction of true positives among them, with add-1
    smoothing. Used as a coarse "are the cube/deck heads roughly tracking?"
    indicator — not the deployment gate (that's bomb_agreement on the draft
    head).
    """

    def __init__(self, name="top_rated_percent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_score = self.add_weight(name="total_score", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)

        def _per_sample(inputs):
            truth, pred = inputs
            n_true = tf.reduce_sum(truth)
            k = tf.cast(tf.math.ceil(n_true * 1.2), tf.int32)
            num_classes = tf.shape(truth)[-1]
            k = tf.minimum(k, num_classes)
            _, top_indices = tf.nn.top_k(pred, k=k, sorted=False)
            hits = tf.reduce_sum(tf.gather(truth, top_indices))
            return (hits + 1.0) / (n_true + 1.0)

        scores = tf.map_fn(_per_sample, (y_true, y_pred), dtype=tf.float32)
        batch_sum = tf.reduce_sum(scores)
        batch_size = tf.cast(tf.shape(scores)[0], tf.float32)

        self.total_score.assign_add(batch_sum)
        self.count.assign_add(batch_size)

    def result(self):
        return tf.math.divide_no_nan(self.total_score, self.count)

    def reset_state(self):
        self.total_score.assign(0.0)
        self.count.assign(0.0)


class ArgmaxCollapseCount(Metric):
    """Counts cards whose model argmax-rate has collapsed to 0% or saturated
    at 100% while the human pick-rate disagrees. The two targeted collapse
    indicators: both should trend toward zero as the model calibrates.

    Per-card accumulators over the evaluation pass:
      in_pack_count[c]     — packs in which card c was offered
      model_argmax_count[c] — times the model's argmax was c
      human_pick_count[c]  — times the human picked c

    `result()` restricts to cards offered `band_lo <= n < band_hi` times and
    counts:
      mode="zero":      model never takes it, humans sometimes do
                        (model_argmax == 0 AND human_pick > 0)
      mode="saturated": model always takes it, humans don't always
                        (model_argmax == in_pack AND human_pick < in_pack)

    Most meaningful on the full val pass; the train-side value covers only
    that epoch's sample, so interpret loosely.
    """

    def __init__(self, num_cards: int, band_lo: int, band_hi: int, mode: str, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        if mode not in ("zero", "saturated"):
            raise ValueError(f"mode must be 'zero' or 'saturated', got {mode!r}")
        self.num_cards = int(num_cards)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.mode = mode
        self.in_pack_count = self.add_weight(name="in_pack_count", shape=(self.num_cards,), initializer="zeros")
        self.model_argmax_count = self.add_weight(
            name="model_argmax_count", shape=(self.num_cards,), initializer="zeros"
        )
        self.human_pick_count = self.add_weight(name="human_pick_count", shape=(self.num_cards,), initializer="zeros")
        # Snapshot of the accumulators as of the most recent reset. Keras
        # reuses the same metric objects for train and val (resetting at val
        # start), so by epoch end the train-pass vectors are gone — except
        # here. See `current_vectors` / `last_pass_vectors`.
        self.last_pass: dict[str, np.ndarray] | None = None

    def current_vectors(self) -> dict[str, np.ndarray]:
        """Per-card accumulators of the pass currently in the variables
        (= the val pass when read at on_epoch_end and validation ran)."""
        return {
            "in_pack": self.in_pack_count.numpy().astype(np.int32),
            "model_argmax": self.model_argmax_count.numpy().astype(np.int32),
            "human_pick": self.human_pick_count.numpy().astype(np.int32),
        }

    def last_pass_vectors(self) -> dict[str, np.ndarray] | None:
        """Per-card accumulators stashed at the most recent reset
        (= the train pass when read at on_epoch_end and validation ran)."""
        return self.last_pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Label smoothing guarantees every offered card has y_true > 0, and
        # the picked card always has the largest mass — so y_true encodes
        # both the pack membership and the human pick.
        in_pack = tf.cast(y_true > 0.0, tf.float32)
        self.in_pack_count.assign_add(tf.reduce_sum(in_pack, axis=0))
        model_idx = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        human_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        self.model_argmax_count.assign_add(
            tf.cast(tf.math.bincount(model_idx, minlength=self.num_cards, maxlength=self.num_cards), tf.float32)
        )
        self.human_pick_count.assign_add(
            tf.cast(tf.math.bincount(human_idx, minlength=self.num_cards, maxlength=self.num_cards), tf.float32)
        )

    def result(self):
        in_band = tf.logical_and(
            self.in_pack_count >= self.band_lo,
            self.in_pack_count < self.band_hi,
        )
        if self.mode == "zero":
            cond = tf.logical_and(
                self.model_argmax_count <= 0.0,
                self.human_pick_count > 0.0,
            )
        else:
            cond = tf.logical_and(
                self.model_argmax_count >= self.in_pack_count,
                self.human_pick_count < self.in_pack_count,
            )
        return tf.reduce_sum(tf.cast(tf.logical_and(in_band, cond), tf.float32))

    def reset_state(self):
        self.last_pass = self.current_vectors()
        for var in (self.in_pack_count, self.model_argmax_count, self.human_pick_count):
            var.assign(tf.zeros_like(var))


class PickHistDivergence(Metric):
    """Mean per-batch symmetric KL between the human and model pick
    histograms — the exact quantity `PickHistogramCCE` regularizes, surfaced
    as a metric so the regularizer's behavior is visible in train AND val."""

    def __init__(self, name="pick_hist_div", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total.assign_add(pick_histogram_divergence(y_true, y_pred))
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
