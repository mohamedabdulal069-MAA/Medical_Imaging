"""Six independent binary targets; no multiclass class_weight conversion."""
import numpy as np
from .dataset import validate_labels


def training_positive_weights(frame):
    from . import LABELS
    if set(frame.partition) != {"train"}:
        raise ValueError("Loss weights may use training labels only")
    y = frame[list(LABELS)].to_numpy()
    validate_labels(y)
    positives = y.sum(axis=0)
    if np.any(positives == 0) or np.any(positives == len(y)):
        raise ValueError("Cannot estimate class weights without both outcomes per label")
    return (len(y)-positives)/positives


def binary_loss(positive_weights=None):
    import tensorflow as tf
    if positive_weights is None:
        return tf.keras.losses.BinaryCrossentropy()
    weights = np.asarray(positive_weights, dtype=np.float32)
    if weights.shape != (6,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Expected six positive finite weights")
    def weighted_bce(y_true, y_pred):
        p = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        return tf.reduce_mean(-(y_true*weights*tf.math.log(p)+(1-y_true)*tf.math.log1p(-p)), axis=-1)
    return weighted_bce

