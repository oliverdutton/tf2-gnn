from typing import Any, Dict, Iterable, List, NamedTuple, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

def micro_f1(logits, labels):
    # Everything on int, because who trusts float anyway?
    predicted = tf.math.round(tf.nn.sigmoid(logits))
    predicted = tf.cast(predicted, dtype=tf.int32)
    labels = tf.cast(labels, dtype=tf.int32)

    true_pos = tf.math.count_nonzero(predicted * labels)
    false_pos = tf.math.count_nonzero(predicted * (labels - 1))
    false_neg = tf.math.count_nonzero((predicted - 1) * labels)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fmeasure = (2 * precision * recall) / (precision + recall)
    return tf.cast(fmeasure, tf.float32)
