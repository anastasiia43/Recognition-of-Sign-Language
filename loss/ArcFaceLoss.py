import tensorflow as tf
from git.RecognitionofSignLanguage.utils.Cfg import Cfg


class ArcFaceLoss(tf.keras.losses.Loss):
    def __init__(self, s=20, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m

    def call(self, y_true, y_pred):
        # One Hot Encode Sparse Target Labels
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, Cfg.NUM_CLASSES, axis=1)
        y_true = tf.squeeze(y_true, axis=2)

        # Normalize feature vectors
        y_pred = tf.math.l2_normalize(y_pred, axis=1)

        # Compute logits
        logits = tf.matmul(y_pred, tf.nn.l2_normalize(tf.eye(Cfg.NUM_CLASSES), axis=0))

        # Add margin to the logits
        theta = tf.acos(tf.clip_by_value(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = tf.cos(theta + self.m)

        # Apply temperature scaling
        target_logits *= self.s

        # Compute softmax cross-entropy loss
        loss = tf.keras.losses.categorical_crossentropy(y_true, target_logits, from_logits=True)

        return loss
