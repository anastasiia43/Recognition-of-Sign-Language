import tensorflow as tf
from git.RecognitionofSignLanguage.utils.Cfg import Cfg


# source:: https://stackoverflow.com/questions/60689185/label-smoothing-for-sparse-categorical-crossentropy
class CrossentropyLabelSmoothing(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.25):
        super(CrossentropyLabelSmoothing, self).__init__()
        self.label_smoothing = label_smoothing
    def call(self, y_true, y_pred):
        # One Hot Encode Sparsely Encoded Target Sign
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, Cfg.NUM_CLASSES, axis=1)
        y_true = tf.squeeze(y_true, axis=2)
        # Categorical Crossentropy with native label smoothing support
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=self.label_smoothing)
