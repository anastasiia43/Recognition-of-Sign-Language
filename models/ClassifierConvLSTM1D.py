from tensorflow.keras import layers
import tensorflow as tf


class ClassifierConvLSTM1D(layers.Layer):
    def __init__(self, lstm_units, drop, use_embedding):
        super().__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.pool2d = layers.AveragePooling1D(pool_size=6)  # Keep the same number of landmark
        else:
            self.pool2d = layers.AveragePooling2D(pool_size=(6, 1))  # Keep the same number of landmark

        self.conv_lstm1D = layers.ConvLSTM1D(filters=lstm_units,
                                             kernel_size=1)  # RNN capable of learning long-term dependencies
        self.dropout = layers.Dropout(drop)
        self.flat = layers.Flatten()
        self.outputs = layers.Dense(250, activation="softmax", name="predictions")

    def call(self, x):  # (None, 23, 88, 64)
        x = self.pool2d(x)
        if self.use_embedding:
            x = tf.expand_dims(x, axis=-2)  # Add height
        x = self.conv_lstm1D(x)  # (None, 88, 250)
        if self.use_embedding:
            x = tf.squeeze(x, axis=-2)  # Delete height
        x = self.dropout(x)
        x = self.flat(x)
        outputs = self.outputs(x)
        return outputs
