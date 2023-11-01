from tensorflow.keras import layers


class ClassifierDense(layers.Layer):
    def __init__(self, output_channels, drop):
        super().__init__()
        self.dense = layers.Dense(output_channels)
        self.batch = layers.BatchNormalization()
        self.gelu = layers.Activation("gelu")
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.dense(x)
        x = self.batch(x)
        x = self.gelu(x)
        x = self.drop(x)
        return x