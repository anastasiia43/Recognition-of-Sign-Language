from tensorflow.keras import layers


class ClassifierLSTM(layers.Layer):
    def __init__(self, lstm_units, drop, use_embedding):
        super().__init__()
        self.use_embedding = use_embedding
        self.dropout = layers.Dropout(drop)
        if self.use_embedding:
            self.pool2d = layers.AveragePooling1D(pool_size=4)  # Keep the same number of landmark
        else:
            self.pool2d = layers.AveragePooling2D(pool_size=(4, 1))  # Keep the same number of landmark
        self.reshape = layers.Reshape((-1, 64))
        self.lstm = layers.LSTM(units=lstm_units, return_sequences=True)
        self.pool1d = layers.AveragePooling1D(pool_size=4)
        self.flat = layers.Flatten()
        self.outputs = layers.Dense(250, activation="softmax", name="predictions")

    def call(self, x):  # (None, 23, 88, 64)
        x = self.pool2d(x)  # (None, 5, 88, 64)
        if not self.use_embedding:
            x = self.reshape(x)  # (None, 2024, 64)
        x = self.lstm(x)  # (None, 2024, 250)
        x = self.dropout(x)
        x = self.pool1d(x)  # (None, 506, 250)
        x = self.flat(x)  # (None, 126_500)
        outputs = self.outputs(x)  # (None, 126 _00)
        return outputs
