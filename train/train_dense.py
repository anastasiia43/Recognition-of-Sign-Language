import os
import random
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import own class
from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from git.RecognitionofSignLanguage.utils.train_utils import TrainUtils
from git.RecognitionofSignLanguage.models.DenseBlock import DenseBlock
from git.RecognitionofSignLanguage.data_preprocess.Preprocess_data import get_split_data

def get_model(
        flat_frame_len=None,
        init_fc=Cfg.STARTING_LAYER_SIZE
):
    input = tf.keras.layers.Input(shape=(flat_frame_len,))
    x = input

    # Define layers
    for i in range(len(Cfg.DROPOUTS)):
        x = DenseBlock(init_fc // (2 ** i), Cfg.DROPOUTS[i])(x)

    # Define output layer
    outputs = tf.keras.layers.Dense(Cfg.NUM_CLASSES, activation="softmax")(x)

    # Build the model
    model = tf.keras.models.Model(inputs=input, outputs=outputs)
    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=Cfg.LR),
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top-5-accuracy"),
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top-10-accuracy")
                 ],
    )

    return model


if __name__ == '__main__':
    train_utils = TrainUtils()
    train_utils.seed_it_all()

    # Create name folder
    name_folder = 'Defoult'

    # Get split data
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data()

    # Reshape data for model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]),
                         order='C').astype(np.float32)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1] * X_val.shape[2] * X_val.shape[3]), order='C').astype(
        np.float32)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]),
                        order='C').astype(np.float32)


    # Create model
    model = get_model(flat_frame_len=X_train.shape[1])
    model.summary()

    # Create Folder
    if not os.path.exists(f'{Cfg.MODEL_OUT_PATH}{name_folder}'):
        os.makedirs(f'{Cfg.MODEL_OUT_PATH}{name_folder}')

    callbacks = train_utils.create_callback(name_folder)

    # change data for save in ram data
    train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(4 * Cfg.BATCH_SIZE).batch(Cfg.BATCH_SIZE)
    del X_train, y_train

    validate = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(Cfg.BATCH_SIZE)
    del X_val, y_val

    model = train_utils.train(train, validate, model, callbacks, name_folder)
    train_utils.inference(X_test, y_test, model, name_folder)
