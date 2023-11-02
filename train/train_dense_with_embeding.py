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
from git.RecognitionofSignLanguage.utils.Landmark_indices import Landmarks as lm
from git.RecognitionofSignLanguage.models.DenseBlock import DenseBlock
from git.RecognitionofSignLanguage.models.Embedding import Embedding
from git.RecognitionofSignLanguage.data_preprocess.Preprocess_data import get_split_data


def get_model(
        shape=None,
        init_fc=Cfg.STARTING_LAYER_SIZE
):
    inputs = layers.Input(shape=shape, name='frames')
    x = inputs

    x = tf.slice(x, [0, 0, 0, 0], [-1, Cfg.INPUT_SIZE, shape[1], Cfg.N_DIMS])
    # LIPS
    lips = tf.slice(x, [0, 0, lm.LIPS_START, 0], [-1, Cfg.INPUT_SIZE, 40, Cfg.N_DIMS])
    lips = tf.where(
        tf.math.equal(lips, 0.0),
        0.0,
        lips,
    )
    # LEFT HAND
    left_hand = tf.slice(x, [0, 0, 40, 0], [-1, Cfg.INPUT_SIZE, 21, Cfg.N_DIMS])
    left_hand = tf.where(
        tf.math.equal(left_hand, 0.0),
        0.0,
        left_hand,
    )
    # POSE
    pose = tf.slice(x, [0, 0, 61, 0], [-1, Cfg.INPUT_SIZE, 5, Cfg.N_DIMS])
    pose = tf.where(
        tf.math.equal(pose, 0.0),
        0.0,
        pose,
    )

    # Flatten
    lips = tf.reshape(lips, [-1, Cfg.INPUT_SIZE, 40 * Cfg.N_DIMS])
    left_hand = tf.reshape(left_hand, [-1, Cfg.INPUT_SIZE, 21 * Cfg.N_DIMS])
    pose = tf.reshape(pose, [-1, Cfg.INPUT_SIZE, 5 * Cfg.N_DIMS])

    # Embedding
    x = Embedding()(lips, left_hand, pose)

    x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])

    # Define layers
    for i in range(len(Cfg.DROPOUTS)):
        x = DenseBlock(init_fc // (2 ** i), Cfg.DROPOUTS[i])(x)

    # Define output layer
    outputs = tf.keras.layers.Dense(Cfg.NUM_CLASSES, activation="softmax")(x)

    # Build the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

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
    name_folder = 'Embedding'

    # Get split data
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data()


    # Create model
    model = get_model(shape=X_train.shape[1:])
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
