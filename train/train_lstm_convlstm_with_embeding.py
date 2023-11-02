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
from git.RecognitionofSignLanguage.utils.Landmark_indices import Landmarks as lm
from git.RecognitionofSignLanguage.utils.train_utils import TrainUtils
from git.RecognitionofSignLanguage.models.DenseBlock import DenseBlock
from git.RecognitionofSignLanguage.models.Embedding import Embedding
from git.RecognitionofSignLanguage.models.ClassifierLSTM import ClassifierLSTM
from git.RecognitionofSignLanguage.models.ClassifierConvLSTM1D import ClassifierConvLSTM1D
from git.RecognitionofSignLanguage.data_preprocess.Preprocess_data import get_split_data
from git.RecognitionofSignLanguage.data_preprocess.calculate_mean_std import get_all_mean_std


def get_model(
        shape=None,
        use_conv=False,
        encoder_units=[254, 128, 64],
        drop=0.6,
        lstm_units=250,
        learning_rate=Cfg.LR,
):
    inputs = layers.Input(shape=shape, name='frames')
    x = inputs

    x = tf.slice(x, [0, 0, 0, 0], [-1, Cfg.INPUT_SIZE, SHAPE[1], Cfg.N_DIMS])
    # LIPS
    lips = tf.slice(x, [0, 0, lm.LIPS_START, 0], [-1, Cfg.INPUT_SIZE, 40, Cfg.N_DIMS])
    lips = tf.where(
        tf.math.equal(lips, 0.0),
        0.0,
        (lips - MEAN_STD['lips_mean']) / MEAN_STD['lips_std'],
    )
    # LEFT HAND
    left_hand = tf.slice(x, [0, 0, 40, 0], [-1, Cfg.INPUT_SIZE, 21, Cfg.N_DIMS])
    left_hand = tf.where(
        tf.math.equal(left_hand, 0.0),
        0.0,
        (left_hand - MEAN_STD['hand_mean']) / MEAN_STD['hand_std'],
    )
    # POSE
    pose = tf.slice(x, [0, 0, 61, 0], [-1, Cfg.INPUT_SIZE, 5, Cfg.N_DIMS])
    pose = tf.where(
        tf.math.equal(pose, 0.0),
        0.0,
        (pose - MEAN_STD['pose_mean']) / MEAN_STD['pose_std'],
    )

    # Flatten
    lips = tf.reshape(lips, [-1, Cfg.INPUT_SIZE, 40 * Cfg.N_DIMS])
    left_hand = tf.reshape(left_hand, [-1, Cfg.INPUT_SIZE, 21 * Cfg.N_DIMS])
    pose = tf.reshape(pose, [-1, Cfg.INPUT_SIZE, 5 * Cfg.N_DIMS])

    # Embedding
    x = Embedding()(lips, left_hand, pose)

    for units in encoder_units:
        x = DenseBlock(units, drop)(x)

    if use_conv:
        outputs = ClassifierConvLSTM1D(lstm_units, drop, use_embedding = True)(x)
    else:
        outputs = ClassifierLSTM(lstm_units, drop, use_embedding = True)(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
    SHAPE = list(X_train.shape[1:])

    MEAN_STD = get_all_mean_std(X_train)

    # Create model
    model = get_model(shape=SHAPE, use_conv = False)
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
