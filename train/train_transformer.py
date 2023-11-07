import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from git.RecognitionofSignLanguage.data_preprocess.Preprocess_data import get_split_data
from git.RecognitionofSignLanguage.data_preprocess.calculate_mean_std import get_all_mean_std
from git.RecognitionofSignLanguage.utils.Landmark_indices import Landmarks as lm
from git.RecognitionofSignLanguage.utils.Transformer_utils import Transformer_Utils as tu

from git.RecognitionofSignLanguage.models.Embedding import Embedding
from git.RecognitionofSignLanguage.models.Transformer import Transformer
from git.RecognitionofSignLanguage.utils.train_utils import TrainUtils

import warnings
warnings.filterwarnings('always')


# source:: https://stackoverflow.com/questions/60689185/label-smoothing-for-sparse-categorical-crossentropy
def scce_with_ls(y_true, y_pred):
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, Cfg.NUM_CLASSES, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    # Categorical Crossentropy with native label smoothing support
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25)


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio=tu.WD_RATIO):
        self.step_counter = 0
        self.wd_ratio = wd_ratio

    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(
            f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')


def get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, n=tu.BATCH_ALL_SIGNS_N):
    # Arrays to store batch in
    X_batch = np.zeros([Cfg.NUM_CLASSES * n, Cfg.INPUT_SIZE, lm.N_COLS, Cfg.N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, Cfg.NUM_CLASSES, step=1 / n, dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros([Cfg.NUM_CLASSES * n, Cfg.INPUT_SIZE], dtype=np.float32)

    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(Cfg.NUM_CLASSES):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)

    while True:
        # Fill batch arrays
        for i in range(Cfg.NUM_CLASSES):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch[i * n:(i + 1) * n] = X[idxs]
            non_empty_frame_idxs_batch[i * n:(i + 1) * n] = NON_EMPTY_FRAME_IDXS[idxs]

        yield {'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch}, y_batch


def get_model():
    # Inputs
    frames = tf.keras.layers.Input([Cfg.INPUT_SIZE, lm.N_COLS, Cfg.N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([Cfg.INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    # Padding Mask
    mask0 = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask0 = tf.expand_dims(mask0, axis=2)
    # Random Frame Masking
    mask = tf.where(
        (tf.random.uniform(tf.shape(mask0)) > 0.25) & tf.math.not_equal(mask0, 0.0),
        1.0,
        0.0,
    )
    # Correct Samples Which are all masked now...
    mask = tf.where(
        tf.math.equal(tf.reduce_sum(mask, axis=[1, 2], keepdims=True), 0.0),
        mask0,
        mask,
    )

    """
        left_hand: 468:489
        pose: 489:522
        right_hand: 522:543
    """
    x = frames
    x = tf.slice(x, [0, 0, 0, 0], [-1, Cfg.INPUT_SIZE, lm.N_COLS, Cfg.N_DIMS])
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
    x = Embedding()(lips, left_hand, pose, non_empty_frame_idxs)

    # Encoder Transformer Blocks
    x = Transformer(tu.NUM_BLOCKS)(x, mask)

    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classifier Dropout
    x = tf.keras.layers.Dropout(tu.CLASSIFIER_DROPOUT_RATIO)(x)
    # Classification Layer
    x = tf.keras.layers.Dense(Cfg.NUM_CLASSES, activation=tf.keras.activations.softmax,
                              kernel_initializer=tu.INIT_GLOROT_UNIFORM)(x)

    outputs = x

    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    # Sparse Categorical Cross Entropy With Label Smoothing
    loss = scce_with_ls

    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)

    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top-5-accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
    ]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

if __name__ == '__main__':

    train_utils = TrainUtils()
    train_utils.seed_it_all()

    # Create name folder
    name_folder = 'Defoult'

    # Get split data
    X_train, X_val, X_test, y_train, y_val, y_test, NON_EMPTY_FRAME_IDXS_TRAIN, NON_EMPTY_FRAME_IDXS_VAL, NON_EMPTY_FRAME_IDXS_TEST = get_split_data(True)

    MEAN_STD = get_all_mean_std(X_train)

    tf.keras.backend.clear_session()

    # Create model
    model = get_model()
    # Plot model summary
    model.summary()

    # Create Folder
    if not os.path.exists(f'{Cfg.MODEL_OUT_PATH}{name_folder}'):
        os.makedirs(f'{Cfg.MODEL_OUT_PATH}{name_folder}')

    callbacks = train_utils.create_callback(name_folder)


    # Learning rate for encoder
    LR_SCHEDULE = [train_utils.lrfn(step, num_warmup_steps=tu.N_WARMUP_EPOCHS, lr_max=tu.LR_MAX, num_cycles=0.50) for step in
                   range(tu.N_EPOCHS)]
    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)


    tf.keras.backend.clear_session()
    # Get new fresh model
    model = get_model()
    # Actual Training
    history = model.fit(
        x=get_train_batch_all_signs(X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN),
        steps_per_epoch=len(X_train) // (Cfg.NUM_CLASSES * tu.BATCH_ALL_SIGNS_N),
        epochs=tu.N_EPOCHS,
        # Only used for validation data since training data is a generator
        batch_size=Cfg.BATCH_SIZE,
        validation_data= ({ 'frames': X_val, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL }, y_val),
        callbacks=[
            callbacks[0],
            callbacks[2],
            lr_callback,
            WeightDecayCallback(),
        ],
        verbose=1,
    )
    train_utils.save_plot(history, "loss", name_folder)
    train_utils.save_plot(history, "accuracy", name_folder)
    train_utils.save_plot(history, "top-5-accuracy", name_folder)
    train_utils.save_plot(history, "top-10-accuracy", name_folder)

    del X_train, X_val
    train_utils.inference({'frames': X_test, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_TEST}, y_test, model, name_folder)
    train_utils.print_classification_report({'frames': X_test, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_TEST}, y_test, model, name_folder)