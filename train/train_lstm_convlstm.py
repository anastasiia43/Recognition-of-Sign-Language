import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers

# Import own class
from git.RecognitionofSignLanguage.utils.TrainUtils import TrainUtils
from git.RecognitionofSignLanguage.models.DenseBlock import DenseBlock
from git.RecognitionofSignLanguage.models.Embedding import Embedding
from git.RecognitionofSignLanguage.models.ClassifierLSTM import ClassifierLSTM
from git.RecognitionofSignLanguage.models.ClassifierConvLSTM1D import ClassifierConvLSTM1D

from git.RecognitionofSignLanguage.data_preprocess.preprocess_data import prepare_data


def get_model(
        train_utils,
        config_loss,
        shape=None,
        use_conv=False,
        encoder_units=[254, 128, 64],
        drop=0.6,
        lstm_units=250
):
    inputs = layers.Input(shape=shape, name='frames')
    x = inputs

    lips, left_hand, pose = train_utils.prepare_for_embedding(x, MEAN_STD)
    # Embedding
    x = Embedding()(lips, left_hand, pose)

    for units in encoder_units:
        x = DenseBlock(units, drop)(x)

    if use_conv:
        outputs = ClassifierConvLSTM1D(lstm_units, drop, use_embedding=True)(x)
    else:
        outputs = ClassifierLSTM(lstm_units, drop, use_embedding=True)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model = train_utils.compice_model(config_loss, model)

    return model


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test, MEAN_STD = prepare_data()
    train_utils = TrainUtils()

    # Create model
    model = get_model(train_utils=train_utils, shape=list(X_train.shape[1:]), use_conv=False,
                      config_loss={'loss': 'CrossentropyLabelSmoothing'})

    train_utils.train_and_inference(model, X_train, X_val, X_test, y_train, y_val, y_test)
