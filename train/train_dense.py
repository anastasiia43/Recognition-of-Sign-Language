import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers

# Import own class
from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from git.RecognitionofSignLanguage.utils.TrainUtils import TrainUtils
from git.RecognitionofSignLanguage.models.DenseBlock import DenseBlock
from git.RecognitionofSignLanguage.models.Embedding import Embedding
from git.RecognitionofSignLanguage.data_preprocess.preprocess_data import prepare_data


def get_model(
        train_utils,
        config_loss,
        shape=None,
        init_fc=Cfg.STARTING_LAYER_SIZE
):
    inputs = layers.Input(shape=shape, name='frames')
    x = inputs

    lips, left_hand, pose = train_utils.prepare_for_embedding(x, MEAN_STD)
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

    model = train_utils.compice_model(config_loss, model)

    return model


if __name__ == '__main__':
    # prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, MEAN_STD = prepare_data()
    train_utils = TrainUtils()

    # Create model
    model = get_model(train_utils=train_utils, config_loss={'loss': 'CrossentropyLabelSmoothing'},
                      shape=X_train.shape[1:])

    train_utils.train_and_inference(model, X_train, X_val, X_test, y_train, y_val, y_test)
