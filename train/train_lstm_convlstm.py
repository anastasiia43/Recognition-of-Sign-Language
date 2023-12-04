import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers

# Import own class
from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from git.RecognitionofSignLanguage.utils.train_utils import TrainUtils
from git.RecognitionofSignLanguage.models.DenseBlock import DenseBlock
from git.RecognitionofSignLanguage.models.Embedding import Embedding
from git.RecognitionofSignLanguage.models.ClassifierLSTM import ClassifierLSTM
from git.RecognitionofSignLanguage.models.ClassifierConvLSTM1D import ClassifierConvLSTM1D
from git.RecognitionofSignLanguage.data_preprocess.Preprocess_data import get_split_data
from git.RecognitionofSignLanguage.data_preprocess.calculate_mean_std import get_all_mean_std

from git.RecognitionofSignLanguage.loss.ArcFaceLoss import ArcFaceLoss
from git.RecognitionofSignLanguage.loss.CrossentropyLabelSmoothing import CrossentropyLabelSmoothing
from git.RecognitionofSignLanguage.loss.CustomLoss import CustomLoss

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
        outputs = ClassifierConvLSTM1D(lstm_units, drop, use_embedding = True)(x)
    else:
        outputs = ClassifierLSTM(lstm_units, drop, use_embedding = True)(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Sparse Categorical Cross Entropy With Label Smoothing
    if config_loss['loss'] == "CustomLoss":
        loss = CustomLoss(config_loss)
    elif config_loss['loss'] == "ArcFaceLoss":
        loss = ArcFaceLoss()
    else:
        loss = CrossentropyLabelSmoothing()

    model.compile(
        loss=loss,
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
    SHAPE = list(X_train.shape[1:])

    MEAN_STD = get_all_mean_std(X_train)

    # Create model
    model = get_model(train_utils=train_utils, shape=SHAPE, use_conv=False, config_loss={'loss': 'CrossentropyLabelSmoothing'})
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

