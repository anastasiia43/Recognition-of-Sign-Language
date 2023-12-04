import os
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

from git.RecognitionofSignLanguage.loss.ArcFaceLoss import ArcFaceLoss
from git.RecognitionofSignLanguage.loss.CrossentropyLabelSmoothing import CrossentropyLabelSmoothing
from git.RecognitionofSignLanguage.loss.CustomLoss import CustomLoss


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio=tu.WD_RATIO):
        self.step_counter = 0
        self.wd_ratio = wd_ratio

    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(
            f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')


def get_model(train_utils, config_loss):
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

    lips, left_hand, pose = train_utils.prepare_for_embedding(frames, MEAN_STD)
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
    if config_loss['loss'] == "CustomLoss":
        loss = CustomLoss(config_loss)
    elif config_loss['loss'] == "ArcFaceLoss":
        loss = ArcFaceLoss()
    else:
        loss = CrossentropyLabelSmoothing()

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

    # Create Folder
    if not os.path.exists(f'{Cfg.MODEL_OUT_PATH}{name_folder}'):
        os.makedirs(f'{Cfg.MODEL_OUT_PATH}{name_folder}')

    callbacks = train_utils.create_callback(name_folder)


    # Learning rate for encoder
    LR_SCHEDULE = [train_utils.lrfn(step, num_warmup_steps=tu.N_WARMUP_EPOCHS, lr_max=Cfg.LR, num_cycles=0.50) for step in
                   range(Cfg.EPOCHS)]
    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)


    tf.keras.backend.clear_session()
    # Create model
    model = get_model(train_utils, {'loss': 'CrossentropyLabelSmoothing'})
    # Plot model summary
    model.summary()
    # Actual Training
    history = model.fit(
        {'frames': X_train, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_TRAIN}, y_train,
        steps_per_epoch=len(X_train) // (Cfg.NUM_CLASSES * 2),
        epochs=Cfg.EPOCHS,
        # Only used for validation data since training data is a generator
        batch_size=Cfg.BATCH_SIZE,
        validation_data=({'frames': X_val, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL }, y_val),
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