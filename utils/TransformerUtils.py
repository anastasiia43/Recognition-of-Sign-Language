import tensorflow as tf


class TransformerUtils:
    # Epsilon value for layer normalisation
    LAYER_NORM_EPS = 1e-6

    # Dense layer units for landmarks
    LIPS_UNITS = 384
    HANDS_UNITS = 384
    POSE_UNITS = 384
    # final embedding and transformer embedding size
    UNITS = 512

    # Transformer
    NUM_BLOCKS = 2
    MLP_RATIO = 2

    # Dropout
    EMBEDDING_DROPOUT = 0.00
    MLP_DROPOUT_RATIO = 0.30
    CLASSIFIER_DROPOUT_RATIO = 0.10

    # Initiailizers
    INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
    INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
    INIT_ZEROS = tf.keras.initializers.constant(0.0)
    # Activations
    GELU = tf.keras.activations.gelu

    N_WARMUP_EPOCHS = 0
    WD_RATIO = 0.05
