import json
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from git.RecognitionofSignLanguage.utils.Landmark_indices import Landmarks as lm
from git.RecognitionofSignLanguage.utils.Transformer_utils import Transformer_Utils as tu

from git.RecognitionofSignLanguage.models.Embedding import Embedding
from git.RecognitionofSignLanguage.models.Transformer import Transformer
from git.RecognitionofSignLanguage.data_preprocess.Preprocess_layer import PreprocessLayer
from git.RecognitionofSignLanguage.loss.CrossentropyLabelSmoothing import CrossentropyLabelSmoothing


def get_model(MEAN_STD = None, config = None):
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
    x = tf.keras.layers.Dense(Cfg.NUM_CLASSES, activation=tf.keras.activations.softmax, kernel_initializer=tu.INIT_GLOROT_UNIFORM)(x)

    outputs = x

    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    # Sparse Categorical Cross Entropy With Label Smoothing

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
preprocess_layer = PreprocessLayer()
MEAN_STD = {'lips_mean': np.array([[0.41547316, 0.4703109],
                                   [0.41883108, 0.46589926],
                                   [0.4251031, 0.46132725],
                                   [0.43443108, 0.45613772],
                                   [0.44922116, 0.45124054],
                                   [0.46507683, 0.45282847],
                                   [0.48094767, 0.4503409],
                                   [0.49706775, 0.45441225],
                                   [0.5074893, 0.459042],
                                   [0.51495683, 0.46322298],
                                   [0.5194041, 0.46749339],
                                   [0.42011154, 0.4743725],
                                   [0.42671776, 0.47870257],
                                   [0.43735498, 0.48376563],
                                   [0.45124313, 0.48715383],
                                   [0.46759006, 0.4878689],
                                   [0.4840219, 0.4864532],
                                   [0.49807987, 0.4823913],
                                   [0.5085224, 0.47673726],
                                   [0.51504785, 0.47189873],
                                   [0.42054957, 0.4698251],
                                   [0.42767105, 0.468039],
                                   [0.43425494, 0.46649823],
                                   [0.44286954, 0.46516383],
                                   [0.45359397, 0.46455082],
                                   [0.4660549, 0.4645243],
                                   [0.47881094, 0.46388814],
                                   [0.48995924, 0.4638992],
                                   [0.49923772, 0.46471104],
                                   [0.50665605, 0.46587592],
                                   [0.4278821, 0.4701379],
                                   [0.43439475, 0.47023264],
                                   [0.44307244, 0.47029954],
                                   [0.45384783, 0.4705873],
                                   [0.46651328, 0.47081664],
                                   [0.47936586, 0.46996093],
                                   [0.49044493, 0.46908432],
                                   [0.49959165, 0.46851924],
                                   [0.50651324, 0.4680516],
                                   [0.51427364, 0.46727678]]),
            'lips_std': np.array([[0.07096294, 0.07538279],
                                  [0.07054356, 0.07548225],
                                  [0.07025307, 0.07560126],
                                  [0.07005524, 0.07574944],
                                  [0.0698991, 0.0758815],
                                  [0.06942811, 0.07605208],
                                  [0.06894466, 0.07579011],
                                  [0.06833776, 0.07551877],
                                  [0.06780227, 0.0752738],
                                  [0.06733409, 0.07506418],
                                  [0.06684714, 0.07490333],
                                  [0.07054198, 0.07545497],
                                  [0.07013606, 0.07567424],
                                  [0.06964085, 0.07606316],
                                  [0.06907539, 0.07639726],
                                  [0.06847447, 0.07646689],
                                  [0.06786159, 0.076308],
                                  [0.06741153, 0.07588963],
                                  [0.06710359, 0.07538889],
                                  [0.06693839, 0.07505855],
                                  [0.07077119, 0.07541107],
                                  [0.07028192, 0.07553016],
                                  [0.06997281, 0.07565057],
                                  [0.06965169, 0.07578998],
                                  [0.06932068, 0.07592123],
                                  [0.06892659, 0.07597831],
                                  [0.0683922, 0.07582324],
                                  [0.06792671, 0.07558156],
                                  [0.0675042, 0.07534475],
                                  [0.06722414, 0.07515131],
                                  [0.07024945, 0.07536358],
                                  [0.06993028, 0.07539681],
                                  [0.06958515, 0.07550832],
                                  [0.06922492, 0.07564673],
                                  [0.06881551, 0.07572436],
                                  [0.06829971, 0.07557193],
                                  [0.06784604, 0.07534362],
                                  [0.06745424, 0.0751317],
                                  [0.06718505, 0.07501231],
                                  [0.06692216, 0.07496734]]),
            'hand_mean': np.array([[0.7567972, 0.6644172],
                                  [0.7086355, 0.62574154],
                                 [0.6586791, 0.58655035],
                                 [0.61850184, 0.56251407],
                                 [0.5904696, 0.548389],
                                 [0.6667918, 0.54283553],
                                 [0.6064103, 0.51566076],
                                 [0.57769305, 0.5134393],
                                 [0.5612646, 0.5131613],
                                 [0.67869467, 0.5552976],
                                 [0.60758406, 0.53667927],
                                 [0.5856956, 0.5441141],
                                 [0.57800967, 0.5484671],
                                 [0.6903964, 0.57661325],
                                 [0.6234671, 0.5645831],
                                 [0.60759574, 0.5736489],
                                 [0.6041455, 0.5780031],
                                 [0.70143384, 0.6028143],
                                 [0.64991665, 0.5941722],
                                 [0.6366553, 0.59798354],
                                 [0.6327495, 0.5991175]]),
            'hand_std': np.array([[0.09937678, 0.12479375],
                                  [0.10329078, 0.12219299],
                                  [0.10760751, 0.12627973],
                                  [0.11237925, 0.13432714],
                                  [0.121261, 0.14325653],
                                  [0.11182029, 0.13598064],
                                  [0.12011228, 0.1525326],
                                  [0.12379488, 0.16460694],
                                  [0.12913615, 0.17510319],
                                  [0.11235697, 0.1442134],
                                  [0.12290164, 0.16327673],
                                  [0.1244994, 0.17435057],
                                  [0.13011923, 0.18345757],
                                  [0.11747714, 0.15215328],
                                  [0.12870327, 0.16923246],
                                  [0.1288817, 0.17655136],
                                  [0.13330464, 0.1828854],
                                  [0.12712607, 0.1587104],
                                  [0.13840523, 0.17165326],
                                  [0.13911052, 0.17733677],
                                  [0.14233603, 0.1823519]]),
            'pose_mean': np.array([[0.96710956, 0.8804063],
                                    [0.7624413, 0.67838985],
                                    [0.7107083, 0.6183697],
                                    [0.67900175, 0.58068377],
                                    [0.6784725, 0.6106137]]),
            'pose_std': np.array([[0.10719363, 0.09366401],
                                  [0.09262041, 0.13212024],
                                  [0.10971711, 0.1551326],
                                  [0.10008331, 0.14517322],
                                  [0.09255257, 0.13572161]])}
model = get_model(MEAN_STD)
model.load_weights(
    '/home/nastya/sign_language/GISLR/git/RecognitionofSignLanguage/checkpoints/transformer_checkpoints/data_frame_22_more_landmarks_False_depth_False_new/Ansamble/Embedding_Smooth_new_data/best_model.hdf5')

def preprocess(points):
    print("Start preprocess")
    test, non_empty_frames_idxs = preprocess_layer(points)
    print("Finish preprocess")
    return test, non_empty_frames_idxs

def predict(test, non_empty_frames_idxs):
    reshaped_data = np.expand_dims(test, axis=0)
    reshaped_NON_EMPTY_FRAME_IDXS_TEST= np.expand_dims(non_empty_frames_idxs, axis=0)
    y_val_pred = model.predict({ 'frames': reshaped_data, 'non_empty_frame_idxs': reshaped_NON_EMPTY_FRAME_IDXS_TEST }, verbose=2)
    return y_val_pred