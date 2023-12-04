import tensorflow as tf
from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from git.RecognitionofSignLanguage.utils.Landmarks import Landmarks as lm


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()

        if Cfg.DEPTH:
            normalisation_correction = tf.constant([
                # Add 0.50 to left hand (original right hand) and substract 0.50 of right hand (original left hand)
                [0] * len(lm.LEYE_IDXS) + [0] * len(lm.REYE_IDXS) + [0] * len(lm.NOSE_IDXS) + [0] * len(lm.LIPS_IDXS) + [0.50] * len(
                    lm.LEFT_HAND_IDXS) + [
                    0.50] * len(lm.POSE_IDXS),
                # Y coordinates stay intact
                [0] * len(lm.LANDMARK_IDXS_LEFT_DOMINANT0),
                # Z coordinates stay intact
                [0] * len(lm.LANDMARK_IDXS_LEFT_DOMINANT0),
            ],
                dtype=tf.float32,
            )
        else:
            normalisation_correction = tf.constant([
                # Add 0.50 to left hand (original right hand) and substract 0.50 of right hand (original left hand)
                [0] * len(lm.LEYE_IDXS) + [0] * len(lm.REYE_IDXS) + [0] * len(lm.NOSE_IDXS) + [0] * len(
                    lm.LIPS_IDXS) + [0.50] * len(lm.LEFT_HAND_IDXS) + [
                    0.50] * len(lm.POSE_IDXS),
                # Y coordinates stay intact
                [0] * len(lm.LANDMARK_IDXS_LEFT_DOMINANT0)
            ],
                dtype=tf.float32,
            )
        self.normalisation_correction = tf.transpose(normalisation_correction, [1, 0])

    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, Cfg.ROWS_PER_FRAME, Cfg.N_DIMS], dtype=tf.float32),),
    )
    def call(self, data0):
        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]

        # Find dominant hand by comparing summed absolute coordinates
        left_hand_sum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(data0, lm.LEFT_HAND_IDXS0, axis=1)), 0, 1))
        right_hand_sum = tf.math.reduce_sum(
            tf.where(tf.math.is_nan(tf.gather(data0, lm.RIGHT_HAND_IDXS0, axis=1)), 0, 1))
        left_dominant = left_hand_sum >= right_hand_sum

        # Count non NaN Hand values in each frame for the dominant hand
        if left_dominant:
            frames_hands_non_nan_sum = tf.math.reduce_sum(
                tf.where(tf.math.is_nan(tf.gather(data0, lm.LEFT_HAND_IDXS0, axis=1)), 0, 1),
                axis=[1, 2],
            )
        else:
            frames_hands_non_nan_sum = tf.math.reduce_sum(
                tf.where(tf.math.is_nan(tf.gather(data0, lm.RIGHT_HAND_IDXS0, axis=1)), 0, 1),
                axis=[1, 2],
            )

        # Find frames indices with coordinates of dominant hand
        non_empty_frames_idxs = tf.where(frames_hands_non_nan_sum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
        # Filter frames
        data = tf.gather(data0, non_empty_frames_idxs, axis=0)

        # Cast Indices in float32 to be compatible with Tensorflow Lite
        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32)
        # Normalize to start with 0
        non_empty_frames_idxs -= tf.reduce_min(non_empty_frames_idxs)

        # Number of Frames in Filtered Video
        N_FRAMES = tf.shape(data)[0]

        # Gather Relevant Landmark Columns
        if left_dominant:
            data = tf.gather(data, lm.LANDMARK_IDXS_LEFT_DOMINANT0, axis=1)
        else:
            data = tf.gather(data, lm.LANDMARK_IDXS_RIGHT_DOMINANT0, axis=1)
            data = (
                    self.normalisation_correction + (
                    (data - self.normalisation_correction) * tf.where(self.normalisation_correction != 0, -1.0, 1.0))
            )

        # Video fits in INPUT_SIZE
        if N_FRAMES < Cfg.INPUT_SIZE:
            # Pad With -1 to indicate padding
            non_empty_frames_idxs = tf.pad(non_empty_frames_idxs, [[0, Cfg.INPUT_SIZE - N_FRAMES]], constant_values=-1)
            # Pad Data With Zeros
            data = tf.pad(data, [[0, Cfg.INPUT_SIZE - N_FRAMES], [0, 0], [0, 0]], constant_values=0)
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Repeat
            if N_FRAMES < Cfg.INPUT_SIZE ** 2:
                repeats = tf.math.floordiv(Cfg.INPUT_SIZE * Cfg.INPUT_SIZE, N_FRAMES0)
                data = tf.repeat(data, repeats=repeats, axis=0)
                non_empty_frames_idxs = tf.repeat(non_empty_frames_idxs, repeats=repeats, axis=0)

            # Pad To Multiple Of Input Size
            pool_size = tf.math.floordiv(len(data), Cfg.INPUT_SIZE)
            if tf.math.mod(len(data), Cfg.INPUT_SIZE) > 0:
                pool_size += 1

            if pool_size == 1:
                pad_size = (pool_size * Cfg.INPUT_SIZE) - len(data)
            else:
                pad_size = (pool_size * Cfg.INPUT_SIZE) % len(data)

            # Pad Start/End with Start/End value
            pad_left = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(Cfg.INPUT_SIZE, 2)
            pad_right = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(Cfg.INPUT_SIZE, 2)
            if tf.math.mod(pad_size, 2) > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')

            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_left, 'LEFT')
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_right, 'RIGHT')

            # Reshape to Mean Pool
            data = tf.reshape(data, [Cfg.INPUT_SIZE, -1, lm.N_COLS, Cfg.N_DIMS])
            non_empty_frames_idxs = tf.reshape(non_empty_frames_idxs, [Cfg.INPUT_SIZE, -1])

            # Mean Pool
            data = tf.experimental.numpy.nanmean(data, axis=1)
            non_empty_frames_idxs = tf.experimental.numpy.nanmean(non_empty_frames_idxs, axis=1)

            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)

            return data, non_empty_frames_idxs
